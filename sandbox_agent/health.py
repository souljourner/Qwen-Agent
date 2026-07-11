"""Deterministic system-health monitoring — silent failures must surface.

Motivating incidents: the chat data layer logged QueuePool errors for days
before a human noticed; a request_user item (the agent asking for SMTP creds)
sat unread for three weeks. This module scans the durable signals — the
activity log, pending user requests, exhausted tasks — and when something
crosses a threshold it (a) emails the user directly (code path, no LLM
judgment in the trigger) and (b) hands the items to the hourly heartbeat so
the agent investigates. Alerts are fingerprint-deduped so a persistent
condition nags at most once per HEALTH_ALERT_MIN_INTERVAL_S.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import List

from sandbox_agent.config import DATA_DIR
from sandbox_agent.tools.notification_tools import send_email_message

logger = logging.getLogger(__name__)

# Event types that indicate degradation, with per-window alert thresholds.
# task_queue_corrupt threshold 1: a quarantined queue is always alert-worthy.
FAILURE_EVENT_THRESHOLDS = {
    "cron_failed": 3,
    "cron_stuck": 3,
    "cron_empty_result": 3,
    "task_queue_corrupt": 1,
    "blocked_exec": 3,
}
HEALTH_STALE_REQUEST_HOURS = int(os.environ.get("HEALTH_STALE_REQUEST_HOURS", 24))
HEALTH_ALERT_MIN_INTERVAL_S = int(os.environ.get("HEALTH_ALERT_MIN_INTERVAL_S", 86_400))
_TAIL_BYTES = 2_000_000  # scan at most the last ~2MB of activity.jsonl


def _recent_event_counts(window_hours: float, data_dir: str) -> dict:
    """Count failure-type events within the window, reading the activity-log
    tail directly (the in-memory ring in activity_log doesn't survive
    restarts, and degradation often spans them)."""
    path = os.path.join(data_dir, "activity.jsonl")
    counts: dict = {}
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > _TAIL_BYTES:
                f.seek(size - _TAIL_BYTES)
                f.readline()  # skip the partial line
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return counts
    cutoff = (datetime.now() - timedelta(hours=window_hours)).isoformat()
    for line in raw.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        etype = e.get("type") or e.get("event")
        if etype not in FAILURE_EVENT_THRESHOLDS:
            continue
        ts = str(e.get("ts") or e.get("timestamp") or "")
        if ts >= cutoff:
            counts[etype] = counts.get(etype, 0) + 1
    return counts


def _stale_pending_requests(data_dir: str) -> List[dict]:
    path = os.path.join(data_dir, "agent_requests.json")
    try:
        with open(path) as f:
            rows = json.load(f)
    except (OSError, ValueError):
        return []
    cutoff = datetime.now() - timedelta(hours=HEALTH_STALE_REQUEST_HOURS)
    stale = []
    for r in rows:
        if r.get("status") != "pending":
            continue
        try:
            created = datetime.strptime(r.get("created", ""), "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if created < cutoff:
            stale.append(r)
    return stale


def collect_health_alerts(window_hours: float = 24, data_dir: str = None) -> List[str]:
    """Return human-readable alert strings for every threshold crossed."""
    data_dir = data_dir or DATA_DIR
    alerts: List[str] = []
    counts = _recent_event_counts(window_hours, data_dir)
    for etype, n in sorted(counts.items()):
        if n >= FAILURE_EVENT_THRESHOLDS[etype]:
            alerts.append(f"{n} `{etype}` events in the last {int(window_hours)}h "
                          f"(threshold {FAILURE_EVENT_THRESHOLDS[etype]})")
    for r in _stale_pending_requests(data_dir):
        alerts.append(
            f"request_user item unanswered for >{HEALTH_STALE_REQUEST_HOURS}h: "
            f"[{r.get('urgency', '?').upper()}] {r.get('subject', '?')} "
            f"(id {r.get('id', '?')}, filed {r.get('created', '?')})")
    return alerts


def _load_stamps(data_dir: str) -> dict:
    try:
        with open(os.path.join(data_dir, "health_alerts.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_stamps(stamps: dict, data_dir: str) -> None:
    try:
        with open(os.path.join(data_dir, "health_alerts.json"), "w") as f:
            json.dump(stamps, f, indent=2)
    except OSError:
        logger.exception("health: cannot persist alert stamps")


def _fingerprint(alert: str) -> str:
    # Stable across changing counts/timestamps: keep the identifying tokens.
    return "".join(c for c in alert if not c.isdigit())[:160]


def run_health_check(window_hours: float = 24, data_dir: str = None) -> List[str]:
    """Collect alerts, email NEW ones (deduped by fingerprint within
    HEALTH_ALERT_MIN_INTERVAL_S), and return the new alerts for the heartbeat.
    Never raises."""
    data_dir = data_dir or DATA_DIR
    try:
        alerts = collect_health_alerts(window_hours, data_dir)
    except Exception:  # noqa: BLE001
        logger.exception("health: collect failed")
        return []
    if not alerts:
        return []

    now = time.time()
    stamps = _load_stamps(data_dir)
    fresh = [a for a in alerts
             if now - stamps.get(_fingerprint(a), 0) > HEALTH_ALERT_MIN_INTERVAL_S]
    if not fresh:
        return []
    for a in fresh:
        stamps[_fingerprint(a)] = now
    _save_stamps(stamps, data_dir)

    body = ("The sandbox agent's health check found:\n\n"
            + "\n".join(f"- {a}" for a in fresh)
            + "\n\nThe hourly heartbeat has been asked to investigate; check the "
              "dashboard (`/status`) and `list_tasks` / `view_requests` for detail.")
    try:
        result = send_email_message(
            subject=f"[sandbox-agent] health alert: {len(fresh)} issue(s)", body=body)
        logger.warning("health: %d alert(s) — email: %s", len(fresh), result)
    except Exception:  # noqa: BLE001 — alerting must never break the caller
        logger.exception("health: alert email failed")
    return fresh
