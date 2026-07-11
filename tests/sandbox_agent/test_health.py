"""Failure-monitoring loop: silent degradation must surface automatically.

Motivating incidents: QueuePool errors repeated for days unnoticed; a
request_user item (agent asking for SMTP creds) sat unread for 3 weeks.
health.run_health_check() detects failure bursts + stale requests, emails the
user deterministically (no LLM in the trigger path), dedups repeats, and
feeds items into the hourly heartbeat for agent-side investigation.
"""

import json
import os
from datetime import datetime, timedelta

import pytest

import sandbox_agent.health as health


def _write_events(data_dir, events):
    with open(os.path.join(data_dir, "activity.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _evt(type_, hours_ago=1.0, **kw):
    ts = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
    return {"ts": ts, "type": type_, **kw}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DATA_DIR", str(tmp_path))
    sent = []
    monkeypatch.setattr(health, "send_email_message",
                        lambda subject, body, to=None: sent.append((subject, body)) or "Email sent")
    return tmp_path, sent


def test_failure_burst_alerts(env):
    d, _ = env
    _write_events(d, [_evt("cron_failed") for _ in range(4)])
    alerts = health.collect_health_alerts(window_hours=24)
    assert any("cron_failed" in a for a in alerts)


def test_below_threshold_quiet(env):
    d, _ = env
    _write_events(d, [_evt("cron_failed"), _evt("cron_failed")])
    assert health.collect_health_alerts(window_hours=24) == []


def test_old_failures_outside_window_ignored(env):
    d, _ = env
    _write_events(d, [_evt("cron_failed", hours_ago=60) for _ in range(10)])
    assert health.collect_health_alerts(window_hours=24) == []


def test_corrupt_queue_always_alerts(env):
    d, _ = env
    _write_events(d, [_evt("task_queue_corrupt")])
    alerts = health.collect_health_alerts(window_hours=24)
    assert any("task_queue_corrupt" in a for a in alerts)


def test_stale_pending_request_alerts(env):
    d, _ = env
    _write_events(d, [])
    old = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M")
    fresh = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(d, "agent_requests.json"), "w") as f:
        json.dump([
            {"id": "a1", "subject": "SMTP password needed", "status": "pending",
             "created": old, "urgency": "high", "detail": "", "project": "", "resolved_at": None},
            {"id": "a2", "subject": "fresh one", "status": "pending",
             "created": fresh, "urgency": "low", "detail": "", "project": "", "resolved_at": None},
        ], f)
    alerts = health.collect_health_alerts(window_hours=24)
    assert any("SMTP password needed" in a for a in alerts)
    assert not any("fresh one" in a for a in alerts)


def test_run_health_check_emails_and_dedups(env):
    d, sent = env
    _write_events(d, [_evt("cron_stuck") for _ in range(5)])

    first = health.run_health_check(window_hours=24)
    assert first, "alerts expected"
    assert len(sent) == 1
    assert "health" in sent[0][0].lower()
    assert "cron_stuck" in sent[0][1]

    # Same condition again within the min interval → suppressed entirely.
    second = health.run_health_check(window_hours=24)
    assert second == []
    assert len(sent) == 1


def test_email_failure_does_not_raise(env, monkeypatch):
    d, _ = env
    _write_events(d, [_evt("cron_stuck") for _ in range(5)])
    monkeypatch.setattr(health, "send_email_message",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("smtp down")))
    alerts = health.run_health_check(window_hours=24)   # must not raise
    assert alerts


def test_heartbeat_includes_health_items(tmp_path, monkeypatch):
    import sandbox_agent.heartbeat.heartbeat_runner as hr
    from sandbox_agent.heartbeat.heartbeat_runner import HeartbeatRunner

    (tmp_path / "HEARTBEAT.md").write_text("# Heartbeat Checklist\n")
    calls = []
    runner = HeartbeatRunner(runner=lambda msgs: calls.append(msgs) or [],
                             task_queue=type("Q", (), {"get_due_tasks": lambda self: []})(),
                             data_dir=str(tmp_path))
    monkeypatch.setattr(hr, "run_health_check", lambda **k: ["3 cron_failed events in 24h"])
    runner.run_once()
    assert calls, "health alerts must force a heartbeat run"
    text = str(calls[0][0].content)
    assert "System Health" in text
    assert "cron_failed" in text


def test_selftest_failure_alerts_immediately(env):
    d, _ = env
    _write_events(d, [_evt("selftest_failed", detail="2 failed")])
    alerts = health.collect_health_alerts(window_hours=24)
    assert any("selftest_failed" in a for a in alerts)
