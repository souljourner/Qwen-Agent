"""Daily digest — rolling 3-day summary of agent activity, updated each heartbeat.

Writes to DATA_DIR/digest/YYYY-MM-DD.md with line items per project per heartbeat.
The digest is always available at DATA_DIR/digest/latest.md (symlink-like copy of today's).
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

DIGEST_DIR = os.path.join(DATA_DIR, "digest")


def _ensure_dir():
    os.makedirs(DIGEST_DIR, exist_ok=True)


def _digest_path(date_str: Optional[str] = None) -> str:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(DIGEST_DIR, f"{date_str}.md")


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def add_digest_entry(
    project: Optional[str] = None,
    task_name: Optional[str] = None,
    summary: str = "",
    source: str = "heartbeat",
) -> None:
    """Append a line item to today's digest.

    Args:
        project: Project name (or None for global tasks)
        task_name: Task that generated this entry
        summary: One-line description of what happened
        source: heartbeat, cron, chat, etc.
    """
    _ensure_dir()
    path = _digest_path()
    timestamp = datetime.now().strftime("%H:%M")
    today = _today_str()

    # Create header if file is new
    is_new = not os.path.exists(path)

    with open(path, "a") as f:
        if is_new:
            f.write(f"# Daily Digest — {today}\n\n")

        project_label = f"**{project}**" if project else "**Global**"
        task_label = f" ({task_name})" if task_name else ""
        f.write(f"- `{timestamp}` {project_label}{task_label}: {summary}\n")

    # Update latest.md
    _update_latest()


def _update_latest():
    """Rebuild latest.md with rolling 3-day history."""
    _ensure_dir()
    latest_path = os.path.join(DIGEST_DIR, "latest.md")
    today = datetime.now()
    days = []

    for i in range(3):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        path = _digest_path(date_str)
        if os.path.exists(path):
            with open(path) as f:
                days.append(f.read().strip())

    with open(latest_path, "w") as f:
        f.write("# Agent Activity Digest (Last 3 Days)\n\n")
        f.write(f"*Updated: {today.strftime('%Y-%m-%d %H:%M')}*\n\n")
        if days:
            f.write("\n\n---\n\n".join(days))
            f.write("\n")
        else:
            f.write("No activity recorded yet.\n")


def cleanup_old_digests(keep_days: int = 3):
    """Remove digest files older than keep_days."""
    _ensure_dir()
    cutoff = datetime.now() - timedelta(days=keep_days)
    for fname in os.listdir(DIGEST_DIR):
        if fname == "latest.md" or not fname.endswith(".md"):
            continue
        try:
            file_date = datetime.strptime(fname.replace(".md", ""), "%Y-%m-%d")
            if file_date < cutoff:
                os.remove(os.path.join(DIGEST_DIR, fname))
                logger.info(f"Cleaned up old digest: {fname}")
        except ValueError:
            pass
