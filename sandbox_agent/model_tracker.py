"""Real-time status tracker — writes agent_status.json for the dashboard.

Tracks models, current task, current tool, and agent status in real-time.
Updated at the START of each action, not after completion.
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Optional

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_STATUS_FILE = os.path.join(DATA_DIR, "agent_status.json")

_state = {
    "agent_status": "idle",
    "current_task": None,
    "current_tool": None,
    "models": {},
    "started_at": None,
}

_start_time = datetime.now()


def model_start(model: str, task: str) -> None:
    """Mark a model as busy with a task."""
    with _lock:
        _state["models"][model] = {
            "status": "busy",
            "task": task[:200],
            "since": datetime.now().isoformat(),
        }
        _write()


def model_done(model: str) -> None:
    """Mark a model as idle."""
    with _lock:
        _state["models"][model] = {
            "status": "idle",
            "task": None,
            "since": None,
        }
        _write()


def set_agent_status(status: str = None, current_task: str = None, current_tool: str = None) -> None:
    """Update the agent's overall status."""
    with _lock:
        if status is not None:
            _state["agent_status"] = status
        if current_task is not None:
            _state["current_task"] = current_task
        if current_tool is not None:
            _state["current_tool"] = current_tool
        _state["started_at"] = datetime.now().isoformat()
        _write()


def clear_agent_status() -> None:
    """Reset agent to idle."""
    with _lock:
        _state["agent_status"] = "idle"
        _state["current_task"] = None
        _state["current_tool"] = None
        _state["started_at"] = None
        _write()


def set_current_tool(tool_name: Optional[str]) -> None:
    """Update just the current tool (called frequently during tool loops)."""
    with _lock:
        _state["current_tool"] = tool_name
        _write()


def _write() -> None:
    """Write current state to disk for the dashboard process."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        data = dict(_state)
        data["uptime_seconds"] = int((datetime.now() - _start_time).total_seconds())
        data["updated_at"] = datetime.now().isoformat()
        with open(_STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def read_status_from_file() -> dict:
    """Read full status from file (used by dashboard process)."""
    try:
        if os.path.exists(_STATUS_FILE):
            with open(_STATUS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "agent_status": "idle",
        "current_task": None,
        "current_tool": None,
        "models": {},
        "uptime_seconds": 0,
        "updated_at": None,
    }
