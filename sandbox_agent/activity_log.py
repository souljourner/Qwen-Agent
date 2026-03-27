"""Structured activity logging — writes events to activity.jsonl and tracks current state."""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_recent_events: deque = deque(maxlen=500)  # Last 500 events in memory
_current_state = {
    "status": "idle",
    "current_task": None,
    "current_tool": None,
    "model_in_use": None,
    "started_at": None,
    "uptime_start": time.time(),
}


def _activity_path() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, "activity.jsonl")


def log_event(
    event_type: str,
    detail: str = "",
    tool_name: Optional[str] = None,
    tool_args: Optional[str] = None,
    tool_result: Optional[str] = None,
    task_id: Optional[str] = None,
    task_name: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Log a structured activity event."""
    event = {
        "ts": datetime.now().isoformat(),
        "type": event_type,
        "detail": detail[:500],
    }
    if tool_name:
        event["tool"] = tool_name
    if tool_args:
        event["args"] = tool_args[:300]
    if tool_result:
        event["result"] = tool_result[:300]
    if task_id:
        event["task_id"] = task_id
    if task_name:
        event["task_name"] = task_name
    if model:
        event["model"] = model

    with _lock:
        _recent_events.append(event)

    # Append to file (best-effort)
    try:
        with open(_activity_path(), "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


def set_state(
    status: Optional[str] = None,
    current_task: Optional[str] = None,
    current_tool: Optional[str] = None,
    model_in_use: Optional[str] = None,
) -> None:
    """Update the current agent state."""
    with _lock:
        if status is not None:
            _current_state["status"] = status
        if current_task is not None:
            _current_state["current_task"] = current_task
        if current_tool is not None:
            _current_state["current_tool"] = current_tool
        if model_in_use is not None:
            _current_state["model_in_use"] = model_in_use
        _current_state["started_at"] = datetime.now().isoformat()


def clear_state() -> None:
    """Reset to idle state."""
    with _lock:
        _current_state["status"] = "idle"
        _current_state["current_task"] = None
        _current_state["current_tool"] = None
        _current_state["model_in_use"] = None
        _current_state["started_at"] = None


def get_state() -> dict:
    """Get the current agent state snapshot."""
    with _lock:
        state = dict(_current_state)
        state["uptime_seconds"] = int(time.time() - _current_state["uptime_start"])
        return state


def get_recent_events(n: int = 20) -> list:
    """Get the N most recent activity events."""
    with _lock:
        return list(_recent_events)[-n:]


def get_status_summary() -> dict:
    """Full status summary for the /status endpoint and Gradio sidebar."""
    state = get_state()
    recent = get_recent_events(100)

    # Count events by type in last 500
    all_events = get_recent_events(500)
    type_counts = {}
    for e in all_events:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Recent tool calls
    recent_tools = [e for e in recent if e.get("type") == "tool_call"][-50:]

    return {
        "state": state,
        "event_counts": type_counts,
        "recent_events": recent,
        "recent_tools": recent_tools,
    }
