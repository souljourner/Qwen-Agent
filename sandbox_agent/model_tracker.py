"""Real-time model status tracker — writes model_status.json for the dashboard.

Tracks what each model is currently doing. Updated at the START of each model call,
not after completion. Supports concurrent model usage.
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
_STATUS_FILE = os.path.join(DATA_DIR, "model_status.json")

# Current state of each model
_models = {}


def model_start(model: str, task: str) -> None:
    """Mark a model as busy with a task."""
    with _lock:
        _models[model] = {
            "status": "busy",
            "task": task[:200],
            "since": datetime.now().isoformat(),
        }
        _write()


def model_done(model: str) -> None:
    """Mark a model as idle."""
    with _lock:
        _models[model] = {
            "status": "idle",
            "task": None,
            "since": None,
        }
        _write()


def get_model_status() -> dict:
    """Get the current status of all models."""
    with _lock:
        return dict(_models)


def _write() -> None:
    """Write current state to disk for the dashboard process."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            "models": dict(_models),
            "updated_at": datetime.now().isoformat(),
        }
        with open(_STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass  # Best effort — don't crash on write failure


def read_model_status_from_file() -> dict:
    """Read model status from file (used by dashboard process)."""
    try:
        if os.path.exists(_STATUS_FILE):
            with open(_STATUS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {"models": {}, "updated_at": None}
