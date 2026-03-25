"""Task state checkpointing for long-running tasks."""

import json
import os
from typing import Any, Dict, Optional

from sandbox_agent.config import DATA_DIR


def _checkpoint_path(task_id: str) -> str:
    checkpoint_dir = os.path.join(DATA_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{task_id}.json")


def save_checkpoint(task_id: str, step: int, state: Dict[str, Any]) -> None:
    """Persist intermediate state for a long-running task."""
    data = {"task_id": task_id, "step": step, "state": state}
    path = _checkpoint_path(task_id)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_checkpoint(task_id: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint for a task. Returns None if no checkpoint exists."""
    path = _checkpoint_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def delete_checkpoint(task_id: str) -> None:
    """Remove checkpoint file after task completes."""
    path = _checkpoint_path(task_id)
    if os.path.exists(path):
        os.remove(path)
