"""Pre-compaction checkpoint storage.

Saves message state before compaction for recovery. Keeps last 5 checkpoints.
"""

import glob
import json
import logging
import os
from datetime import datetime
from typing import List, Optional

from qwen_agent.llm.schema import Message

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(DATA_DIR, "compaction_checkpoints")
MAX_CHECKPOINTS = 5


def save_checkpoint(messages: List[Message], label: str) -> Optional[str]:
    """Save messages to a checkpoint file. Returns path or None on failure."""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{label}.json"
        path = os.path.join(CHECKPOINT_DIR, filename)

        data = []
        for msg in messages:
            entry = {"role": msg.role}
            if isinstance(msg.content, str):
                entry["content"] = msg.content
            else:
                entry["content"] = str(msg.content)
            if msg.name:
                entry["name"] = msg.name
            if msg.function_call:
                entry["function_call"] = {
                    "name": msg.function_call.name,
                    "arguments": msg.function_call.arguments,
                }
            if msg.extra:
                entry["extra"] = msg.extra
            data.append(entry)

        with open(path, "w") as f:
            json.dump({"label": label, "timestamp": timestamp, "message_count": len(messages), "messages": data}, f)

        _prune_old_checkpoints()
        logger.info(f"Compaction checkpoint saved: {filename} ({len(messages)} messages)")
        return path
    except Exception as e:
        logger.warning(f"Failed to save compaction checkpoint: {e}")
        return None


def load_checkpoint(path: str) -> List[Message]:
    """Load messages from a checkpoint file."""
    with open(path) as f:
        data = json.load(f)

    messages = []
    for entry in data["messages"]:
        msg = Message(role=entry["role"], content=entry.get("content", ""))
        if "name" in entry:
            msg.name = entry["name"]
        if "extra" in entry:
            msg.extra = entry["extra"]
        messages.append(msg)
    return messages


def list_checkpoints() -> list:
    """List available checkpoints with metadata."""
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    results = []
    for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            results.append({
                "path": path,
                "label": data.get("label", ""),
                "timestamp": data.get("timestamp", ""),
                "message_count": data.get("message_count", 0),
            })
        except Exception:
            pass
    return results


def _prune_old_checkpoints():
    """Keep only the most recent MAX_CHECKPOINTS files."""
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.json")))
    while len(files) > MAX_CHECKPOINTS:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
        except Exception:
            pass
