"""Append-only raw-history archive.

Persisted compaction is destructive: the compacted list replaces the
working history, and the sidecar is the ONLY agent-consumable copy (once a
sidecar exists, chat_app never falls back to rebuilding from chat.db). So
nothing may be destroyed until the destroyed part is safely on disk.

Only the DELTA is archived — the messages this compaction actually removed
— not a fresh copy of the whole history each time (that would be O(n²)
across a long conversation). Replaying every segment in order and appending
the current sidecar reconstructs the full raw conversation.

Mirrors the archive-then-compact pattern already used for MEMORIES.md.
"""

import json
import logging
import os
import time
from typing import List

from qwen_agent.llm.schema import Message

from sandbox_agent.config import COMPACTION_ARCHIVE_MAX_BYTES, DATA_DIR

logger = logging.getLogger(__name__)


def _archive_path(thread_id: str) -> str:
    from sandbox_agent.chat_history import _path
    return _path(thread_id).replace(".json", ".raw.jsonl")


def archive_append(thread_id: str, messages: List[Message]) -> bool:
    """Append destroyed messages as one JSONL segment.

    Returns True only if the bytes are durably written — the caller MUST
    NOT destroy anything when this returns False.
    """
    if not thread_id:
        return False
    if not messages:
        return True  # nothing to preserve; destroying nothing is safe
    path = _archive_path(thread_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > COMPACTION_ARCHIVE_MAX_BYTES:
            logger.error("Raw archive for %s exceeds %d bytes — refusing to "
                         "destroy more history", thread_id, COMPACTION_ARCHIVE_MAX_BYTES)
            return False
        segment = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": len(messages),
            "messages": [m.model_dump(mode="json") for m in messages],
        }
        with open(path, "a") as f:
            f.write(json.dumps(segment) + "\n")
            f.flush()
            os.fsync(f.fileno())
        logger.info("Archived %d raw messages for thread %s", len(messages), thread_id)
        return True
    except Exception:  # noqa: BLE001 — a failed archive must block compaction
        logger.exception("Raw archive failed for %s — compaction must not proceed", thread_id)
        return False


def load_archive(thread_id: str) -> List[Message]:
    """Replay every archived segment in order (oldest first)."""
    path = _archive_path(thread_id)
    out: List[Message] = []
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for raw in json.loads(line).get("messages", []):
                    try:
                        out.append(Message(**raw))
                    except Exception:  # noqa: BLE001
                        continue
    except Exception:  # noqa: BLE001
        logger.exception("Could not read raw archive for %s", thread_id)
    return out
