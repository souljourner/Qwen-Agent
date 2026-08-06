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

Appends are idempotent at the leading edge. If the sidecar is ever lost or
corrupted, resume rebuilds the FULL raw conversation from chat.db, and the
next compaction would re-archive messages already on disk — so replaying
the segments would double-count them. Rather than trust that never to
happen, `archive_append` skips the leading run of messages it has already
archived. That also covers a retried or double-run compaction.
"""

import hashlib
import json
import logging
import os
import time
from typing import List, Set

from qwen_agent.llm.schema import Message

from sandbox_agent.config import COMPACTION_ARCHIVE_MAX_BYTES, DATA_DIR

logger = logging.getLogger(__name__)


def _archive_path(thread_id: str) -> str:
    from sandbox_agent.chat_history import _path
    return _path(thread_id).replace(".json", ".raw.jsonl")


def _fingerprint(msg) -> str:
    """Stable identity for a message, independent of object identity."""
    if hasattr(msg, "model_dump"):
        d = msg.model_dump(mode="json")
    else:
        d = dict(msg or {})
    fc = d.get("function_call") or {}
    parts = [str(d.get("role") or ""), str(d.get("name") or ""),
             str(d.get("content") or ""),
             str(fc.get("name") or ""), str(fc.get("arguments") or "")]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def _archived_fingerprints(path: str) -> Set[str]:
    out: Set[str] = set()
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for raw in json.loads(line).get("messages", []):
                    out.add(_fingerprint(raw))
    except Exception:  # noqa: BLE001 — unreadable archive: dedupe nothing
        logger.exception("Could not read archive fingerprints for %s", path)
    return out


def _strip_already_archived(messages: List[Message], known: Set[str]) -> List[Message]:
    """Drop the LEADING run of messages already on disk.

    Leading-only, deliberately: a re-archive after a chat.db rebuild repeats
    a PREFIX of the conversation. Deduping every position would collapse
    genuinely repeated messages (a user saying "yes" twice) into one.
    """
    i = 0
    while i < len(messages) and _fingerprint(messages[i]) in known:
        i += 1
    return messages[i:]


def archive_append(thread_id: str, messages: List[Message]) -> bool:
    """Append destroyed messages as one JSONL segment.

    Returns True only if the destroyed messages are durably on disk — the
    caller MUST NOT destroy anything when this returns False. Messages
    already archived count as preserved, so skipping them still returns True.
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

        original_count = len(messages)
        messages = _strip_already_archived(messages, _archived_fingerprints(path))
        skipped = original_count - len(messages)
        if skipped:
            logger.info("Archive for %s: %d of %d messages already archived "
                        "(sidecar was likely rebuilt from chat.db) — skipping them",
                        thread_id, skipped, original_count)
        if not messages:
            return True  # everything already preserved

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
    """Replay every archived segment in order (oldest first).

    Defensive mirror of the write-time rule: a segment's LEADING run of
    already-seen messages is skipped, so legacy segments written before
    write-time dedupe still replay cleanly. Deliberately not a global
    per-message dedupe — that would collapse genuinely repeated messages
    (a user saying "yes" twice) into one.
    """
    path = _archive_path(thread_id)
    out: List[Message] = []
    seen: Set[str] = set()
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raws = json.loads(line).get("messages", [])
                start = 0
                while start < len(raws) and _fingerprint(raws[start]) in seen:
                    start += 1
                for raw in raws[start:]:
                    seen.add(_fingerprint(raw))
                    try:
                        out.append(Message(**raw))
                    except Exception:  # noqa: BLE001
                        continue
    except Exception:  # noqa: BLE001
        logger.exception("Could not read raw archive for %s", thread_id)
    return out
