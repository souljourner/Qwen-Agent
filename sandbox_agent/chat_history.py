"""Per-thread agent-history sidecar — the fix for reload amnesia.

chat.db persists what the USER saw; this persists what the AGENT knew — the
exact qwen-agent Message list including tool-call/result pairs and multimodal
content. on_chat_resume loads it verbatim, so a page reload or container
restart no longer costs the agent its gathered evidence (previously ~200k
tokens of tool results per long thread were dropped, leaving a ~44k text-only
context). Growth is managed by the existing per-turn compactor, not here.
"""

import json
import logging
import os
from typing import List, Optional

from qwen_agent.llm.schema import Message

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)


def _path(thread_id: str) -> str:
    safe = "".join(c for c in thread_id if c.isalnum() or c in "-_")
    return os.path.join(DATA_DIR, "chat_history", f"{safe}.json")


def save_history(thread_id: str, history: List[Message]) -> None:
    """Atomically persist the agent-side history for a thread. Best-effort —
    a failed save must never break the turn (the sidecar is an optimization;
    chat.db reconstruction remains the fallback)."""
    if not thread_id:
        return
    try:
        path = _path(thread_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m
                   for m in history]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        logger.exception("chat_history: failed to save sidecar for %s", thread_id)


def load_history(thread_id: str) -> Optional[List[Message]]:
    """Load the persisted agent history, or None (missing/corrupt → caller
    falls back to chat.db reconstruction)."""
    if not thread_id:
        return None
    try:
        with open(_path(thread_id)) as f:
            payload = json.load(f)
        return [Message(**m) for m in payload]
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        logger.exception("chat_history: corrupt sidecar for %s — falling back", thread_id)
        return None
