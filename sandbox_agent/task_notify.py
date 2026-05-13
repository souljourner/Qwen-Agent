"""In-memory inbox of "background task finished" events.

The cron lane pushes a record here when a scheduled task / pipeline stage
completes or fails; the chat surface (`chat_app`'s notifier loop) drains it and
pushes a notice into the live Chainlit session — a Hermes-style completion alert,
so the agent/user learns a job finished without having to ask. If no session is
connected the events sit in the queue and are delivered as soon as one is.

This is intentionally tiny: a thread-safe queue + a producer + a drain. Routing
to the UI lives in `chat_app`; nothing here knows about Chainlit.
"""

import queue
import time
from typing import Dict, List

_events: "queue.Queue[Dict]" = queue.Queue()


def notify_task_done(task_id: str, name: str, result: str = "", *, source: str = "cron", ok: bool = True) -> None:
    """Record that a background task finished. Thread-safe — called from the
    cron worker thread.

    Args:
        task_id: the task's id.
        name: the task name (e.g. "pipeline: moon-ai stage 3" or a user task name).
        result: a short result/summary (truncated here to 1000 chars).
        source: "cron" | "pipeline" — for display/routing.
        ok: True on success, False if the task failed.
    """
    _events.put({
        "task_id": task_id,
        "name": name,
        "result": (result or "")[:1000],
        "ts": time.time(),
        "source": source,
        "ok": ok,
    })


def drain() -> List[Dict]:
    """Pop all pending completion events in FIFO order. Thread-safe."""
    out: List[Dict] = []
    while True:
        try:
            out.append(_events.get_nowait())
        except queue.Empty:
            break
    return out


def pending_count() -> int:
    return _events.qsize()
