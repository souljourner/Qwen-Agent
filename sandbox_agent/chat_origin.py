"""Per-thread "current chat origin" tag — `{session_id, thread_id}` that the
worker thread running an `agent.run()` carries so tools like `schedule_task` /
`start_pipeline` can stamp the originating chat onto any task they create.

When the task later completes, the notifier (chat_app._completion_notifier_loop)
reads `task.origin` and delivers the completion notice (and triggers the
synthetic follow-up agent turn) only to the matching session — instead of
broadcasting to every connected tab, which would fire redundant agent runs.

Mirror of `sandbox_agent.cancellation`'s threading.local pattern. Set in
chat_app._run_agent_in_thread (the worker thread for a chat turn) and in
main._run_cron_task (so a cron task originated by a chat propagates its origin
to any sub-tasks it spawns). Cron / heartbeat runs that have no chat origin
just leave it unset; tools read `current_origin()` → None → don't stamp.
"""

import threading
from typing import Optional

_local = threading.local()


def set_current_origin(origin: Optional[dict]) -> None:
    """Tag the current thread with an origin dict (or None to clear). Called by
    chat_app._run_agent_in_thread at the top, main._run_cron_task on entry."""
    _local.origin = origin


def current_origin() -> Optional[dict]:
    """Read the current thread's origin tag. None if not set (cron without a
    chat origin, heartbeat, REPL, tests)."""
    return getattr(_local, "origin", None)
