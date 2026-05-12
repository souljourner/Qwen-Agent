"""Cooperative run cancellation — Hermes-Agent-style interrupt for agent runs.

A "run" is one `agent.run(...)` invocation, executed on a single thread (the
cron worker, or a chat agent thread). While a run is active it registers a
handle holding:
  * a `threading.Event` — the cancel flag
  * the set of child process-group ids spawned (by `exec` / `code_interpreter`)
    during the run

`cancel(run_id)` does two things, mirroring Hermes:
  1. sets the event — the run's generator wrapper (`cancellation.guard`, used by
     `_compacting_run`) raises `RunCancelled` at the next yield point, so the
     run unwinds and its locks are released;
  2. SIGKILLs every registered child process group — a tool call wedged inside a
     subprocess returns immediately, so the loop reaches that next yield point.

What this does NOT do: make an in-flight LLM HTTP call itself interruptible — a
stuck inference still has to time out on its own (bounded by `request_timeout`).
Hermes wraps the HTTP call on a background thread to dodge that; out of scope here.

Python can't force-kill a thread, so cancellation is cooperative — but every
yield point in the agent loop (between tool calls, between streamed chunks) is a
checkpoint, and the only thing that can block longer than a request timeout is a
subprocess, which we do kill.
"""

import logging
import os
import signal
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Set

logger = logging.getLogger(__name__)


class RunCancelled(Exception):
    """Raised inside a run's generator when its cancel event has been set."""

    def __init__(self, run_id: str):
        super().__init__(f"run {run_id!r} was cancelled")
        self.run_id = run_id


class _RunHandle:
    __slots__ = ("run_id", "cancel_event", "child_pgids", "lock")

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.cancel_event = threading.Event()
        self.child_pgids: Set[int] = set()
        self.lock = threading.Lock()


_REGISTRY: "dict[str, _RunHandle]" = {}
_REGISTRY_LOCK = threading.Lock()
_local = threading.local()  # _local.run_id — the run id the current thread is executing


def current_run_id() -> Optional[str]:
    return getattr(_local, "run_id", None)


def _current_handle() -> Optional[_RunHandle]:
    rid = current_run_id()
    if rid is None:
        return None
    with _REGISTRY_LOCK:
        return _REGISTRY.get(rid)


def is_cancelled() -> bool:
    """True if the run executing on THIS thread has been cancelled."""
    h = _current_handle()
    return bool(h and h.cancel_event.is_set())


def check_cancelled() -> None:
    """Raise `RunCancelled` if this thread's run has been cancelled."""
    h = _current_handle()
    if h is not None and h.cancel_event.is_set():
        raise RunCancelled(h.run_id)


def guard(it: Iterator) -> Iterator:
    """Wrap a generator so it raises `RunCancelled` at every yield point once
    the current thread's run is cancelled. `_compacting_run` wraps the agent's
    run generator with this."""
    check_cancelled()
    for item in it:
        check_cancelled()
        yield item


@contextmanager
def begin_run(run_id: str):
    """Register a cancellable run handle for the duration of the block, and tag
    the current thread with `run_id` so `exec` / `code_interpreter` can find the
    handle to register their child process groups against."""
    handle = _RunHandle(run_id)
    with _REGISTRY_LOCK:
        _REGISTRY[run_id] = handle  # a stale handle for this id (if any) is replaced
    prev = getattr(_local, "run_id", None)
    _local.run_id = run_id
    try:
        yield handle
    finally:
        _local.run_id = prev
        with _REGISTRY_LOCK:
            # Only remove our own handle — if the id was re-registered by a
            # newer run (e.g. a re-queued task), leave that one alone.
            if _REGISTRY.get(run_id) is handle:
                del _REGISTRY[run_id]


def register_child_pgid(pgid: int) -> None:
    """Record a subprocess process-group id against the current thread's run so
    `cancel()` can SIGKILL it. No-op when there's no active run on this thread
    (e.g. a tool invoked directly in a test)."""
    h = _current_handle()
    if h is None:
        return
    with h.lock:
        h.child_pgids.add(pgid)


def unregister_child_pgid(pgid: int) -> None:
    h = _current_handle()
    if h is None:
        return
    with h.lock:
        h.child_pgids.discard(pgid)


def cancel(run_id: str) -> bool:
    """Cancel the run with this id: set its event and SIGKILL its registered
    child process groups. Returns True iff a matching active run was found."""
    with _REGISTRY_LOCK:
        h = _REGISTRY.get(run_id)
    if h is None:
        return False
    h.cancel_event.set()
    with h.lock:
        pgids = list(h.child_pgids)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
            logger.info("cancel(%s): SIGKILL'd child process group %d", run_id, pgid)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return True


def is_active(run_id: str) -> bool:
    with _REGISTRY_LOCK:
        return run_id in _REGISTRY
