"""Tests for sandbox_agent.cancellation — cooperative run cancellation."""

import subprocess
import sys
import threading
import time

import pytest

from sandbox_agent import cancellation as cx


def test_begin_run_sets_and_clears_thread_context():
    assert cx.current_run_id() is None
    with cx.begin_run("r1"):
        assert cx.current_run_id() == "r1"
        assert cx.is_active("r1")
    assert cx.current_run_id() is None
    assert not cx.is_active("r1")


def test_is_cancelled_and_check_cancelled():
    with cx.begin_run("r2"):
        assert cx.is_cancelled() is False
        cx.check_cancelled()  # no raise
        assert cx.cancel("r2") is True
        assert cx.is_cancelled() is True
        with pytest.raises(cx.RunCancelled):
            cx.check_cancelled()


def test_cancel_unknown_run_returns_false():
    assert cx.cancel("does-not-exist") is False


def test_no_active_run_is_a_noop():
    # Outside any begin_run() these must not raise and not report cancelled.
    cx.register_child_pgid(999999)
    cx.unregister_child_pgid(999999)
    assert cx.is_cancelled() is False
    cx.check_cancelled()  # no raise


def test_guard_raises_at_next_yield_after_cancel():
    def forever():
        n = 0
        while True:
            n += 1
            yield n

    with cx.begin_run("r3"):
        g = cx.guard(forever())
        assert next(g) == 1
        assert next(g) == 2
        cx.cancel("r3")
        with pytest.raises(cx.RunCancelled):
            next(g)


def test_guard_passes_through_when_not_cancelled():
    with cx.begin_run("r4"):
        assert list(cx.guard(iter([1, 2, 3]))) == [1, 2, 3]


def test_cancel_sigkills_registered_child_pgid():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        with cx.begin_run("r5"):
            cx.register_child_pgid(proc.pid)  # start_new_session=True → pgid == pid
            assert cx.cancel("r5") is True
            # The child should be SIGKILL'd promptly.
            proc.wait(timeout=5)
            assert proc.returncode == -9  # -SIGKILL
            cx.unregister_child_pgid(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_run_ids_are_per_thread():
    seen = {}
    barrier = threading.Barrier(2)

    def worker(name):
        with cx.begin_run(name):
            barrier.wait()  # both threads inside their begin_run at once
            seen[name] = cx.current_run_id()

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start()
    t1.join(timeout=5); t2.join(timeout=5)
    assert seen == {"a": "a", "b": "b"}


def test_late_cleanup_does_not_clobber_re_registration():
    # An abandoned run's begin_run() `finally` may run after the same run id was
    # re-registered by a new run (e.g. a re-queued task on a fresh worker). The
    # late cleanup must only remove its OWN handle, not the new one.
    cm1 = cx.begin_run("dup")
    cm1.__enter__()
    h1 = cx._REGISTRY["dup"]
    cm2 = cx.begin_run("dup")  # the takeover
    cm2.__enter__()
    h2 = cx._REGISTRY["dup"]
    assert h2 is not h1
    cm1.__exit__(None, None, None)  # late cleanup of the abandoned run
    assert cx._REGISTRY.get("dup") is h2  # new handle survived
    assert cx.is_active("dup")
    cm2.__exit__(None, None, None)
    assert not cx.is_active("dup")
