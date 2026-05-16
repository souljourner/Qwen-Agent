"""Tests for sandbox_agent.task_notify — the background-task completion inbox."""

from sandbox_agent import task_notify


def setup_function():
    task_notify.drain()  # start each test with an empty queue


def test_notify_and_drain_fifo():
    task_notify.notify_task_done("t1", "task one", "did a thing")
    task_notify.notify_task_done("t2", "task two", "did another", source="pipeline")
    assert task_notify.pending_count() == 2
    evts = task_notify.drain()
    assert [e["task_id"] for e in evts] == ["t1", "t2"]  # FIFO
    assert evts[0] == {"task_id": "t1", "name": "task one", "result": "did a thing",
                       "ts": evts[0]["ts"], "source": "cron", "ok": True, "origin": None}
    assert evts[1]["source"] == "pipeline"
    assert evts[1]["origin"] is None  # default when not passed
    assert task_notify.pending_count() == 0
    assert task_notify.drain() == []  # already drained


def test_origin_is_carried_through():
    task_notify.notify_task_done(
        "t99", "moon-ai-bench", "result snippet",
        source="cron", origin={"session_id": "S", "thread_id": "T"},
    )
    (evt,) = task_notify.drain()
    assert evt["origin"] == {"session_id": "S", "thread_id": "T"}


def test_failure_event():
    task_notify.notify_task_done("t3", "flaky task", "FAILED: boom", source="cron", ok=False)
    (evt,) = task_notify.drain()
    assert evt["ok"] is False and evt["result"] == "FAILED: boom"


def test_result_is_truncated():
    task_notify.notify_task_done("t4", "big", "x" * 5000)
    (evt,) = task_notify.drain()
    assert len(evt["result"]) == 1000


def test_format_task_notice():
    from sandbox_agent.chat_app import _format_task_notice
    ok = _format_task_notice({"name": "research moon-ai", "result": "found 3 sources", "ok": True})
    assert "✅" in ok and "research moon-ai" in ok and "found 3 sources" in ok and "list_tasks" in ok
    fail = _format_task_notice({"name": "build mvp", "result": "FAILED: import error", "ok": False})
    assert "⚠️" in fail and "build mvp" in fail and "FAILED: import error" in fail
    bare = _format_task_notice({"name": "ping", "ok": True})  # no result
    assert "✅" in bare and "ping" in bare and "list_tasks" not in bare
