"""Scheduler hardening tests: atomic persistence, corruption quarantine,
generation-token race protection, interval drift, missed-window annotation,
cron pre-validation, empty-result detection."""

import glob
import json
import os
from datetime import datetime, timedelta

import pytest

from sandbox_agent.scheduler.task_queue import TaskQueue


@pytest.fixture
def queue(tmp_path, monkeypatch):
    # No git autocommit noise in tests
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "autocommit", lambda *a, **k: None)
    return TaskQueue(data_dir=str(tmp_path))


# --- atomic writes + quarantine -------------------------------------------

def test_interrupted_save_leaves_old_file_intact(queue, tmp_path, monkeypatch):
    queue.add_task(name="keep-me", description="d")
    tasks_file = tmp_path / "tasks.json"
    before = tasks_file.read_text()
    assert "keep-me" in before

    # Simulate a crash mid-serialization on the NEXT save.
    import sandbox_agent.scheduler.task_queue as tq_mod
    real_dump = json.dump
    def exploding_dump(*a, **k):
        raise OSError("disk full mid-write")
    monkeypatch.setattr(tq_mod.json, "dump", exploding_dump)
    try:
        queue.add_task(name="doomed", description="d")
    except Exception:
        pass
    monkeypatch.setattr(tq_mod.json, "dump", real_dump)

    # The on-disk file must still be valid JSON with the original task.
    data = json.loads(tasks_file.read_text())
    assert any(t["name"] == "keep-me" for t in data)


def test_corrupt_tasks_file_quarantined_not_discarded(tmp_path, monkeypatch):
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "autocommit", lambda *a, **k: None)
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text('{"broken json...')
    q = TaskQueue(data_dir=str(tmp_path))
    assert q.list_tasks() == []
    quarantined = glob.glob(str(tmp_path / "tasks.json.corrupt-*"))
    assert len(quarantined) == 1
    assert open(quarantined[0]).read() == '{"broken json...'


# --- generation token (stuck-worker double-run race) ------------------------

def test_stale_worker_update_ignored_after_generation_bump(queue):
    task = queue.add_task(name="racy", description="d")
    gen = task.run_generation                       # worker captures at start

    new_gen = queue.bump_generation(task.id)        # stuck-detector abandons
    assert new_gen == gen + 1

    # Zombie worker finishes later and tries to mark completed with stale gen.
    out = queue.update_task(task.id, status="completed", result="zombie",
                            expected_generation=gen)
    assert out is None
    current = queue.get_task(task.id)
    assert current.status != "completed"
    assert current.result != "zombie"


def test_matching_generation_applies(queue):
    task = queue.add_task(name="ok", description="d")
    gen = task.run_generation
    out = queue.update_task(task.id, status="completed", result="fine",
                            expected_generation=gen)
    assert out is not None


def test_update_without_generation_unchanged_behavior(queue):
    task = queue.add_task(name="legacy", description="d")
    queue.bump_generation(task.id)
    out = queue.update_task(task.id, status="completed", result="r")
    assert out is not None  # callers not passing expected_generation are unaffected


# --- every-interval drift ----------------------------------------------------

def test_every_interval_no_drift(queue):
    t0 = datetime.now() + timedelta(seconds=50)   # scheduled slightly in future
    task = queue.add_task(name="tick", description="d", schedule_type="every",
                          interval_seconds=3600)
    task.next_run = t0
    # Completing "now" (before/at the scheduled time) → next = t0 + interval,
    # NOT completion_time + interval.
    queue.update_task(task.id, status="completed", result="ok")
    nxt = queue.get_task(task.id).next_run
    expected = t0 + timedelta(seconds=3600)
    assert abs((nxt - expected).total_seconds()) < 2


def test_every_interval_overdue_does_not_burst(queue):
    long_ago = datetime.now() - timedelta(days=3)
    task = queue.add_task(name="stale-tick", description="d", schedule_type="every",
                          interval_seconds=3600)
    task.next_run = long_ago
    queue.update_task(task.id, status="completed", result="ok")
    nxt = queue.get_task(task.id).next_run
    assert nxt >= datetime.now() - timedelta(seconds=2)  # clamped to now, no burst


# --- missed-window annotation -------------------------------------------------

def test_missed_cron_windows_annotated(queue):
    task = queue.add_task(name="daily", description="d", schedule_type="cron",
                          cron="0 9 * * *")
    task.next_run = datetime.now() - timedelta(days=3)
    counts = queue.annotate_missed_windows()
    assert counts.get(task.id, 0) >= 2
    assert "missed" in (queue.get_task(task.id).last_error or "").lower()


def test_on_time_cron_not_annotated(queue):
    task = queue.add_task(name="future", description="d", schedule_type="cron",
                          cron="0 9 * * *")
    counts = queue.annotate_missed_windows()
    assert task.id not in counts


# --- schedule_task pre-validation ---------------------------------------------

@pytest.fixture
def sched_tool(queue, monkeypatch):
    import sandbox_agent.scheduler.scheduler_tools as st
    monkeypatch.setattr(st, "get_task_queue", lambda: queue)
    from sandbox_agent.scheduler.scheduler_tools import ScheduleTask
    return ScheduleTask()


def test_invalid_cron_rejected(sched_tool):
    out = sched_tool.call({"name": "bad", "description": "d",
                           "schedule_type": "cron", "cron": "not a cron"})
    assert out.lower().startswith("error")
    assert "cron" in out.lower()


def test_invalid_run_at_rejected(sched_tool):
    out = sched_tool.call({"name": "bad", "description": "d",
                           "schedule_type": "at", "run_at": "next tuesday-ish"})
    assert out.lower().startswith("error")
    assert "run_at" in out.lower()


def test_valid_cron_still_schedules(sched_tool, queue):
    out = sched_tool.call({"name": "good", "description": "d",
                           "schedule_type": "cron", "cron": "0 9 * * *"})
    assert not out.lower().startswith("error")
    assert any(t.name == "good" for t in queue.list_tasks())


# --- empty-result detection (main.py helper) -----------------------------------

def test_is_empty_result():
    from sandbox_agent.main import _is_empty_result
    assert _is_empty_result("")
    assert _is_empty_result("   \n\n\t  ")
    assert not _is_empty_result("DONE")
    assert not _is_empty_result("x")
