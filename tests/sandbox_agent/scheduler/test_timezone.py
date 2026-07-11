"""Timezone-aware scheduling: DST-invariant cron evaluation + in-place reschedule.

The failure being fixed: "12:30pm PT Fridays" scheduled as a fixed-offset cron
drifts ±1h across DST (the live Weekly COT scan sat at `31 19 * * 5`, and the
agent hand-rolled TWO seasonal EDT/EST monitor tasks). Cron is now wall-clock
in Task.timezone (default Pacific).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import sandbox_agent.scheduler.task_queue as tq_mod
from sandbox_agent.scheduler.models import Task
from sandbox_agent.scheduler.task_queue import TaskQueue

LA = ZoneInfo("America/Los_Angeles")


@pytest.fixture
def queue(tmp_path, monkeypatch):
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "autocommit", lambda *a, **k: None)
    # Pin the container-local zone so tests don't depend on the host TZ.
    monkeypatch.setattr(tq_mod, "_local_tz", lambda: LA)
    return TaskQueue(data_dir=str(tmp_path))


def _next_run(queue, cron, timezone, now_local_naive):
    task = Task(id="t", name="n", description="d", schedule_type="cron",
                cron=cron, timezone=timezone)
    return queue._compute_next_run(task, now_local_naive)


# --- DST invariance (the core effectiveness proof) ---------------------------

def test_pacific_cron_holds_wall_clock_in_winter_and_summer(queue):
    cron = "30 12 * * 5"  # Fridays 12:30pm PT
    # January (PST) and July (PDT) reference "now"s, both Wednesday local noon.
    for ref in (datetime(2026, 1, 7, 12, 0), datetime(2026, 7, 8, 12, 0)):
        nxt = _next_run(queue, cron, "America/Los_Angeles", ref)
        assert nxt.tzinfo is None                      # stored naive-local
        assert (nxt.hour, nxt.minute) == (12, 30), f"drifted at {ref}: {nxt}"
        assert nxt.weekday() == 4                       # Friday


def test_eastern_cron_fires_three_hours_earlier_local(queue):
    cron = "30 12 * * 5"
    ref = datetime(2026, 7, 8, 6, 0)
    pacific = _next_run(queue, cron, "America/Los_Angeles", ref)
    eastern = _next_run(queue, cron, "US/Eastern", ref)
    assert (pacific - eastern).total_seconds() == 3 * 3600


def test_crossing_spring_forward_keeps_wall_clock(queue):
    # Scheduled Friday 2026-03-06 (PST); next weekly fire 2026-03-13 is after
    # spring-forward (2026-03-08). Wall clock must stay 12:30.
    ref = datetime(2026, 3, 6, 13, 0)  # Friday just after that week's fire
    nxt = _next_run(queue, "30 12 * * 5", "America/Los_Angeles", ref)
    assert nxt == datetime(2026, 3, 13, 12, 30)


def test_legacy_task_without_timezone_loads_and_defaults(queue, tmp_path):
    legacy = Task(id="x", name="old", description="d", schedule_type="cron",
                  cron="0 9 * * *")
    data = legacy.model_dump(mode="json")
    data.pop("timezone", None)                          # pre-feature JSON shape
    reloaded = Task(**data)
    assert reloaded.timezone == "America/Los_Angeles"
    nxt = queue._compute_next_run(reloaded, datetime(2026, 7, 8, 6, 0))
    assert (nxt.hour, nxt.minute) == (9, 0)


# --- reschedule_task (in place) -----------------------------------------------

def test_reschedule_changes_cron_and_tz_in_place(queue, monkeypatch):
    import sandbox_agent.scheduler.scheduler_tools as st
    monkeypatch.setattr(st, "get_task_queue", lambda: queue)
    from sandbox_agent.scheduler.scheduler_tools import RescheduleTask, ScheduleTask

    out = ScheduleTask().call({"name": "cot-scan", "description": "d",
                               "schedule_type": "cron", "cron": "31 19 * * 5"})
    task = next(t for t in queue.list_tasks() if t.name == "cot-scan")
    task.checkpoint = {"step": 3}
    queue._save()

    out = RescheduleTask().call({"task_id": task.id, "cron": "30 12 * * 5",
                                 "timezone": "America/Los_Angeles"})
    assert not out.lower().startswith("error"), out
    updated = queue.get_task(task.id)
    assert updated.id == task.id                        # same task
    assert updated.cron == "30 12 * * 5"
    assert updated.checkpoint == {"step": 3}            # history preserved
    assert (updated.next_run.hour, updated.next_run.minute) == (12, 30)
    assert updated.next_run.weekday() == 4


def test_reschedule_invalid_timezone_clean_error(queue, monkeypatch):
    import sandbox_agent.scheduler.scheduler_tools as st
    monkeypatch.setattr(st, "get_task_queue", lambda: queue)
    from sandbox_agent.scheduler.scheduler_tools import RescheduleTask
    task = queue.add_task(name="t", description="d", schedule_type="cron", cron="0 9 * * *")
    out = RescheduleTask().call({"task_id": task.id, "timezone": "Pacific/Fakeville"})
    assert out.lower().startswith("error") and "timezone" in out.lower()


def test_reschedule_unknown_task(queue, monkeypatch):
    import sandbox_agent.scheduler.scheduler_tools as st
    monkeypatch.setattr(st, "get_task_queue", lambda: queue)
    from sandbox_agent.scheduler.scheduler_tools import RescheduleTask
    out = RescheduleTask().call({"task_id": "nope", "cron": "0 9 * * *"})
    assert "not found" in out.lower()


def test_schedule_task_accepts_timezone_param(queue, monkeypatch):
    import sandbox_agent.scheduler.scheduler_tools as st
    monkeypatch.setattr(st, "get_task_queue", lambda: queue)
    from sandbox_agent.scheduler.scheduler_tools import ScheduleTask
    out = ScheduleTask().call({"name": "et-open", "description": "d",
                               "schedule_type": "cron", "cron": "30 9 * * 1-5",
                               "timezone": "US/Eastern"})
    assert not out.lower().startswith("error"), out
    task = next(t for t in queue.list_tasks() if t.name == "et-open")
    assert task.timezone == "US/Eastern"
    # 9:30am ET == 6:30am PT (local storage)
    assert (task.next_run.hour, task.next_run.minute) == (6, 30)


def test_make_naive_converts_before_stripping(queue):
    aware_utc = datetime(2026, 7, 8, 19, 30, tzinfo=ZoneInfo("UTC"))
    naive = TaskQueue._make_naive(aware_utc)
    assert naive == datetime(2026, 7, 8, 12, 30)        # 19:30Z == 12:30 PDT
