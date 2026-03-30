"""Tests for the task queue, checkpoint, and scheduler tools."""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import pytest

from sandbox_agent.scheduler.checkpoint import (
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from sandbox_agent.scheduler.models import Task
from sandbox_agent.scheduler.task_queue import TaskQueue


@pytest.fixture
def tmp_data_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def task_queue(tmp_data_dir):
    return TaskQueue(data_dir=tmp_data_dir)


class TestTaskModel:

    def test_defaults(self):
        task = Task(id="abc", name="test", description="desc")
        assert task.status == "pending"
        assert task.schedule_type == "at"
        assert task.retry_count == 0
        assert task.depends_on == []

    def test_serialization_roundtrip(self):
        task = Task(id="abc", name="test", description="desc", priority=5)
        data = task.model_dump(mode="json")
        restored = Task(**data)
        assert restored.id == "abc"
        assert restored.priority == 5


class TestTaskQueue:

    def test_add_one_shot_task(self, task_queue):
        task = task_queue.add_task(
            name="test", description="do something", schedule_type="at",
        )
        assert task.id
        assert task.status == "pending"
        assert task.next_run is not None

    def test_add_cron_task(self, task_queue):
        task = task_queue.add_task(
            name="hourly", description="check stuff",
            schedule_type="cron", cron="0 * * * *",
        )
        assert task.cron == "0 * * * *"
        assert task.next_run > datetime.now()

    def test_add_every_task(self, task_queue):
        task = task_queue.add_task(
            name="frequent", description="poll",
            schedule_type="every", interval_seconds=300,
        )
        assert task.interval_seconds == 300
        expected = datetime.now() + timedelta(seconds=300)
        assert abs((task.next_run - expected).total_seconds()) < 2

    def test_get_due_tasks_immediate(self, task_queue):
        task_queue.add_task(
            name="now", description="run now", schedule_type="at",
            run_at=datetime.now() - timedelta(seconds=1),
        )
        due = task_queue.get_due_tasks()
        assert len(due) == 1
        assert due[0].name == "now"

    def test_get_due_tasks_future_not_due(self, task_queue):
        task_queue.add_task(
            name="later", description="not yet", schedule_type="at",
            run_at=datetime.now() + timedelta(hours=1),
        )
        due = task_queue.get_due_tasks()
        assert len(due) == 0

    def test_dependency_blocking(self, task_queue):
        t1 = task_queue.add_task(name="first", description="first task", schedule_type="at")
        t2 = task_queue.add_task(
            name="second", description="depends on first",
            schedule_type="at", depends_on=[t1.id],
        )
        # t2 should not be due because t1 is not completed
        due = task_queue.get_due_tasks()
        assert any(t.id == t1.id for t in due)
        assert not any(t.id == t2.id for t in due)

        # Complete t1
        task_queue.update_task(t1.id, status="completed")
        due = task_queue.get_due_tasks()
        assert any(t.id == t2.id for t in due)

    def test_priority_ordering(self, task_queue):
        task_queue.add_task(name="low", description="low pri", schedule_type="at", priority=1)
        task_queue.add_task(name="high", description="high pri", schedule_type="at", priority=10)
        due = task_queue.get_due_tasks()
        assert due[0].name == "high"
        assert due[1].name == "low"

    def test_complete_one_shot(self, task_queue):
        task = task_queue.add_task(name="once", description="one shot", schedule_type="at")
        result = task_queue.update_task(task.id, status="completed", result="done")
        assert result.status == "completed"
        assert result.next_run is None

    def test_complete_recurring_resets(self, task_queue):
        task = task_queue.add_task(
            name="repeat", description="every 5 min",
            schedule_type="every", interval_seconds=300,
        )
        result = task_queue.update_task(task.id, status="completed")
        # Should be reset to pending with new next_run
        assert result.status == "pending"
        assert result.next_run > datetime.now()
        assert result.retry_count == 0

    def test_failure_exponential_backoff(self, task_queue):
        task = task_queue.add_task(name="flaky", description="might fail", schedule_type="at")
        # First failure: 30s backoff
        result = task_queue.update_task(task.id, status="failed", last_error="oops")
        assert result.status == "pending"  # Reset to pending for retry
        assert result.retry_count == 1
        assert result.next_run > datetime.now()

    def test_max_retries_one_shot(self, task_queue):
        task = task_queue.add_task(
            name="doomed", description="always fails", schedule_type="at",
        )
        # Fail max_retries times
        for _ in range(task.max_retries):
            task_queue.update_task(task.id, status="failed", last_error="fail")

        # Should no longer be in due tasks
        t = task_queue.get_task(task.id)
        # After max retries, stays failed
        due = task_queue.get_due_tasks()
        assert not any(d.id == task.id for d in due)

    def test_persistence(self, tmp_data_dir):
        tq1 = TaskQueue(data_dir=tmp_data_dir)
        task = tq1.add_task(name="persist", description="saved to disk", schedule_type="at")

        # Create new TaskQueue instance (simulates restart)
        tq2 = TaskQueue(data_dir=tmp_data_dir)
        loaded = tq2.get_task(task.id)
        assert loaded is not None
        assert loaded.name == "persist"

    def test_list_tasks_filter(self, task_queue):
        t1 = task_queue.add_task(name="a", description="a", schedule_type="at")
        t2 = task_queue.add_task(name="b", description="b", schedule_type="at")
        task_queue.update_task(t1.id, status="completed")

        current = task_queue.list_tasks(category="current")
        completed = task_queue.list_tasks(category="completed")
        assert len(current) == 1  # Only t2 (pending)
        assert len(completed) == 1  # t1 archived


class TestCheckpoint:

    def test_save_and_load(self, tmp_data_dir, monkeypatch):
        monkeypatch.setattr("sandbox_agent.scheduler.checkpoint.DATA_DIR", tmp_data_dir)
        save_checkpoint("task123", step=3, state={"progress": "step 3 done"})
        loaded = load_checkpoint("task123")
        assert loaded["step"] == 3
        assert loaded["state"]["progress"] == "step 3 done"

    def test_load_nonexistent(self, tmp_data_dir, monkeypatch):
        monkeypatch.setattr("sandbox_agent.scheduler.checkpoint.DATA_DIR", tmp_data_dir)
        assert load_checkpoint("nonexistent") is None

    def test_delete(self, tmp_data_dir, monkeypatch):
        monkeypatch.setattr("sandbox_agent.scheduler.checkpoint.DATA_DIR", tmp_data_dir)
        save_checkpoint("task456", step=1, state={})
        delete_checkpoint("task456")
        assert load_checkpoint("task456") is None


class TestSchedulerTools:

    @pytest.fixture(autouse=True)
    def setup_tools(self, task_queue):
        """Inject test task queue into scheduler tools."""
        from sandbox_agent.scheduler import scheduler_tools
        scheduler_tools.set_task_queue(task_queue)
        yield

    def test_schedule_task_tool(self):
        from sandbox_agent.scheduler.scheduler_tools import ScheduleTask
        tool = ScheduleTask()
        result = tool.call(json.dumps({
            "name": "test task",
            "description": "do the thing",
            "schedule_type": "at",
        }))
        parsed = json.loads(result)
        assert parsed["status"] == "scheduled"
        assert parsed["task_id"]

    def test_list_tasks_tool(self, task_queue):
        from sandbox_agent.scheduler.scheduler_tools import ListTasks
        task_queue.add_task(name="a", description="a", schedule_type="at")
        tool = ListTasks()
        result = tool.call("{}")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "a"

    def test_complete_task_tool(self, task_queue):
        from sandbox_agent.scheduler.scheduler_tools import CompleteTask
        task = task_queue.add_task(name="finish me", description="x", schedule_type="at")
        tool = CompleteTask()
        result = tool.call(json.dumps({"task_id": task.id, "result": "all done"}))
        parsed = json.loads(result)
        assert parsed["status"] == "completed"

    def test_checkpoint_tool(self, task_queue, tmp_data_dir, monkeypatch):
        from sandbox_agent.scheduler.scheduler_tools import UpdateTaskCheckpoint
        monkeypatch.setattr("sandbox_agent.scheduler.checkpoint.DATA_DIR", tmp_data_dir)
        task = task_queue.add_task(name="long job", description="x", schedule_type="at")
        tool = UpdateTaskCheckpoint()
        result = tool.call(json.dumps({
            "task_id": task.id,
            "step": 5,
            "checkpoint": {"items_processed": 50},
        }))
        parsed = json.loads(result)
        assert parsed["status"] == "checkpointed"
        assert parsed["step"] == 5

    def test_schedule_cron_task(self):
        from sandbox_agent.scheduler.scheduler_tools import ScheduleTask
        tool = ScheduleTask()
        result = tool.call(json.dumps({
            "name": "hourly check",
            "description": "check stocks",
            "schedule_type": "cron",
            "cron": "0 * * * *",
        }))
        parsed = json.loads(result)
        assert parsed["status"] == "scheduled"
