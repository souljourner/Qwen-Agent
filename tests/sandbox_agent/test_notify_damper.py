"""Tests for the notification-turn damper + the cancel_task serialization fix
(both from the 2026-07-16 runaway-loop incident)."""

import json
from datetime import datetime

import pytest

from sandbox_agent import notify_damper as nd


@pytest.fixture(autouse=True)
def _fresh():
    nd.reset()
    yield
    nd.reset()


class TestReactionCooldown:

    def test_first_notification_reacts(self):
        assert nd.should_trigger_synthetic_turn("pipeline:x:stage_2", now=1000.0)

    def test_repeat_within_cooldown_suppressed(self):
        assert nd.should_trigger_synthetic_turn("pipeline:x:stage_2", now=1000.0)
        assert not nd.should_trigger_synthetic_turn("pipeline:x:stage_2", now=1000.0 + 60)
        assert not nd.should_trigger_synthetic_turn(
            "pipeline:x:stage_2", now=1000.0 + nd.REACT_COOLDOWN_S - 1)

    def test_reacts_again_after_cooldown(self):
        assert nd.should_trigger_synthetic_turn("pipeline:x:stage_2", now=1000.0)
        assert nd.should_trigger_synthetic_turn(
            "pipeline:x:stage_2", now=1000.0 + nd.REACT_COOLDOWN_S + 1)

    def test_different_tasks_independent(self):
        assert nd.should_trigger_synthetic_turn("task-a", now=1000.0)
        assert nd.should_trigger_synthetic_turn("task-b", now=1000.0)


class TestStopMute:

    def test_stop_mutes_all_synthetic_turns(self):
        nd.mute_on_stop(now=1000.0)
        assert not nd.should_trigger_synthetic_turn("anything", now=1000.0 + 1)
        assert not nd.should_trigger_synthetic_turn("else", now=1000.0 + nd.STOP_MUTE_S - 1)

    def test_mute_expires(self):
        nd.mute_on_stop(now=1000.0)
        assert nd.should_trigger_synthetic_turn("anything", now=1000.0 + nd.STOP_MUTE_S + 1)


class TestCancelTaskSerialization:

    def test_cancel_task_result_is_valid_json_with_datetimes(self, tmp_path, monkeypatch):
        # 2026-07-16: cancel_task removed the task AND killed the run, then
        # crashed serializing the response (datetime in task_definition) —
        # the agent saw an error, retried, and escalated to cancelling the
        # running pipeline stage. The response must serialize.
        import sandbox_agent.scheduler.scheduler_tools as st
        from sandbox_agent.scheduler.task_queue import TaskQueue

        # Hermetic: fresh queue on a tmp dir; bypass the process singleton
        # (other tests mutate/replace it).
        tq = TaskQueue(data_dir=str(tmp_path))
        monkeypatch.setattr(st, "get_task_queue", lambda: tq)
        out = st.ScheduleTask().call(json.dumps({
            "description": "damper test task",
            "name": "damper-test",
            "schedule_type": "at",
            "at": "2099-01-01 00:00",
        }))
        task_id = json.loads(out)["task_id"]
        result = st.CancelTask().call(json.dumps({"task_id": task_id}))
        parsed = json.loads(result)  # must not raise, must be valid JSON
        assert parsed["status"] == "cancelled"
        assert parsed["task_id"] == task_id
        assert "damper-test" in parsed["name"]
