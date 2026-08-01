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


def test_cancelled_task_archive_is_listable(tmp_path, monkeypatch):
    # remove_task stamps status='cancelled', but the Task model's Literal
    # didn't allow it — list_tasks(category='cancelled') crashed on pydantic
    # validation when reloading the archive (2026-07-16).
    import sandbox_agent.scheduler.scheduler_tools as st
    from sandbox_agent.scheduler.task_queue import TaskQueue
    tq = TaskQueue(data_dir=str(tmp_path))
    t = tq.add_task(name="to-cancel", description="d")
    tq.remove_task(t.id)
    cancelled = tq.list_tasks(category="cancelled")
    assert [c.id for c in cancelled] == [t.id]
    assert cancelled[0].status == "cancelled"


def test_git_autocommit_breaks_stale_index_lock(tmp_path, monkeypatch):
    # A crashed git process left index.lock in DATA_DIR/.git for 2 DAYS
    # (2026-07-15 → 17); every autocommit failed until manual cleanup. Stale
    # locks (>10 min) must be broken automatically.
    import time
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "DATA_DIR", str(tmp_path))
    ga.ensure_git_repo()
    lock = tmp_path / ".git" / "index.lock"
    lock.write_text("")
    old = time.time() - 3600
    import os as _os
    _os.utime(lock, (old, old))
    (tmp_path / "f.txt").write_text("hello")
    ga.autocommit("f.txt", "test commit")
    assert not lock.exists()
    assert "test commit" in ga._run_git("log", "--oneline", cwd=str(tmp_path))


def test_git_autocommit_respects_fresh_lock(tmp_path, monkeypatch):
    # A FRESH lock means a live git process — never break it.
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "DATA_DIR", str(tmp_path))
    ga.ensure_git_repo()
    lock = tmp_path / ".git" / "index.lock"
    lock.write_text("")
    (tmp_path / "f.txt").write_text("hello")
    ga.autocommit("f.txt", "test commit")
    assert lock.exists()  # untouched


class TestUiHardening:
    """Chainlit UI audit (2026-08-01): scroll/paint stall fixes."""

    def test_step_output_cap_passthrough_and_headtail(self):
        from sandbox_agent.ui_output import (_STEP_OUTPUT_CAP_CHARS,
                                            _cap_step_output)
        small = "x" * (_STEP_OUTPUT_CAP_CHARS - 1)
        assert _cap_step_output(small) is small
        big = "H" * 30_000 + "T" * 5_000
        out = _cap_step_output(big)
        assert len(out) < len(big)
        assert out.startswith("H" * 100)
        assert out.endswith("T" * 100)
        assert "chars omitted" in out and "agent received the full output" in out

    def test_step_output_cap_tail_mode(self):
        from sandbox_agent.ui_output import _cap_step_output
        big = "E" * 30_000 + "LATEST-LINES"
        out = _cap_step_output(big, keep="tail")
        assert out.endswith("LATEST-LINES")
        assert "earlier chars omitted" in out
        assert len(out) < len(big)

    def test_ui_deliverables_in_place(self):
        # Drift-locks: the shim exists, config references it, cot collapsed,
        # chainlit pinned, app root baked into the image.
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        shim = repo / "public" / "ui-unstick.js"
        assert shim.exists() and "MutationObserver" in shim.read_text()
        cfg = (repo / ".chainlit" / "config.toml").read_text()
        assert 'custom_js = "/public/ui-unstick.js"' in cfg
        assert 'cot = "tool_call"' in cfg
        dockerfile = (repo / "sandbox_agent" / "docker" / "Dockerfile").read_text()
        assert "chainlit==2.11.1" in dockerfile
        assert "CHAINLIT_APP_ROOT=/app/chainlit_root" in dockerfile
        assert "COPY public/ /app/chainlit_root/public/" in dockerfile
