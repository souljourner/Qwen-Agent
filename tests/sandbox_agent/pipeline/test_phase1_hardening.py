"""Phase-1 trading-pipeline hardening tests: parser fix, no-reschedule-after-
exhausted, terminal/orphan guards, budget enforcement, pipeline-state write-guard."""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import pytest

import sandbox_agent.pipeline.orchestrator as orch
import sandbox_agent.pipeline.stage_runner as sr
from sandbox_agent.pipeline.stage_runner import _parse_stage_task_name


@pytest.fixture
def tmp_data_dir(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE", os.path.join(d, "pipeline.lock"))
    monkeypatch.setattr("sandbox_agent.tools.project_tools.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.tools.project_tools.PROJECTS_DIR", os.path.join(d, "projects"))
    os.makedirs(os.path.join(d, "projects", "proj"), exist_ok=True)
    yield d
    shutil.rmtree(d)


# --- parser -------------------------------------------------------------------

def test_parser_plain_stage():
    assert _parse_stage_task_name("pipeline:proj:stage_2") == ("proj", 2)


def test_parser_suffixed_stage():
    # Agents have created duplicate tasks like "pipeline:X:stage_4_verdict" —
    # this crashed the old int(replace(...)) parser.
    assert _parse_stage_task_name("pipeline:proj:stage_4_verdict") == ("proj", 4)


def test_parser_garbage_raises():
    with pytest.raises(ValueError):
        _parse_stage_task_name("pipeline:proj:notastage")
    with pytest.raises(ValueError):
        _parse_stage_task_name("pipeline:proj:stage_9")   # out of range
    with pytest.raises(ValueError):
        _parse_stage_task_name("garbage")


# --- mark_stage_failed: exhausted stages must NOT re-enqueue -------------------

def test_exhausted_stage_not_rescheduled(tmp_data_dir, monkeypatch):
    state = orch.init_pipeline("proj", "test", pipeline_type="trading")
    stage = state.stages[2]
    stage.run_count = stage.max_attempts - 1     # next failure exhausts
    orch.save_state(state)

    scheduled = []
    monkeypatch.setattr(orch, "_schedule_stage", lambda st, n: scheduled.append(n))
    # RequestUser notification inside _finalize_exhausted_stage — no-op it
    import sandbox_agent.tools.notification_tools as nt
    monkeypatch.setattr(nt.RequestUser, "call", lambda self, p, **k: "ok")

    orch.mark_stage_failed("proj", 2, "boom")

    reloaded = orch.load_state("proj")
    assert reloaded.stages[2].status in ("failed-no-more-attempts", "completed-no-more-attempts")
    assert 2 not in scheduled, "exhausted stage must not be re-enqueued"


def test_non_exhausted_failure_still_retries(tmp_data_dir, monkeypatch):
    state = orch.init_pipeline("proj", "test", pipeline_type="trading")
    orch.save_state(state)
    scheduled = []
    monkeypatch.setattr(orch, "_schedule_stage", lambda st, n: scheduled.append(n))
    orch.mark_stage_failed("proj", 2, "boom")
    reloaded = orch.load_state("proj")
    assert reloaded.stages[2].status == "scheduled"
    assert scheduled == [2]


# --- _execute_stage guards -----------------------------------------------------

@pytest.fixture
def guarded_stage(tmp_data_dir, monkeypatch):
    state = orch.init_pipeline("proj", "test", pipeline_type="trading")
    orch.save_state(state)

    # If the guard fails, the agent would run — make that loud.
    def _explode(*a, **k):
        raise AssertionError("agent should not run for guarded stage")
    import sandbox_agent.main as main_mod
    monkeypatch.setattr(main_mod, "run_on_best_available", _explode)
    return state


def test_terminal_stage_skipped(guarded_stage):
    state = guarded_stage
    state.stages[1].status = "completed"
    orch.save_state(state)
    out = sr._execute_stage("proj", 1, task_id="t-1", system_message="sys")
    assert "skip" in out.lower() or "terminal" in out.lower()


def test_orphan_task_skipped(guarded_stage):
    state = guarded_stage
    state.stages[1].task_id = "current-task"
    state.stages[1].status = "scheduled"
    orch.save_state(state)
    out = sr._execute_stage("proj", 1, task_id="stale-task", system_message="sys")
    assert "skip" in out.lower() or "stale" in out.lower() or "orphan" in out.lower()


def test_budget_exhausted_finalizes_without_running(guarded_stage, monkeypatch):
    state = guarded_stage
    stage = state.stages[1]
    stage.task_id = "t-1"
    stage.budget_seconds = 60
    stage.first_started_at = datetime.now() - timedelta(hours=2)
    orch.save_state(state)

    import sandbox_agent.tools.notification_tools as nt
    monkeypatch.setattr(nt.RequestUser, "call", lambda self, p, **k: "ok")

    out = sr._execute_stage("proj", 1, task_id="t-1", system_message="sys")
    assert "budget" in out.lower()
    reloaded = orch.load_state("proj")
    assert reloaded.stages[1].status in ("failed-no-more-attempts", "completed-no-more-attempts")


# --- pipeline-state write-guard --------------------------------------------------

def test_write_guard_denies_pipeline_state(tmp_data_dir):
    from sandbox_agent.tools.project_tools import ProjectWriteFile
    for path in ("pipeline/state.json", "pipeline/pipeline_state.json", "status.md"):
        out = ProjectWriteFile().call({"project": "proj", "path": path,
                                       "content": '{"fake": true}'})
        assert out.lower().startswith("error"), f"{path} should be denied"
        assert "orchestrator" in out.lower()


def test_write_guard_denies_metrics_json(tmp_data_dir):
    # Phase 2: pipeline/metrics.json is evaluator-written after the stage-4
    # gate passes — agent writes are denied like the other state files.
    from sandbox_agent.tools.project_tools import ProjectWriteFile
    out = ProjectWriteFile().call({"project": "proj", "path": "pipeline/metrics.json",
                                   "content": '{"verdict": null}'})
    assert out.lower().startswith("error")


def test_apply_patch_guard_denies_pipeline_state(tmp_data_dir):
    from sandbox_agent.tools.project_tools import ProjectApplyPatch
    patch = ("*** Begin Patch\n"
             "*** Add File: pipeline/pipeline_state.json\n"
             "+{\"fake\": true}\n"
             "*** End Patch")
    out = ProjectApplyPatch().call({"project": "proj", "patch": patch})
    assert out.lower().startswith("error")
    assert "orchestrator" in out.lower()
