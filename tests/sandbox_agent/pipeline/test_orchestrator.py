"""Tests for pipeline orchestrator — state machine, lock, stage management."""

import json
import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.models import PipelineState, StageState
from sandbox_agent.pipeline.orchestrator import (
    STARTUP_STAGES as STAGES,
    TRADING_STAGES,
    _verdict_is_reject,
    acquire_lock,
    clear_lock_on_startup,
    get_num_stages,
    init_pipeline,
    load_stage_instructions,
    load_state,
    mark_stage_part_completion,
    release_lock,
    save_state,
)

NUM_STAGES = get_num_stages("startup")


@pytest.fixture
def tmp_data_dir(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE", os.path.join(d, "pipeline.lock"))
    os.makedirs(os.path.join(d, "projects"), exist_ok=True)
    yield d
    shutil.rmtree(d)


class TestStageDefinitions:

    def test_all_stages_defined(self):
        assert len(STAGES) == NUM_STAGES
        for i in range(1, NUM_STAGES + 1):
            assert i in STAGES
            assert "name" in STAGES[i]
            assert "inputs" in STAGES[i]
            assert "outputs" in STAGES[i]
            assert "required_sections" in STAGES[i]

    def test_stage_names(self):
        assert STAGES[1]["name"] == "market_research"
        assert STAGES[2]["name"] == "brd"
        assert STAGES[3]["name"] == "prd"
        assert STAGES[4]["name"] == "vc_pitch"
        assert STAGES[5]["name"] == "mvp"
        assert STAGES[6]["name"] == "review"

    def test_stage_1_has_no_inputs(self):
        assert STAGES[1]["inputs"] == []

    def test_later_stages_have_inputs(self):
        for i in range(2, NUM_STAGES + 1):
            assert len(STAGES[i]["inputs"]) > 0, f"Stage {i} should have inputs"


class TestSaveLoadState:

    def test_save_and_load(self, tmp_data_dir):
        state = PipelineState(
            project_name="test-project",
            description="Test idea",
            current_stage=1,
            stages={
                1: StageState(stage_number=1, stage_name="market_research"),
            },
        )
        save_state(state)
        loaded = load_state("test-project")
        assert loaded is not None
        assert loaded.project_name == "test-project"
        assert loaded.current_stage == 1
        assert 1 in loaded.stages

    def test_load_nonexistent(self, tmp_data_dir):
        assert load_state("nonexistent") is None

    def test_state_creates_directory(self, tmp_data_dir):
        state = PipelineState(project_name="new-project", description="New")
        save_state(state)
        assert os.path.exists(os.path.join(tmp_data_dir, "projects", "new-project", "pipeline", "state.json"))


class TestInitPipeline:

    def test_creates_new_pipeline(self, tmp_data_dir):
        state = init_pipeline("test", "Test idea")
        assert state.project_name == "test"
        assert state.status == "running"
        assert len(state.stages) == NUM_STAGES
        for i in range(1, NUM_STAGES + 1):
            assert state.stages[i].status == "scheduled"

    def test_does_not_reset_active_pipeline(self, tmp_data_dir):
        state = init_pipeline("test", "Test idea")
        state.current_stage = 3
        state.stages[1].status = "completed"
        state.stages[2].status = "completed"
        save_state(state)

        # Try to init again — should return existing state
        state2 = init_pipeline("test", "Updated description")
        assert state2.current_stage == 3  # Not reset

    def test_resets_completed_pipeline(self, tmp_data_dir):
        state = init_pipeline("test", "Test idea")
        state.status = "completed"
        save_state(state)

        # Init again — should reset
        state2 = init_pipeline("test", "Updated description")
        assert state2.status == "running"
        assert state2.current_stage == 1
        for i in range(1, NUM_STAGES + 1):
            assert state2.stages[i].status == "scheduled"


class TestLock:

    def test_acquire_and_release(self, tmp_data_dir):
        assert acquire_lock("task-1") is True
        assert acquire_lock("task-2") is False  # Already locked
        release_lock()
        assert acquire_lock("task-2") is True  # Now free
        release_lock()

    def test_stale_lock_broken(self, tmp_data_dir, monkeypatch):
        # Set stale threshold very low for testing
        monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_STALE_SECONDS", 0)
        acquire_lock("old-task")
        # Should break the stale lock
        assert acquire_lock("new-task") is True
        release_lock()

    def test_clear_on_startup(self, tmp_data_dir):
        acquire_lock("task-1")
        lock_path = os.path.join(tmp_data_dir, "pipeline.lock")
        assert os.path.exists(lock_path)

        clear_lock_on_startup()
        assert not os.path.exists(lock_path)

    def test_clear_on_startup_resets_running_stages(self, tmp_data_dir):
        state = init_pipeline("test", "Test")
        state.stages[1].status = "running"
        save_state(state)

        clear_lock_on_startup()

        loaded = load_state("test")
        assert loaded.stages[1].status == "scheduled"


class TestLoadStageInstructions:

    def test_loads_existing_instruction(self):
        instructions = load_stage_instructions(1, "startup")
        assert "Market Research" in instructions
        assert "Competitors" in instructions

    def test_loads_all_stages(self):
        for i in range(1, NUM_STAGES + 1):
            instructions = load_stage_instructions(i, "startup")
            assert len(instructions) > 100, f"Stage {i} instructions too short"

    def test_loads_trading_stages(self):
        for i in range(1, 7):
            instructions = load_stage_instructions(i, "trading")
            assert len(instructions) > 100, f"Trading stage {i} instructions too short"


class TestStage6OutputsNoStatusMd:

    def test_startup_stage6_no_status_md(self):
        assert "status.md" not in STAGES[6]["outputs"]
        assert "pipeline/review.md" in STAGES[6]["outputs"]

    def test_trading_stage6_no_status_md(self):
        assert "status.md" not in TRADING_STAGES[6]["outputs"]
        assert "pipeline/review.md" in TRADING_STAGES[6]["outputs"]


class TestBudgetsAndCeilings:

    def test_trading_research_loop_budget_and_ceiling(self, tmp_data_dir):
        # Stage 2 (research_loop) is the only budgeted trading stage in the
        # redesigned pipeline — 48h wall-clock, 150 part-completions.
        state = init_pipeline("tx", "a strategy", pipeline_type="trading")
        stage2 = state.stages[2]
        assert stage2.max_part_completions == 150
        assert stage2.budget_seconds == 172800

    def test_startup_stages_use_defaults(self, tmp_data_dir):
        state = init_pipeline("startup-x", "idea", pipeline_type="startup")
        # No startup stage should carry a budget
        for num, stage in state.stages.items():
            assert stage.budget_seconds is None, f"stage {num} has unexpected budget"
            assert stage.max_part_completions == 20

    def test_trading_non_budget_stages_use_defaults(self, tmp_data_dir):
        state = init_pipeline("tx", "a strategy", pipeline_type="trading")
        # Everything except stage 2 uses defaults.
        for num in (1, 3, 4, 5, 6):
            assert state.stages[num].budget_seconds is None
            assert state.stages[num].max_part_completions == 20


@pytest.fixture
def stubbed_scheduler(monkeypatch):
    """Replace the scheduler's TaskQueue with an in-memory stub so orchestrator
    calls that normally persist to tasks.json don't touch DATA_DIR."""
    import sys
    import types
    import uuid

    class _FakeTask:
        def __init__(self, name, project):
            self.id = str(uuid.uuid4())
            self.name = name
            self.project = project

    class _FakeTaskQueue:
        def __init__(self):
            self.added = []

        def add_task(self, *, name, description, schedule_type, run_at, project, origin=None):
            task = _FakeTask(name, project)
            self.added.append(task)
            return task

    fake_queue = _FakeTaskQueue()

    # Stub scheduler_tools.get_task_queue — _schedule_stage imports it lazily.
    mod = types.ModuleType("sandbox_agent.scheduler.scheduler_tools")
    mod.get_task_queue = lambda: fake_queue
    monkeypatch.setitem(sys.modules, "sandbox_agent.scheduler.scheduler_tools", mod)

    # Stub notification_tools.RequestUser — used by _finalize_exhausted_stage
    # on the no-artifacts path.
    notif = types.ModuleType("sandbox_agent.tools.notification_tools")

    class _StubRequestUser:
        def __init__(self):
            self.calls = []

        def call(self, payload):
            self.calls.append(payload)
            return ""

    notif.RequestUser = _StubRequestUser
    monkeypatch.setitem(sys.modules, "sandbox_agent.tools.notification_tools", notif)

    return fake_queue


class TestPartCompletionCeiling:

    def test_part_completion_increments_counter(self, tmp_data_dir, stubbed_scheduler):
        state = init_pipeline("tx", "d", pipeline_type="trading")
        assert state.stages[3].part_completion_count == 0
        mark_stage_part_completion("tx", 3, "chunk done")
        reloaded = load_state("tx")
        assert reloaded.stages[3].part_completion_count == 1
        assert reloaded.stages[3].status == "part-completion"
        # Same-stage re-schedule, not advance
        assert reloaded.current_stage == 1

    def test_ceiling_advances_to_next_stage_when_artifacts_exist(
        self, tmp_data_dir, stubbed_scheduler,
    ):
        state = init_pipeline("tx", "d", pipeline_type="trading")
        state.stages[3].max_part_completions = 2
        # Create an artifact that matches trading stage 3 (full_validation) outputs.
        full_dir = os.path.join(tmp_data_dir, "projects", "tx", "backtest", "full")
        os.makedirs(full_dir, exist_ok=True)
        with open(os.path.join(full_dir, "results.md"), "w") as f:
            f.write("# Results\nplaceholder\n")
        save_state(state)

        mark_stage_part_completion("tx", 3, "chunk 1")
        mark_stage_part_completion("tx", 3, "chunk 2")  # hits ceiling

        reloaded = load_state("tx")
        assert reloaded.stages[3].part_completion_count == 2
        assert reloaded.stages[3].status == "completed-no-more-attempts"
        assert reloaded.current_stage == 4

    def test_ceiling_fails_stage_when_no_artifacts(
        self, tmp_data_dir, stubbed_scheduler,
    ):
        state = init_pipeline("tx-nofile", "d", pipeline_type="trading")
        state.stages[3].max_part_completions = 1
        save_state(state)

        mark_stage_part_completion("tx-nofile", 3, "stuck")

        reloaded = load_state("tx-nofile")
        assert reloaded.stages[3].status == "failed-no-more-attempts"
        # Still advances to next stage after failure
        assert reloaded.current_stage == 4

    def test_part_completion_ignored_when_stage_already_completed(
        self, tmp_data_dir, stubbed_scheduler,
    ):
        # Reproduces the prem14a corruption: an orphaned stage-N task runs
        # after stage N was already marked completed and stage N+1 was
        # scheduled. mark_stage_part_completion must NOT flip stage N back.
        state = init_pipeline("tx-done", "d", pipeline_type="trading")
        state.stages[4].status = "completed"
        state.stages[4].part_completion_count = 0
        state.current_stage = 5
        save_state(state)

        mark_stage_part_completion("tx-done", 4, "late ghost run")

        reloaded = load_state("tx-done")
        assert reloaded.stages[4].status == "completed"
        assert reloaded.stages[4].part_completion_count == 0
        assert reloaded.current_stage == 5


class TestVerdictGateAndRejectedStatus:

    def _write_verdict(self, tmp_data_dir, project, body):
        vdir = os.path.join(tmp_data_dir, "projects", project, "pipeline")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "verdict.md"), "w") as f:
            f.write(body)

    def test_verdict_missing_returns_false(self, tmp_data_dir):
        init_pipeline("tx-v", "d", pipeline_type="trading")
        assert _verdict_is_reject("tx-v") is False

    def test_verdict_promote_returns_false(self, tmp_data_dir):
        init_pipeline("tx-p", "d", pipeline_type="trading")
        self._write_verdict(
            tmp_data_dir, "tx-p",
            "## Final Recommendation\npromote\n\n## Rationale\nAll gates passed.",
        )
        assert _verdict_is_reject("tx-p") is False

    def test_verdict_reject_returns_true(self, tmp_data_dir):
        init_pipeline("tx-r", "d", pipeline_type="trading")
        self._write_verdict(
            tmp_data_dir, "tx-r",
            "## Final Recommendation\nreject\n\n## Rationale\nOOS collapsed.",
        )
        assert _verdict_is_reject("tx-r") is True

    def test_schedule_next_stage_marks_rejected_and_skips(
        self, tmp_data_dir, stubbed_scheduler,
    ):
        from sandbox_agent.pipeline.orchestrator import _schedule_next_stage

        state = init_pipeline("tx-skip", "d", pipeline_type="trading")
        state.stages[4].status = "completed"
        state.current_stage = 4
        save_state(state)
        self._write_verdict(
            tmp_data_dir, "tx-skip",
            "## Final Recommendation\nreject\n\n## Rationale\nno edge.",
        )

        _schedule_next_stage(state, 5)

        assert state.status == "completed_rejected"
        # No stage 5 task enqueued.
        added_names = [t.name for t in stubbed_scheduler.added]
        assert not any("paper_trading" in n for n in added_names)
        assert not any("_stage5" in n for n in added_names)

    def test_schedule_next_stage_promotes_normally(
        self, tmp_data_dir, stubbed_scheduler,
    ):
        from sandbox_agent.pipeline.orchestrator import _schedule_next_stage

        state = init_pipeline("tx-go", "d", pipeline_type="trading")
        state.stages[4].status = "completed"
        state.current_stage = 4
        save_state(state)
        self._write_verdict(
            tmp_data_dir, "tx-go",
            "## Final Recommendation\npromote\n\n## Rationale\nAll gates passed.",
        )

        _schedule_next_stage(state, 5)

        assert state.status != "completed_rejected"
        assert state.current_stage == 5


class TestStageInstructionOverrides:
    """DATA_DIR overrides must EXTEND the bundled instructions, not replace
    them — the agent writes extension-style overrides ("all bundled rules
    still apply") via the stage-6 review follow-up; returning only the
    override would strip the entire bundled workflow from the next run."""

    def test_override_composes_with_bundled(self, tmp_data_dir):
        override_dir = os.path.join(tmp_data_dir, "pipeline_stages", "trading")
        os.makedirs(override_dir)
        with open(os.path.join(override_dir, "stage_2_research_loop.md"), "w") as f:
            f.write("### Learned rule: validate metadata at init.")
        from sandbox_agent.pipeline.orchestrator import load_stage_instructions
        text = load_stage_instructions(2, "trading")
        assert "Learned rule: validate metadata at init." in text  # override present
        assert "required_data_types" in text  # bundled stage-2 text still present
        assert text.index("required_data_types") < text.index("Learned rule")

    def test_no_override_returns_bundled_unchanged(self, tmp_data_dir):
        from sandbox_agent.pipeline.orchestrator import load_stage_instructions
        text = load_stage_instructions(2, "trading")
        assert "required_data_types" in text
        assert "Learned overrides" not in text


class TestCompletionEmail:
    """Every terminal pipeline outcome (both pipeline types share these code
    paths) must email the owner with the result — no relying on the LLM to
    remember to notify."""

    @pytest.fixture
    def sent(self, monkeypatch):
        captured = []
        import sandbox_agent.tools.notification_tools as nt

        def fake_send(subject, body, to=None, html=True):
            captured.append((subject, body))
            return "Email sent to owner"

        monkeypatch.setattr(nt, "send_email_message", fake_send)
        return captured

    def _finish_all_but_last(self, project, ptype):
        from sandbox_agent.pipeline.orchestrator import (
            advance_pipeline, get_num_stages, init_pipeline, save_state)
        state = init_pipeline(project, "test pipeline", pipeline_type=ptype)
        n = get_num_stages(ptype)
        for i in range(1, n):
            state.stages[i].status = "completed"
        state.current_stage = n
        save_state(state)
        return n

    def test_startup_completion_sends_email(self, tmp_data_dir, sent, monkeypatch):
        from sandbox_agent.pipeline import orchestrator as o
        monkeypatch.setattr(o, "_schedule_stage", lambda *a, **k: None)
        n = self._finish_all_but_last("mail-startup", "startup")
        o.advance_pipeline("mail-startup", n, True, "review looks good")
        assert len(sent) == 1
        subject, body = sent[0]
        assert "mail-startup" in subject
        assert "completed" in subject.lower()
        assert "review" in body  # per-stage summary present

    def test_trading_reject_sends_email_with_verdict(self, tmp_data_dir, sent, monkeypatch):
        from sandbox_agent.pipeline import orchestrator as o
        monkeypatch.setattr(o, "_schedule_stage", lambda *a, **k: None)
        state = o.init_pipeline("mail-reject", "d", pipeline_type="trading")
        for i in range(1, 5):
            state.stages[i].status = "completed"
        state.current_stage = 4
        o.save_state(state)
        vdir = os.path.join(tmp_data_dir, "projects", "mail-reject", "pipeline")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "verdict.md"), "w") as f:
            f.write("# Verdict\n\n## Final Recommendation\n\nreject\n\n"
                    "## Rationale\n\nOOS Sharpe collapsed to 0.1.\n")
        o.advance_pipeline("mail-reject", 4, True, "verdict written")
        assert o.load_state("mail-reject").status == "completed_rejected"
        assert len(sent) == 1
        subject, body = sent[0]
        assert "rejected" in subject.lower()
        assert "OOS Sharpe collapsed" in body  # verdict excerpt included

    def test_exhausted_final_stage_sends_email(self, tmp_data_dir, sent, monkeypatch):
        from sandbox_agent.pipeline import orchestrator as o
        monkeypatch.setattr(o, "_schedule_stage", lambda *a, **k: None)
        monkeypatch.setattr(o, "_check_artifacts_exist", lambda *a, **k: True)
        n = self._finish_all_but_last("mail-exhausted", "startup")
        state = o.load_state("mail-exhausted")
        o._finalize_exhausted_stage(state, state.stages[n], "retry ceiling")
        o.save_state(state)
        assert len(sent) == 1
        assert "best effort" in sent[0][0].lower() or "exhausted" in sent[0][1].lower()

    def test_mid_pipeline_stage_does_not_email(self, tmp_data_dir, sent, monkeypatch):
        from sandbox_agent.pipeline import orchestrator as o
        monkeypatch.setattr(o, "_schedule_stage", lambda *a, **k: None)
        state = o.init_pipeline("mail-mid", "d", pipeline_type="startup")
        o.save_state(state)
        o.advance_pipeline("mail-mid", 1, True, "ok")
        assert sent == []


class TestCancelPipeline:
    """cancel_pipeline: the sanctioned way to stop a pipeline. Before this,
    the agent could only cancel stage TASKS — the pipeline state stayed
    'running' forever (zombie) and state.json is write-protected against
    direct edits, so there was NO way to stop a pipeline (2026-07-16)."""

    def _queue(self, monkeypatch, tmp_data_dir):
        from sandbox_agent.scheduler.task_queue import TaskQueue
        import sandbox_agent.scheduler.scheduler_tools as st
        tq = TaskQueue(data_dir=tmp_data_dir)
        monkeypatch.setattr(st, "get_task_queue", lambda: tq)
        return tq

    def test_cancels_state_tasks_and_lock(self, tmp_data_dir, monkeypatch):
        from sandbox_agent.pipeline import orchestrator as o
        tq = self._queue(monkeypatch, tmp_data_dir)
        state = o.init_pipeline("zombie", "d", pipeline_type="trading")
        o.save_state(state)
        stage_task = tq.add_task(name="pipeline:zombie:stage_2", description="d")
        tq.add_task(name="unrelated", description="d")
        # simulate the running stage holding the pipeline lock
        assert o.acquire_lock(stage_task.id)

        out = o.cancel_pipeline("zombie")
        assert "cancelled" in out.lower()
        st2 = o.load_state("zombie")
        assert st2.status == "cancelled"
        assert st2.lock_holder is None
        names = [t.name for t in tq.list_tasks()]
        assert "pipeline:zombie:stage_2" not in names
        assert "unrelated" in names  # untouched
        assert not os.path.exists(o.LOCK_FILE)  # lock released

    def test_cancelled_pipeline_can_be_rerun(self, tmp_data_dir, monkeypatch):
        from sandbox_agent.pipeline import orchestrator as o
        self._queue(monkeypatch, tmp_data_dir)
        state = o.init_pipeline("zombie2", "d", pipeline_type="startup")
        o.save_state(state)
        o.cancel_pipeline("zombie2")
        fresh = o.init_pipeline("zombie2", "d2", pipeline_type="startup")
        assert fresh.status == "running"  # reset, not the stale cancelled state

    def test_unknown_or_finished_pipeline(self, tmp_data_dir, monkeypatch):
        from sandbox_agent.pipeline import orchestrator as o
        self._queue(monkeypatch, tmp_data_dir)
        assert "No pipeline" in o.cancel_pipeline("nope")
        state = o.init_pipeline("done", "d", pipeline_type="startup")
        state.status = "completed"
        o.save_state(state)
        assert "already" in o.cancel_pipeline("done")

    def test_tool_registered(self):
        import sandbox_agent.pipeline.pipeline_tools  # noqa: F401 — registration on import
        from qwen_agent.tools.base import TOOL_REGISTRY
        from sandbox_agent.config import TOOL_LIST
        assert "cancel_pipeline" in TOOL_REGISTRY
        assert "cancel_pipeline" in TOOL_LIST
