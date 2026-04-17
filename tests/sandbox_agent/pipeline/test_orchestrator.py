"""Tests for pipeline orchestrator — state machine, lock, stage management."""

import json
import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.models import PipelineState, StageState
from sandbox_agent.pipeline.orchestrator import (
    STARTUP_STAGES as STAGES,
    acquire_lock,
    clear_lock_on_startup,
    get_num_stages,
    init_pipeline,
    load_stage_instructions,
    load_state,
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
