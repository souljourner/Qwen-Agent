"""Tests for pipeline stage runner — prompt building, artifact loading, part-completion detection."""

import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.models import PipelineState, StageState
from sandbox_agent.pipeline.stage_runner import (
    _build_prompt,
    _detect_part_completion,
    _load_artifacts,
)


@pytest.fixture
def tmp_data_dir(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.stage_runner.DATA_DIR", d)
    os.makedirs(os.path.join(d, "projects", "test-project"), exist_ok=True)
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_state():
    return PipelineState(
        project_name="test-project",
        description="AI pet health monitoring for busy pet owners",
        current_stage=2,
        stages={
            1: StageState(stage_number=1, stage_name="market_research", status="completed",
                          artifacts=["research/market-research.md"]),
            2: StageState(stage_number=2, stage_name="brd", status="scheduled", run_count=1,
                          notes="### Attempt 1 feedback:\nMissing scalability section",
                          artifacts=["business/brd.md"]),
        },
    )


def _state(project="test-project", pipeline_type="startup"):
    return PipelineState(
        project_name=project,
        description="test",
        pipeline_type=pipeline_type,
    )


class TestLoadArtifacts:

    def test_no_inputs_for_stage_1(self, tmp_data_dir):
        result = _load_artifacts(_state(), 1)
        assert "No input artifacts" in result

    def test_loads_existing_artifact(self, tmp_data_dir):
        research_dir = os.path.join(tmp_data_dir, "projects", "test-project", "research")
        os.makedirs(research_dir, exist_ok=True)
        with open(os.path.join(research_dir, "market-research.md"), "w") as f:
            f.write("# Market Research\n\nTAM is $5B")

        result = _load_artifacts(_state(), 2)
        assert "Market Research" in result
        assert "TAM is $5B" in result

    def test_missing_artifact_noted(self, tmp_data_dir):
        result = _load_artifacts(_state(), 2)
        assert "Not yet created" in result

    def test_large_artifact_truncated(self, tmp_data_dir):
        research_dir = os.path.join(tmp_data_dir, "projects", "test-project", "research")
        os.makedirs(research_dir, exist_ok=True)
        with open(os.path.join(research_dir, "market-research.md"), "w") as f:
            f.write("x" * 150000)

        result = _load_artifacts(_state(), 2)
        assert "truncated" in result


class TestBuildPrompt:

    def test_includes_project_info(self, sample_state):
        stage = sample_state.stages[2]
        prompt = _build_prompt(sample_state, stage, "Do research", "No artifacts")
        assert "test-project" in prompt
        assert "AI pet health monitoring" in prompt

    def test_includes_instructions(self, sample_state):
        stage = sample_state.stages[2]
        prompt = _build_prompt(sample_state, stage, "## Instructions\nDo the BRD", "No artifacts")
        assert "Do the BRD" in prompt

    def test_includes_attempt_count(self, sample_state):
        stage = sample_state.stages[2]
        prompt = _build_prompt(sample_state, stage, "instructions", "artifacts")
        assert "Attempt: 2 of 5" in prompt

    def test_includes_previous_notes(self, sample_state):
        stage = sample_state.stages[2]
        prompt = _build_prompt(sample_state, stage, "instructions", "artifacts")
        assert "Missing scalability section" in prompt

    def test_includes_output_requirements(self, sample_state):
        stage = sample_state.stages[2]
        prompt = _build_prompt(sample_state, stage, "instructions", "artifacts")
        assert "project_write_file" in prompt
        assert "business/brd.md" in prompt

    def test_no_notes_section_when_empty(self, sample_state):
        stage = sample_state.stages[1]  # Stage 1 has no notes
        prompt = _build_prompt(sample_state, stage, "instructions", "artifacts")
        assert "Previous Attempts" not in prompt


class TestDetectPartCompletion:

    def test_explicit_part_completion_signal(self, tmp_data_dir):
        assert _detect_part_completion(_state(), 5, "Saved progress. Status: part-completion") is True

    def test_tool_call_exhaustion_signal(self, tmp_data_dir):
        assert _detect_part_completion(_state(), 5, "Ran out of tool calls, will continue next run") is True

    def test_normal_completion_not_detected(self, tmp_data_dir):
        assert _detect_part_completion(_state(), 1, "Research complete, saved to file") is False

    def test_mvp_partial_files(self, tmp_data_dir):
        mvp_dir = os.path.join(tmp_data_dir, "projects", "test-project", "mvp")
        os.makedirs(mvp_dir, exist_ok=True)
        with open(os.path.join(mvp_dir, "app.py"), "w") as f:
            f.write("from flask import Flask")
        # Has code but no README or tests
        assert _detect_part_completion(_state(), 5, "Built backend") is True

    def test_mvp_complete_files(self, tmp_data_dir):
        mvp_dir = os.path.join(tmp_data_dir, "projects", "test-project", "mvp")
        os.makedirs(os.path.join(mvp_dir, "tests"), exist_ok=True)
        with open(os.path.join(mvp_dir, "app.py"), "w") as f:
            f.write("from flask import Flask")
        with open(os.path.join(mvp_dir, "README.md"), "w") as f:
            f.write("# MVP")
        with open(os.path.join(mvp_dir, "tests", "test_app.py"), "w") as f:
            f.write("def test_hello(): pass")
        assert _detect_part_completion(_state(), 5, "MVP complete") is False

    def test_non_mvp_stage_no_file_check(self, tmp_data_dir):
        # MVP heuristic only applies to startup stage 5
        assert _detect_part_completion(_state(), 2, "BRD written") is False

    def test_trading_stage_5_no_mvp_heuristic(self, tmp_data_dir):
        # Trading pipeline stage 5 is paper trading, not MVP — MVP file heuristic must not fire
        mvp_dir = os.path.join(tmp_data_dir, "projects", "test-project", "mvp")
        os.makedirs(mvp_dir, exist_ok=True)
        with open(os.path.join(mvp_dir, "app.py"), "w") as f:
            f.write("from flask import Flask")
        assert _detect_part_completion(_state(pipeline_type="trading"), 5, "Scaffold complete") is False
