"""Tests for pipeline stage runner — prompt building, artifact loading, part-completion detection."""

import json
import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.models import PipelineState, StageState
from sandbox_agent.pipeline.stage_runner import (
    _build_loop_state_block,
    _build_prompt,
    _check_verdict_skip,
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


class TestTimeBudgetBlock:

    def _build_budget_prompt(self, budget_seconds=28800, started_at=None,
                             part_done=0, max_parts=40):
        from datetime import datetime
        state = PipelineState(
            project_name="tx", description="d", pipeline_type="trading",
            current_stage=3,
            stages={
                3: StageState(
                    stage_number=3, stage_name="full_validation",
                    artifacts=["backtest/full/metrics.json"],
                    budget_seconds=budget_seconds,
                    max_part_completions=max_parts,
                    part_completion_count=part_done,
                    first_started_at=started_at,
                ),
            },
        )
        return _build_prompt(state, state.stages[3], "do the pipeline", "no artifacts")

    def test_no_budget_block_when_budget_not_set(self, sample_state):
        stage = sample_state.stages[2]  # startup stage, no budget
        prompt = _build_prompt(sample_state, stage, "instructions", "artifacts")
        assert "Time Budget" not in prompt

    def test_budget_block_appears_when_set(self):
        prompt = self._build_budget_prompt(budget_seconds=28800)
        assert "## Time Budget" in prompt
        assert "28800s" in prompt
        assert "Part-completions used: 0 of 40" in prompt

    def test_budget_block_shows_elapsed(self):
        from datetime import datetime, timedelta
        started = datetime.now() - timedelta(hours=2)
        prompt = self._build_budget_prompt(budget_seconds=28800, started_at=started)
        assert "## Time Budget" in prompt
        assert "Elapsed so far: 2h" in prompt
        # Remaining should be roughly 6h
        assert "Remaining: 6h" in prompt

    def test_budget_block_first_run_no_start(self):
        prompt = self._build_budget_prompt(budget_seconds=172800, started_at=None,
                                           part_done=0, max_parts=200)
        assert "172800s" in prompt
        assert "this is the first run" in prompt
        assert "Part-completions used: 0 of 200" in prompt

    def test_budget_block_counter_visible(self):
        prompt = self._build_budget_prompt(budget_seconds=28800, part_done=7, max_parts=40)
        assert "Part-completions used: 7 of 40" in prompt


class TestLoopStateBlock:

    def _trading_state(self):
        return PipelineState(
            project_name="test-project", description="a strategy",
            pipeline_type="trading", current_stage=2,
            stages={
                2: StageState(stage_number=2, stage_name="research_loop",
                              artifacts=["strategy/loop_state.json"]),
            },
        )

    def test_block_only_for_trading_stage_2(self, sample_state):
        # sample_state is startup — block should be None.
        assert _build_loop_state_block(sample_state, 2) is None

    def test_block_absent_for_trading_stage_other(self, tmp_data_dir):
        state = self._trading_state()
        assert _build_loop_state_block(state, 3) is None

    def test_missing_loop_state_tells_agent_to_init(self, tmp_data_dir):
        state = self._trading_state()
        os.makedirs(os.path.join(tmp_data_dir, "projects", "test-project"), exist_ok=True)
        block = _build_loop_state_block(state, 2)
        assert block is not None
        assert "first run" in block.lower()
        assert "init" in block.lower()

    def test_renders_pilot_history_as_table(self, tmp_data_dir):
        state = self._trading_state()
        strat = os.path.join(tmp_data_dir, "projects", "test-project", "strategy")
        os.makedirs(strat, exist_ok=True)
        loop_state = {
            "next_step": "run_pilot_backtest",
            "current_phase": "pilot",
            "oos_cutoff_date": "2024-04-01",
            "hypothesis_count": 1,
            "pilot_history": [
                {"hyp": 1, "iter": 1, "sharpe": 0.12, "trades": 8},
                {"hyp": 1, "iter": 2, "sharpe": 0.44, "trades": 18},
            ],
        }
        with open(os.path.join(strat, "loop_state.json"), "w") as f:
            json.dump(loop_state, f)
        block = _build_loop_state_block(state, 2)
        assert "next_step" in block
        assert "run_pilot_backtest" in block
        assert "pilot_history" in block
        assert "0.12" in block and "0.44" in block

    def test_trims_run_notes_to_last_three(self, tmp_data_dir):
        state = self._trading_state()
        strat = os.path.join(tmp_data_dir, "projects", "test-project", "strategy")
        os.makedirs(strat, exist_ok=True)
        notes = [
            {"id": f"r{i}", "step": "run_pilot_backtest", "hyp": 1, "iter": i,
             "what_i_did": f"iteration {i}"}
            for i in range(1, 6)
        ]
        with open(os.path.join(strat, "loop_state.json"), "w") as f:
            json.dump({"run_notes": notes}, f)
        block = _build_loop_state_block(state, 2)
        # Last 3 IDs present, earlier ones dropped.
        assert "r3" in block
        assert "r4" in block
        assert "r5" in block
        assert "r1" not in block
        assert "r2" not in block

    def test_includes_all_hypothesis_notes(self, tmp_data_dir):
        state = self._trading_state()
        strat = os.path.join(tmp_data_dir, "projects", "test-project", "strategy")
        os.makedirs(strat, exist_ok=True)
        hyp_notes = [
            {"hyp": 1, "final_sharpe": 0.44, "iterations_spent": 5,
             "why_abandoned": "short-side signal doesn't exist",
             "lessons_for_future_hypotheses": ["try mid-caps", "aggregate weekly"]},
            {"hyp": 2, "final_sharpe": 0.6, "iterations_spent": 4,
             "why_abandoned": "too much turnover",
             "lessons_for_future_hypotheses": ["reduce rebalance frequency"]},
        ]
        with open(os.path.join(strat, "loop_state.json"), "w") as f:
            json.dump({"hypothesis_notes": hyp_notes}, f)
        block = _build_loop_state_block(state, 2)
        assert "hyp 1" in block
        assert "hyp 2" in block
        assert "try mid-caps" in block
        assert "reduce rebalance frequency" in block


class TestVerdictSkip:

    def _trading_state(self, stage_number=5):
        return PipelineState(
            project_name="test-project", description="a strategy",
            pipeline_type="trading", current_stage=stage_number,
            stages={
                stage_number: StageState(
                    stage_number=stage_number,
                    stage_name="paper_trading" if stage_number == 5 else "review",
                    status="running",
                ),
            },
        )

    def test_no_verdict_file_proceeds(self, tmp_data_dir):
        state = self._trading_state()
        os.makedirs(os.path.join(tmp_data_dir, "projects", "test-project"), exist_ok=True)
        assert _check_verdict_skip(state, 5) is None

    def test_verdict_promote_proceeds(self, tmp_data_dir):
        state = self._trading_state()
        vdir = os.path.join(tmp_data_dir, "projects", "test-project", "pipeline")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "verdict.md"), "w") as f:
            f.write("## Final Recommendation\npromote\n\n## Rationale\nAll gates passed.\n")
        assert _check_verdict_skip(state, 5) is None
        assert state.status == "running"

    def test_verdict_reject_skips_and_marks_completed_rejected(self, tmp_data_dir, monkeypatch):
        # save_state writes to disk; stub it to no-op for the test.
        monkeypatch.setattr(
            "sandbox_agent.pipeline.stage_runner.save_state", lambda s: None
        )
        state = self._trading_state(5)
        vdir = os.path.join(tmp_data_dir, "projects", "test-project", "pipeline")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "verdict.md"), "w") as f:
            f.write("## Final Recommendation\nreject\n\n## Rationale\nOOS collapse.\n")
        result = _check_verdict_skip(state, 5)
        assert result is not None
        assert "reject" in result
        assert state.status == "completed_rejected"
        assert state.stages[5].status == "completed"

    def test_verdict_skip_works_for_stage_6(self, tmp_data_dir, monkeypatch):
        monkeypatch.setattr(
            "sandbox_agent.pipeline.stage_runner.save_state", lambda s: None
        )
        state = self._trading_state(6)
        vdir = os.path.join(tmp_data_dir, "projects", "test-project", "pipeline")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "verdict.md"), "w") as f:
            f.write("## Final Recommendation\nreject\n")
        result = _check_verdict_skip(state, 6)
        assert result is not None
        assert state.status == "completed_rejected"


class TestArtifactTotalBudget:

    def test_total_cap_across_artifacts(self, tmp_data_dir):
        # Startup stage 6 has 5 inputs; each per-artifact cap is 120k chars,
        # so 5 near-cap artifacts used to produce an unbounded ~600k-char
        # block (~150k tokens) — a base prompt no compaction tier can shrink.
        # A TOTAL budget must bound the block regardless of artifact count.
        proj = os.path.join(tmp_data_dir, "projects", "test-project")
        for rel in ["research/market-research.md", "business/brd.md",
                    "product/prd.md", "business/vc-pitch.md", "mvp/README.md"]:
            full = os.path.join(proj, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write("A" * 119_000)

        result = _load_artifacts(_state(), 6)
        assert len(result) <= 245_000  # 240k budget + headers/separators
        # Every artifact still represented (later ones truncated, not dropped)
        for rel in ["market-research", "brd", "prd", "vc-pitch", "README"]:
            assert rel in result

    def test_small_artifacts_unaffected(self, tmp_data_dir):
        research_dir = os.path.join(tmp_data_dir, "projects", "test-project", "research")
        os.makedirs(research_dir, exist_ok=True)
        with open(os.path.join(research_dir, "market-research.md"), "w") as f:
            f.write("# Market Research\n\nTAM is $5B")
        result = _load_artifacts(_state(), 2)
        assert "TAM is $5B" in result
        assert "truncated" not in result
