"""Tests for trading-pipeline support: registry, backcompat, validator, tool wiring."""

import json
import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.models import PipelineState
from sandbox_agent.pipeline.orchestrator import (
    STAGE_REGISTRY,
    STARTUP_STAGES,
    TRADING_STAGES,
    get_acceptance_path,
    get_instructions_dir,
    get_num_stages,
    get_stages,
    init_pipeline,
    load_state,
)
from sandbox_agent.pipeline.pipeline_tools import (
    StartTradingPipeline,
    _validate_strategy,
)


@pytest.fixture
def tmp_data_dir(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator._projects_dir",
                        lambda: os.path.join(d, "projects"))
    yield d
    shutil.rmtree(d)


class TestStageRegistry:

    def test_registry_has_both_pipeline_types(self):
        assert "startup" in STAGE_REGISTRY
        assert "trading" in STAGE_REGISTRY

    def test_get_stages_startup_unchanged(self):
        stages = get_stages("startup")
        assert stages is STARTUP_STAGES
        assert stages[1]["name"] == "market_research"
        assert stages[6]["name"] == "review"

    def test_get_stages_trading(self):
        stages = get_stages("trading")
        assert stages is TRADING_STAGES
        assert stages[1]["name"] == "data_landscape"
        assert stages[2]["name"] == "research_loop"
        assert stages[3]["name"] == "full_validation"
        assert stages[4]["name"] == "verdict"
        assert stages[5]["name"] == "paper_trading"
        assert stages[6]["name"] == "review"

    def test_trading_stage_3_requires_metrics_outputs(self):
        assert "backtest/full/metrics.json" in TRADING_STAGES[3]["outputs"]
        assert "backtest/full/results.md" in TRADING_STAGES[3]["outputs"]

    def test_trading_stage_3_requires_metric_sections(self):
        required = TRADING_STAGES[3]["required_sections"]
        assert "OOS Sharpe" in required
        assert "Walk-Forward" in required
        assert "Trade Count" in required
        assert "t-Statistic" in required

    def test_trading_stage_2_has_budget_and_ceiling(self):
        # Only stage 2 (research_loop) is budgeted in the redesigned pipeline.
        assert TRADING_STAGES[2]["budget_seconds"] == 172800
        assert TRADING_STAGES[2]["max_part_completions"] == 150

    def test_trading_other_stages_have_no_budget(self):
        for n in (1, 3, 4, 5, 6):
            assert "budget_seconds" not in TRADING_STAGES[n]
            assert "max_part_completions" not in TRADING_STAGES[n]

    def test_get_num_stages_is_six(self):
        assert get_num_stages("startup") == 6
        assert get_num_stages("trading") == 6

    def test_unknown_pipeline_type_raises(self):
        with pytest.raises(ValueError):
            get_stages("nonexistent")

    def test_instructions_and_acceptance_paths_diverge(self):
        assert get_instructions_dir("startup") != get_instructions_dir("trading")
        assert get_acceptance_path("startup") != get_acceptance_path("trading")

    def test_stage_file_paths_exist(self):
        # Sanity check: the trading stage markdown files were authored
        for n in range(1, 7):
            stage_name = TRADING_STAGES[n]["name"]
            md = get_instructions_dir("trading") / f"stage_{n}_{stage_name}.md"
            assert md.exists(), f"Missing stage file: {md}"
        assert get_acceptance_path("trading").exists()


class TestBackcompat:

    def test_old_state_json_defaults_to_startup(self):
        # Simulate a state.json written before pipeline_type existed
        raw = {
            "project_name": "legacy-project",
            "description": "An old project",
            "current_stage": 2,
            "status": "running",
            "stages": {},
        }
        state = PipelineState(**raw)
        assert state.pipeline_type == "startup"

    def test_new_state_json_with_trading_type(self):
        raw = {
            "project_name": "trading-project",
            "description": "A new strategy",
            "pipeline_type": "trading",
            "current_stage": 1,
            "status": "running",
            "stages": {},
        }
        state = PipelineState(**raw)
        assert state.pipeline_type == "trading"

    def test_init_pipeline_persists_type(self, tmp_data_dir, monkeypatch):
        # Stub the scheduler to avoid DB dependencies
        from sandbox_agent.pipeline import orchestrator
        monkeypatch.setattr(orchestrator, "DATA_DIR", tmp_data_dir)

        state = init_pipeline("tx-proj", "a strategy", pipeline_type="trading")
        assert state.pipeline_type == "trading"
        assert state.stages[1].stage_name == "data_landscape"

        # Round-trip through disk
        loaded = load_state("tx-proj")
        assert loaded is not None
        assert loaded.pipeline_type == "trading"
        assert loaded.stages[4].stage_name == "verdict"

    def test_init_pipeline_rejects_unknown_type(self, tmp_data_dir, monkeypatch):
        from sandbox_agent.pipeline import orchestrator
        monkeypatch.setattr(orchestrator, "DATA_DIR", tmp_data_dir)
        with pytest.raises(ValueError):
            init_pipeline("bad-proj", "x", pipeline_type="nonsense")


class TestValidateStrategy:

    def test_rejects_too_short(self):
        ok, _ = _validate_strategy("momentum")
        assert ok is False

    def test_rejects_missing_universe(self):
        ok, feedback = _validate_strategy(
            "A momentum signal using 20/50 crossover with 2% stop loss position sizing."
        )
        assert ok is False
        assert "universe" in feedback.lower()

    def test_rejects_missing_signal(self):
        ok, feedback = _validate_strategy(
            "Trade S&P 500 stocks with 2% stop loss and 5% position cap on drawdown."
        )
        assert ok is False
        assert "signal" in feedback.lower()

    def test_accepts_universe_plus_signal(self):
        ok, feedback = _validate_strategy(
            "Trade QQQ stocks using a 20-day / 50-day SMA crossover momentum signal."
        )
        assert ok is True

    def test_surfaces_missing_risk_in_feedback(self):
        ok, feedback = _validate_strategy(
            "Trade QQQ stocks using a 20-day / 50-day SMA crossover momentum signal."
        )
        assert ok is True
        assert "risk" in feedback.lower()

    def test_accepts_full_description(self):
        ok, feedback = _validate_strategy(
            "Universe: QQQ constituents. Signal: 20/50 SMA crossover momentum. "
            "Risk: 2% stop loss, max 5% position size, 15% drawdown circuit breaker."
        )
        assert ok is True
        assert "proceed" in feedback.lower()


class TestStartTradingPipelineTool:

    def test_tool_name_and_description(self):
        tool = StartTradingPipeline()
        assert tool.name == "start_trading_pipeline"
        assert "trading" in tool.description.lower()
        assert "lock" in tool.description.lower()

    def test_invalid_description_returns_feedback_not_pipeline(self, tmp_data_dir, monkeypatch):
        from sandbox_agent.pipeline import orchestrator
        monkeypatch.setattr(orchestrator, "DATA_DIR", tmp_data_dir)

        tool = StartTradingPipeline()
        result = tool.call(json.dumps({"name": "bad", "description": "too short"}))
        assert "clarification" in result.lower() or "too short" in result.lower()
