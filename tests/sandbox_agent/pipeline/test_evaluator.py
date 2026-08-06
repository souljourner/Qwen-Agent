"""Tests for pipeline stage evaluator — programmatic checks."""

import json
import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.evaluator import (
    _check_backtest_placeholders,
    _check_backtest_uses_real_signals,
    _check_cache_coverage,
    _check_full_validation_metrics,
    _check_hypothesis_data_feasibility,
    _check_llm_cache_variance,
    _check_paper_deploy_compiles,
    _check_pilot_avoids_oos,
    _check_strategy_reads_llm_cache,
    _check_verdict_matches_metrics,
    _decide_from_pilot_history,
    _normalize_data_type,
    _programmatic_checks,
    _stage_specific_checks,
)
from sandbox_agent.pipeline.orchestrator import STARTUP_STAGES as STAGES


@pytest.fixture
def tmp_project_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestProgrammaticChecks:

    def test_missing_artifact_fails(self, tmp_project_dir):
        stage_defn = STAGES[1]
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert not passed
        assert "Missing artifact" in feedback

    def test_empty_artifact_fails(self, tmp_project_dir):
        stage_defn = STAGES[1]
        # Create the artifact but make it too short
        artifact_path = os.path.join(tmp_project_dir, "research")
        os.makedirs(artifact_path, exist_ok=True)
        with open(os.path.join(artifact_path, "market-research.md"), "w") as f:
            f.write("short")
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert not passed
        assert "too short" in feedback

    def test_missing_section_fails(self, tmp_project_dir):
        stage_defn = STAGES[1]
        artifact_path = os.path.join(tmp_project_dir, "research")
        os.makedirs(artifact_path, exist_ok=True)
        # Write enough content but missing required sections
        with open(os.path.join(artifact_path, "market-research.md"), "w") as f:
            f.write("# Market Research Report\n\n" + "Lorem ipsum dolor sit amet. " * 50)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert not passed
        assert "Missing section" in feedback

    def test_valid_artifact_passes(self, tmp_project_dir):
        stage_defn = STAGES[1]
        artifact_path = os.path.join(tmp_project_dir, "research")
        os.makedirs(artifact_path, exist_ok=True)
        content = (
            "# Market Research\n\n"
            "## Market Size\nThe total addressable market is $10B. " * 10 + "\n\n"
            "## Competitors\nThere are several key competitors. " * 10 + "\n\n"
            "## Target Customers\nOur target demographic is. " * 10 + "\n\n"
            "## Timing\nThe market is ready because. " * 10
        )
        with open(os.path.join(artifact_path, "market-research.md"), "w") as f:
            f.write(content)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert passed

    def test_brd_checks(self, tmp_project_dir):
        stage_defn = STAGES[2]
        brd_path = os.path.join(tmp_project_dir, "business")
        os.makedirs(brd_path, exist_ok=True)
        content = (
            "# BRD\n\n"
            "## Branding\nOur brand is. " * 10 + "\n\n"
            "## Legal\nKey legal considerations. " * 10 + "\n\n"
            "## Scalability\nWe scale by. " * 10 + "\n\n"
            "## Operations\nDay to day operations. " * 10 + "\n\n"
            "## Finance\nOur revenue model. " * 10
        )
        with open(os.path.join(brd_path, "brd.md"), "w") as f:
            f.write(content)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert passed

    def test_mvp_checks_readme_exists(self, tmp_project_dir):
        stage_defn = STAGES[5]
        mvp_path = os.path.join(tmp_project_dir, "mvp")
        os.makedirs(mvp_path, exist_ok=True)
        with open(os.path.join(mvp_path, "README.md"), "w") as f:
            f.write("# MVP\n\n" + "This is the MVP documentation. " * 20)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert passed  # MVP only requires README.md to exist (no required sections)


def _write_results_md(project_dir, body):
    bt = os.path.join(project_dir, "backtest")
    os.makedirs(bt, exist_ok=True)
    with open(os.path.join(bt, "results.md"), "w") as f:
        f.write(body)


class TestBacktestPlaceholderCheck:

    def test_placeholder_tbd_rejected(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "## Sharpe\nTBD\n"
            "## Max Drawdown\n-18.4%\n"
            "## CAGR\n14.2%\n"
            "## Win Rate\n42.8%\n"
        ))
        passed, feedback = _check_backtest_placeholders(tmp_project_dir)
        assert not passed
        assert "placeholder 'TBD'" in feedback
        assert "sharpe" in feedback.lower()

    def test_missing_heading_rejected(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "## Sharpe\n0.5\n"
            "## CAGR\n14.2%\n"
            "## Win Rate\n42.8%\n"
        ))
        passed, feedback = _check_backtest_placeholders(tmp_project_dir)
        assert not passed
        assert "max drawdown" in feedback.lower()

    def test_no_numeric_value_rejected(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "## Sharpe\nstrong risk-adjusted returns\n"
            "## Max Drawdown\n-18.4%\n"
            "## CAGR\n14.2%\n"
            "## Win Rate\n42.8%\n"
        ))
        passed, feedback = _check_backtest_placeholders(tmp_project_dir)
        assert not passed
        assert "no numeric value" in feedback

    def test_to_be_calculated_rejected(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "## Sharpe\n0.87\n"
            "## Max Drawdown\n-18.4%\n"
            "## CAGR\nto be calculated from equity curve (14.2%)\n"
            "## Win Rate\n42.8%\n"
        ))
        passed, feedback = _check_backtest_placeholders(tmp_project_dir)
        assert not passed
        assert "to be calculated" in feedback

    def test_real_numbers_pass(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "# Backtest Results\n\n"
            "## Sharpe\n0.87 (annualized)\n"
            "## Max Drawdown\n-18.4%\n"
            "## CAGR\n14.2%\n"
            "## Win Rate\n42.8%\n"
            "## Interpretation\nModest but honest.\n"
        ))
        passed, _ = _check_backtest_placeholders(tmp_project_dir)
        assert passed

    def test_negative_and_zero_values_accepted(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "## Sharpe\n-0.2\n"
            "## Max Drawdown\n0%\n"
            "## CAGR\n0.5%\n"
            "## Win Rate\n51\n"
        ))
        passed, _ = _check_backtest_placeholders(tmp_project_dir)
        assert passed

    def test_missing_file(self, tmp_project_dir):
        passed, feedback = _check_backtest_placeholders(tmp_project_dir)
        assert not passed
        assert "Missing artifact" in feedback


class TestBacktestRealSignalsCheck:

    def test_hash_based_simulation_rejected(self, tmp_project_dir):
        # Mirrors the prem14a-event-driven failure mode.
        _write_results_md(tmp_project_dir, (
            "# Backtest Results\n\n"
            "**Signal Generation:** Deterministic hash-based simulation (mimics LLM output)\n\n"
            "## Sharpe\n2.90\n## Max Drawdown\n-7.49%\n"
            "## CAGR\n83.9%\n## Win Rate\n70%\n"
        ))
        passed, feedback = _check_backtest_uses_real_signals(tmp_project_dir)
        assert not passed
        assert "fake signals" in feedback
        assert "hash-based" in feedback

    def test_mock_signals_rejected(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "## Strategy\nWe generated mock signals for each ticker.\n"
            "## Sharpe\n1.2\n## Max Drawdown\n-10%\n"
            "## CAGR\n15%\n## Win Rate\n60%\n"
        ))
        passed, feedback = _check_backtest_uses_real_signals(tmp_project_dir)
        assert not passed
        assert "mock signal" in feedback.lower()

    def test_real_signals_pass(self, tmp_project_dir):
        _write_results_md(tmp_project_dir, (
            "# Backtest Results\n\n"
            "**Signal Generation:** LLM classifications from backtest/llm_cache/\n"
            "## Sharpe\n0.87\n## Max Drawdown\n-18.4%\n"
            "## CAGR\n14.2%\n## Win Rate\n42.8%\n"
        ))
        passed, feedback = _check_backtest_uses_real_signals(tmp_project_dir)
        assert passed, feedback

    def test_missing_file(self, tmp_project_dir):
        passed, feedback = _check_backtest_uses_real_signals(tmp_project_dir)
        assert not passed
        assert "Missing artifact" in feedback

    # Stage 4 is now the verdict stage — mock-signal phrase blacklist lives
    # inside evaluate_trading_stage_2_decision (the research-loop gates),
    # not in the stage-4 dispatcher. See TestPilotUsesRealSignals below.


class TestPaperDeployCompileCheck:

    def test_missing_file(self, tmp_project_dir):
        passed, feedback = _check_paper_deploy_compiles(tmp_project_dir)
        assert not passed
        assert "Missing artifact" in feedback

    def test_valid_python_passes(self, tmp_project_dir):
        paper = os.path.join(tmp_project_dir, "paper")
        os.makedirs(paper, exist_ok=True)
        with open(os.path.join(paper, "deploy.py"), "w") as f:
            f.write("import sys\n\ndef main():\n    print('ok')\n\nif __name__ == '__main__':\n    main()\n")
        passed, feedback = _check_paper_deploy_compiles(tmp_project_dir)
        assert passed, feedback

    def test_syntax_error_rejected(self, tmp_project_dir):
        paper = os.path.join(tmp_project_dir, "paper")
        os.makedirs(paper, exist_ok=True)
        with open(os.path.join(paper, "deploy.py"), "w") as f:
            f.write("def main(:\n    print('broken'\n")
        passed, feedback = _check_paper_deploy_compiles(tmp_project_dir)
        assert not passed
        assert "does not compile" in feedback


class TestStageSpecificDispatcher:

    def test_startup_passes_through(self, tmp_project_dir):
        # Startup pipeline has no stage-specific check; dispatcher returns (True, "")
        passed, feedback = _stage_specific_checks(tmp_project_dir, "startup", 4)
        assert passed
        assert feedback == ""

    def test_trading_stage_3_routes_to_full_validation_check(self, tmp_project_dir):
        # No metrics.json → full-validation check rejects
        passed, feedback = _stage_specific_checks(tmp_project_dir, "trading", 3)
        assert not passed
        assert "backtest/full/metrics.json" in feedback

    def test_trading_stage_4_routes_to_verdict_check(self, tmp_project_dir):
        # No verdict.md → verdict check rejects
        passed, feedback = _stage_specific_checks(tmp_project_dir, "trading", 4)
        assert not passed
        assert "pipeline/verdict.md" in feedback

    def test_trading_stage_5_routes_to_compile_check(self, tmp_project_dir):
        # No deploy.py → compile check rejects
        passed, feedback = _stage_specific_checks(tmp_project_dir, "trading", 5)
        assert not passed
        assert "paper/deploy.py" in feedback

    def test_trading_stage_1_no_specific_check(self, tmp_project_dir):
        passed, feedback = _stage_specific_checks(tmp_project_dir, "trading", 1)
        assert passed
        assert feedback == ""

    def test_trading_stage_2_no_specific_check(self, tmp_project_dir):
        # Stage 2 uses evaluate_trading_stage_2_decision, not _stage_specific_checks.
        passed, feedback = _stage_specific_checks(tmp_project_dir, "trading", 2)
        assert passed
        assert feedback == ""


# ---------------------------------------------------------------------------
# Trading stage 2 (research loop) gates
# ---------------------------------------------------------------------------


def _write(project_dir, relpath, content):
    full = os.path.join(project_dir, relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)


def _write_json(project_dir, relpath, obj):
    _write(project_dir, relpath, json.dumps(obj))


class TestHypothesisDataFeasibility:

    def test_passes_when_no_hypothesis_file_yet(self, tmp_project_dir):
        # First run: hypothesis not yet written — gate permissive.
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed

    def test_rejects_when_required_data_types_missing(self, tmp_project_dir):
        _write(tmp_project_dir, "strategy/hypothesis_v1.md",
               "# Hypothesis\n\nSomething goes up when X happens.")
        _write(tmp_project_dir, "research/data-landscape.md",
               "## Data Sources\nSEC EDGAR 8-K filings...")
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert not passed
        assert "required_data_types" in msg

    def test_rejects_when_required_type_absent_from_landscape(self, tmp_project_dir):
        _write(tmp_project_dir, "strategy/hypothesis_v1.md",
               "# Hypothesis\nrequired_data_types: [Twitter feed, options chain]")
        _write(tmp_project_dir, "research/data-landscape.md",
               "## Data Sources\nOnly SEC EDGAR 8-K filings available.")
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert not passed
        assert "Twitter feed" in msg or "options chain" in msg

    def test_passes_when_landscape_mentions_types(self, tmp_project_dir):
        _write(tmp_project_dir, "strategy/hypothesis_v1.md",
               "# Hypothesis\nrequired_data_types: [SEC 8-K, PRNewswire]")
        _write(tmp_project_dir, "research/data-landscape.md",
               "## Data Sources\nSEC 8-K filings and PRNewswire press releases.")
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed, msg

    def test_accepts_markdown_list_form(self, tmp_project_dir):
        _write(tmp_project_dir, "strategy/hypothesis_v1.md", (
            "# Hypothesis\n\n"
            "## Required Data Types\n"
            "- SEC 8-K\n"
            "- PRNewswire\n"
        ))
        _write(tmp_project_dir, "research/data-landscape.md",
               "## Data Sources\nSEC 8-K and PRNewswire are covered.")
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed, msg

    def test_accepts_backticked_bullets_with_descriptions(self, tmp_project_dir):
        # Incident replay (neumann-four-corners-v3): a Sharpe-2.221 hypothesis
        # was abandoned because bullets carried backticks + descriptions and
        # the whole line became the search needle. Formatting must not be able
        # to fail a hypothesis whose data types ARE in the landscape.
        _write(tmp_project_dir, "strategy/hypothesis_v1.md", (
            "# Hypothesis\n\n"
            "## required_data_types\n"
            "- `yfinance_ohlcv` \u2014 daily OHLCV bars for the universe\n"
            "- `eodhd_earnings`: EPS surprise history\n"
            "- **SEC 8-K** - material contract filings\n"
        ))
        _write(tmp_project_dir, "research/data-landscape.md", (
            "## Data Sources\n"
            "DATA_TYPE: yfinance_ohlcv\nDATA_TYPE: eodhd_earnings\n"
            "SEC 8-K filings from EDGAR.\n"
        ))
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed, msg

    def test_accepts_bold_and_quoted_inline_items(self, tmp_project_dir):
        _write(tmp_project_dir, "strategy/hypothesis_v1.md",
               "# Hypothesis\nrequired_data_types: [\"SEC 8-K\", **PRNewswire**]")
        _write(tmp_project_dir, "research/data-landscape.md",
               "## Data Sources\nSEC 8-K filings and PRNewswire press releases.")
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed, msg

    def test_word_fallback_matches_punctuated_landscape(self, tmp_project_dir):
        # Needle "IEX news feed" vs landscape "news feed (IEX)" — exact
        # substring fails but every significant word is present.
        _write(tmp_project_dir, "strategy/hypothesis_v1.md", (
            "# Hypothesis\n\n"
            "## Required Data Types\n"
            "- IEX news feed\n"
        ))
        _write(tmp_project_dir, "research/data-landscape.md",
               "## Data Sources\nWe have a news feed (IEX) with tick coverage.")
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed, msg

    def test_still_rejects_genuinely_missing_type(self, tmp_project_dir):
        # Leniency is about formatting, not substance: a data type that is
        # truly absent from the landscape must still fail, and the feedback
        # must show the CLEAN parsed name plus a format hint.
        _write(tmp_project_dir, "strategy/hypothesis_v1.md", (
            "# Hypothesis\n\n"
            "## required_data_types\n"
            "- `satellite parking-lot imagery` \u2014 weekly car counts\n"
        ))
        _write(tmp_project_dir, "research/data-landscape.md",
               "## Data Sources\nDATA_TYPE: yfinance_ohlcv\n")
        passed, msg = _check_hypothesis_data_feasibility(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert not passed
        assert "satellite parking-lot imagery" in msg
        assert "required_data_types" in msg  # format hint present
        # Feedback must steer the agent to fix the DECLARATION, not abandon
        # the hypothesis (v4 abandoned a good hypothesis over this), and
        # must list the names that ARE available.
        assert "Keep the hypothesis" in msg
        assert "yfinance_ohlcv" in msg  # available names listed from landscape


class TestNormalizeDataType:

    def test_backtick_span_wins(self):
        assert _normalize_data_type("`SEC 8-K` \u2014 material contracts") == "SEC 8-K"

    def test_strips_formatting_and_description(self):
        assert _normalize_data_type("**IEX news feed**: realtime ticks") == "IEX news feed"
        assert _normalize_data_type("'eodhd_earnings' - EPS surprises") == "eodhd_earnings"

    def test_hyphenated_names_survive(self):
        # No spaces around the hyphen — must NOT be treated as a description
        # separator.
        assert _normalize_data_type("SEC 8-K") == "SEC 8-K"
        assert _normalize_data_type("point-in-time fundamentals") == "point-in-time fundamentals"

    def test_plain_name_unchanged(self):
        assert _normalize_data_type("yfinance_ohlcv") == "yfinance_ohlcv"


class TestPilotAvoidsOOS:

    def test_passes_before_cutoff_set(self, tmp_project_dir):
        passed, msg = _check_pilot_avoids_oos(tmp_project_dir, {})
        assert passed

    def test_rejects_pilot_code_referencing_oos_dir(self, tmp_project_dir):
        _write(tmp_project_dir, "backtest/pilot/strategy_v1.py",
               "path = 'data/processed/oos/aapl.parquet'\n")
        passed, msg = _check_pilot_avoids_oos(
            tmp_project_dir, {"oos_cutoff_date": "2024-04-01"}
        )
        assert not passed
        assert "OOS" in msg or "oos" in msg

    def test_cutoff_rewrite_rejected(self, tmp_project_dir):
        # First pass — lay down the shadow with the original cutoff.
        passed, _ = _check_pilot_avoids_oos(
            tmp_project_dir, {"oos_cutoff_date": "2024-04-01"}
        )
        assert passed
        # Second pass with a different cutoff should be rejected.
        passed, msg = _check_pilot_avoids_oos(
            tmp_project_dir, {"oos_cutoff_date": "2024-06-01"}
        )
        assert not passed
        assert "frozen" in msg.lower() or "never be rewritten" in msg.lower()

    def test_same_cutoff_ok(self, tmp_project_dir):
        passed, _ = _check_pilot_avoids_oos(
            tmp_project_dir, {"oos_cutoff_date": "2024-04-01"}
        )
        assert passed
        passed, _ = _check_pilot_avoids_oos(
            tmp_project_dir, {"oos_cutoff_date": "2024-04-01"}
        )
        assert passed


class TestStrategyReadsLLMCache:

    def test_rejects_when_no_pilot_dir(self, tmp_project_dir):
        passed, msg = _check_strategy_reads_llm_cache(tmp_project_dir)
        assert not passed
        assert "backtest/pilot" in msg

    def test_rejects_when_no_strategy_files(self, tmp_project_dir):
        os.makedirs(os.path.join(tmp_project_dir, "backtest", "pilot"))
        passed, msg = _check_strategy_reads_llm_cache(tmp_project_dir)
        assert not passed
        assert "strategy_v" in msg

    def test_rejects_when_strategy_does_not_reference_cache(self, tmp_project_dir):
        _write(tmp_project_dir, "backtest/pilot/strategy_v1.py",
               "import hashlib\n\nsig = hashlib.md5(b'a').hexdigest()\n")
        passed, msg = _check_strategy_reads_llm_cache(tmp_project_dir)
        assert not passed
        assert "llm_cache" in msg

    def test_accepts_when_strategy_reads_cache(self, tmp_project_dir):
        _write(tmp_project_dir, "backtest/pilot/strategy_v1.py",
               "CACHE = 'data/processed/llm_cache/pilot/'\n")
        passed, msg = _check_strategy_reads_llm_cache(tmp_project_dir)
        assert passed, msg


class TestLLMCacheVariance:

    def test_rejects_when_cache_missing(self, tmp_project_dir):
        passed, msg = _check_llm_cache_variance(tmp_project_dir)
        assert not passed
        assert "llm_cache/pilot" in msg

    def test_rejects_degenerate_classifier(self, tmp_project_dir):
        # 20 entries, all "neutral" → dominant bucket = 100%
        for i in range(20):
            _write_json(tmp_project_dir,
                        f"data/processed/llm_cache/pilot/{i}.json",
                        {"classification": "neutral"})
        passed, msg = _check_llm_cache_variance(tmp_project_dir)
        assert not passed

    def test_rejects_fewer_than_three_distinct(self, tmp_project_dir):
        for i in range(10):
            lbl = "pos" if i % 2 == 0 else "neg"
            _write_json(tmp_project_dir,
                        f"data/processed/llm_cache/pilot/{i}.json",
                        {"classification": lbl})
        passed, msg = _check_llm_cache_variance(tmp_project_dir)
        assert not passed
        assert "distinct" in msg

    def test_passes_with_variance(self, tmp_project_dir):
        for i in range(30):
            lbl = ["pos", "neg", "neutral"][i % 3]
            _write_json(tmp_project_dir,
                        f"data/processed/llm_cache/pilot/{i}.json",
                        {"classification": lbl})
        passed, msg = _check_llm_cache_variance(tmp_project_dir)
        assert passed, msg


class TestCacheCoverage:

    def test_missing_universe_is_skip(self, tmp_project_dir):
        passed, msg, ratio = _check_cache_coverage(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed  # nothing to check

    def test_low_coverage_rejected(self, tmp_project_dir):
        _write_json(tmp_project_dir, "strategy/universe_v1.json",
                    ["AAPL", "MSFT", "GOOG", "AMZN", "META"])
        # Only 1 of 5 covered.
        _write_json(tmp_project_dir,
                    "data/processed/llm_cache/pilot/a.json",
                    {"ticker": "AAPL", "classification": "pos"})
        passed, msg, ratio = _check_cache_coverage(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert not passed
        assert ratio < 0.5

    def test_full_coverage_passes(self, tmp_project_dir):
        _write_json(tmp_project_dir, "strategy/universe_v1.json",
                    ["AAPL", "MSFT"])
        _write_json(tmp_project_dir,
                    "data/processed/llm_cache/pilot/a.json",
                    {"ticker": "AAPL", "classification": "pos"})
        _write_json(tmp_project_dir,
                    "data/processed/llm_cache/pilot/b.json",
                    {"ticker": "MSFT", "classification": "neg"})
        passed, msg, ratio = _check_cache_coverage(
            tmp_project_dir, {"hypothesis_count": 1}
        )
        assert passed
        assert ratio == 1.0


class TestDecideFromPilotHistory:

    def _state(self, hist, **kwargs):
        s = {
            "hypothesis_count": kwargs.get("hypothesis_count", 1),
            "iteration_within_hypothesis": kwargs.get("iter_within", len(hist)),
            "total_iterations": kwargs.get("total", len(hist)),
            "pilot_history": hist,
        }
        return s

    def test_empty_history_iterate(self):
        d = _decide_from_pilot_history(self._state([]))
        assert d.type == "iterate"
        assert d.next_step == "run_pilot_backtest"

    def test_converge(self):
        # Plateau = last two consecutive deltas < 0.15, AND sharpe >= 0.8, AND trades >= 30.
        hist = [
            {"hyp": 1, "iter": 1, "sharpe": 0.80, "trades": 32},
            {"hyp": 1, "iter": 2, "sharpe": 0.86, "trades": 35},
            {"hyp": 1, "iter": 3, "sharpe": 0.90, "trades": 40},
        ]
        d = _decide_from_pilot_history(self._state(hist))
        assert d.type == "converge"
        assert d.passed is True

    def test_dead_end_plateau_below_bar(self):
        hist = [
            {"hyp": 1, "iter": 1, "sharpe": 0.20, "trades": 35},
            {"hyp": 1, "iter": 2, "sharpe": 0.25, "trades": 36},
            {"hyp": 1, "iter": 3, "sharpe": 0.27, "trades": 38},
        ]
        d = _decide_from_pilot_history(self._state(hist))
        assert d.type == "dead_end"
        assert d.next_step == "revise_hypothesis"

    def test_needs_more_data(self):
        hist = [{"hyp": 1, "iter": 1, "sharpe": 1.6, "trades": 12}]
        d = _decide_from_pilot_history(self._state(hist))
        assert d.type == "needs_more_data"

    def test_insufficient_signal(self):
        hist = [
            {"hyp": 1, "iter": 1, "sharpe": 0.1, "trades": 3},
            {"hyp": 1, "iter": 2, "sharpe": 0.15, "trades": 5},
        ]
        d = _decide_from_pilot_history(self._state(hist))
        assert d.type == "insufficient_signal"
        assert d.next_step == "revise_hypothesis"

    def test_terminate_when_hypothesis_ceiling_hit(self):
        hist = [
            {"hyp": 3, "iter": 1, "sharpe": 0.2, "trades": 40},
        ]
        d = _decide_from_pilot_history(self._state(hist, hypothesis_count=3))
        assert d.type == "terminate"
        assert d.passed is True  # advances to verdict stage

    def test_terminate_when_total_iter_ceiling_exceeded(self):
        hist = [{"hyp": 2, "iter": 1, "sharpe": 0.3, "trades": 30}]
        d = _decide_from_pilot_history(
            self._state(hist, hypothesis_count=2, total=25)
        )
        assert d.type == "terminate"

    def test_iterate_default(self):
        # Big jump in Sharpe, no plateau yet.
        hist = [
            {"hyp": 1, "iter": 1, "sharpe": 0.1, "trades": 20},
            {"hyp": 1, "iter": 2, "sharpe": 0.45, "trades": 25},
        ]
        d = _decide_from_pilot_history(self._state(hist))
        assert d.type == "iterate"


# ---------------------------------------------------------------------------
# Trading stage 3 (full validation) and stage 4 (verdict) gates
# ---------------------------------------------------------------------------


def _good_full_metrics():
    return {
        "pilot_sharpe": 1.0,
        "oos_sharpe": 0.7,
        "pilot_sortino": 1.4,
        "oos_sortino": 1.0,
        "pilot_annualized_return_pct": 12.5,
        "oos_annualized_return_pct": 9.1,
        "walk_forward_win_rate": 0.7,
        "total_trades": 150,
        "t_stat_daily_returns": 2.5,
        "deflated_sharpe": 0.3,
        "turnover": 4.0,
        "declared_turnover": 4.0,
    }


class TestFullValidationMetrics:

    def test_missing_file(self, tmp_project_dir):
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "backtest/full/metrics.json" in msg

    def test_all_pass(self, tmp_project_dir):
        _write_json(tmp_project_dir, "backtest/full/metrics.json",
                    _good_full_metrics())
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert passed, msg

    def test_oos_collapse_rejected(self, tmp_project_dir):
        m = _good_full_metrics()
        m["oos_sortino"] = 0.2  # ratio 0.14 < 0.5 — sortino is the key metric
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "OOS collapse" in msg or "oos_sortino" in msg

    def test_low_walk_forward_rejected(self, tmp_project_dir):
        m = _good_full_metrics()
        m["walk_forward_win_rate"] = 0.4
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "walk_forward_win_rate" in msg

    def test_trades_below_minimum_rejected(self, tmp_project_dir):
        # Floor lowered 100 → 20 (user: don't force high-frequency behavior;
        # trade count is a bare statistical-significance floor, not a target).
        m = _good_full_metrics()
        m["total_trades"] = 12
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "total_trades" in msg

    def test_forty_trades_pass(self, tmp_project_dir):
        m = _good_full_metrics()
        m["total_trades"] = 40
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert passed, msg

    def test_sortino_is_the_consistency_metric(self, tmp_project_dir):
        # OOS/pilot consistency is keyed on SORTINO (the user's comparison
        # metric), not Sharpe: sortino collapse fails even with healthy sharpe.
        m = _good_full_metrics()
        m["oos_sortino"] = 0.3  # ratio 0.21 < 0.5
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "sortino" in msg.lower()

    def test_missing_sortino_rejected(self, tmp_project_dir):
        m = _good_full_metrics()
        del m["pilot_sortino"]
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "sortino" in msg.lower()

    def test_missing_annualized_return_rejected(self, tmp_project_dir):
        # Report-only but REQUIRED-present: the user asked for pilot + OOS
        # annualized returns and neither was being returned.
        m = _good_full_metrics()
        del m["oos_annualized_return_pct"]
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "annualized" in msg.lower()

    def test_negative_annualized_return_is_not_a_gate(self, tmp_project_dir):
        m = _good_full_metrics()
        m["oos_annualized_return_pct"] = -3.2  # reported, never gated
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert passed, msg

    def test_low_tstat_rejected(self, tmp_project_dir):
        m = _good_full_metrics()
        m["t_stat_daily_returns"] = 1.0
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "t_stat" in msg

    def test_negative_deflated_sharpe_rejected(self, tmp_project_dir):
        m = _good_full_metrics()
        m["deflated_sharpe"] = -0.1
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "deflated_sharpe" in msg

    def test_turnover_deviation_rejected(self, tmp_project_dir):
        m = _good_full_metrics()
        m["turnover"] = 9.0
        m["declared_turnover"] = 4.0
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        passed, msg = _check_full_validation_metrics(tmp_project_dir)
        assert not passed
        assert "turnover" in msg


class TestVerdictMatchesMetrics:

    def test_missing_verdict(self, tmp_project_dir):
        passed, msg = _check_verdict_matches_metrics(tmp_project_dir)
        assert not passed
        assert "pipeline/verdict.md" in msg

    def test_promote_matches_pass(self, tmp_project_dir):
        _write_json(tmp_project_dir, "backtest/full/metrics.json",
                    _good_full_metrics())
        _write(tmp_project_dir, "pipeline/verdict.md", (
            "## Final Recommendation\npromote\n\n## Rationale\nAll gates pass.\n"
        ))
        passed, msg = _check_verdict_matches_metrics(tmp_project_dir)
        assert passed, msg

    def test_reject_matches_fail(self, tmp_project_dir):
        m = _good_full_metrics()
        m["oos_sortino"] = 0.1
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        _write(tmp_project_dir, "pipeline/verdict.md", (
            "## Final Recommendation\nreject\n\n## Rationale\nOOS collapse.\n"
        ))
        passed, msg = _check_verdict_matches_metrics(tmp_project_dir)
        assert passed, msg

    def test_promote_when_metrics_fail_rejected(self, tmp_project_dir):
        m = _good_full_metrics()
        m["oos_sortino"] = 0.1
        _write_json(tmp_project_dir, "backtest/full/metrics.json", m)
        _write(tmp_project_dir, "pipeline/verdict.md", (
            "## Final Recommendation\npromote\n\n## Rationale\nLooks good.\n"
        ))
        passed, msg = _check_verdict_matches_metrics(tmp_project_dir)
        assert not passed
        assert "must be 'reject'" in msg or "mismatch" in msg.lower()

    def test_reject_when_metrics_pass_rejected(self, tmp_project_dir):
        _write_json(tmp_project_dir, "backtest/full/metrics.json",
                    _good_full_metrics())
        _write(tmp_project_dir, "pipeline/verdict.md", (
            "## Final Recommendation\nreject\n\n## Rationale\nI don't like it.\n"
        ))
        passed, msg = _check_verdict_matches_metrics(tmp_project_dir)
        assert not passed

    def test_missing_final_recommendation(self, tmp_project_dir):
        _write_json(tmp_project_dir, "backtest/full/metrics.json",
                    _good_full_metrics())
        _write(tmp_project_dir, "pipeline/verdict.md",
               "## Summary\nWe ran the strategy.\n")
        passed, msg = _check_verdict_matches_metrics(tmp_project_dir)
        assert not passed
        assert "Final Recommendation" in msg


class TestSortinoKeyMetricInLoop:
    """Stage-2 convergence keys on annualized Sortino (the user's comparison
    metric); legacy pilot_history rows without sortino fall back to Sharpe."""

    def _ls(self, rows):
        return {"pilot_history": rows, "hypothesis_count": 1,
                "total_iterations": len(rows)}

    def test_converges_on_sortino_plateau(self):
        rows = [
            {"hyp": 1, "sortino": 1.19, "sharpe": 0.5, "trades": 40},
            {"hyp": 1, "sortino": 1.20, "sharpe": 0.5, "trades": 40},
            {"hyp": 1, "sortino": 1.20, "sharpe": 0.5, "trades": 40},
        ]
        d = _decide_from_pilot_history(self._ls(rows))
        assert d.type == "converge", d.feedback  # sharpe 0.5 < 0.8 — sortino decides

    def test_below_sortino_bar_plateau_is_dead_end(self):
        rows = [
            {"hyp": 1, "sortino": 0.6, "sharpe": 1.2, "trades": 40},
            {"hyp": 1, "sortino": 0.6, "sharpe": 1.2, "trades": 40},
            {"hyp": 1, "sortino": 0.6, "sharpe": 1.2, "trades": 40},
        ]
        d = _decide_from_pilot_history(self._ls(rows))
        assert d.type == "dead_end", d.feedback  # good sharpe doesn't rescue it

    def test_legacy_rows_fall_back_to_sharpe(self):
        rows = [
            {"hyp": 1, "sharpe": 0.9, "trades": 40},
            {"hyp": 1, "sharpe": 0.9, "trades": 40},
            {"hyp": 1, "sharpe": 0.9, "trades": 40},
        ]
        d = _decide_from_pilot_history(self._ls(rows))
        assert d.type == "converge", d.feedback


class TestTradeCountGateIsAStatisticalFloor:
    """The trade-count gate is a bare statistical floor (20), NOT a frequency
    target. Fewer trades does not make a strategy worse — a small universe is
    a property of the asset class, not a defect.

    Anchored to a real case: prem14a-event-driven-v3 passed 5 of 6 gates with
    a t-stat of 4.52 and OOS Sharpe ABOVE its pilot Sharpe, and was rejected
    on 2026-04-18 purely for having 30 trades against a 100 gate. ff5d077
    lowered that gate to 20 on 2026-07-16.
    """

    PREM14A = {
        "pilot_sharpe": 0.8189, "oos_sharpe": 0.8783, "oos_pilot_ratio": 1.0726,
        "walk_forward_wins": 5, "walk_forward_total": 5, "walk_forward_win_rate": 1.0,
        "total_trades": 30, "t_stat_daily_returns": 4.52, "deflated_sharpe": 0.4,
        "turnover_annual": 24.33, "holding_period_days_claimed": 90,
        "holding_period_days_observed": 90, "benchmark": "SPY",
        # The April artifact predates ff5d077, which made annualized Sortino
        # the key metric and required annualized returns to be present. A
        # re-run must emit these; the stored metrics.json cannot simply be
        # re-judged. Values consistent with its Sharpe/trade profile.
        "pilot_sortino": 1.21, "oos_sortino": 1.30,
        "pilot_annualized_return_pct": 14.2, "oos_annualized_return_pct": 15.1,
    }

    def _run(self, tmp_path, metrics):
        import json as _json
        d = tmp_path / "backtest" / "full"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(_json.dumps(metrics))
        from sandbox_agent.pipeline.evaluator import _check_full_validation_metrics
        return _check_full_validation_metrics(str(tmp_path))

    def test_gate_value_is_twenty(self):
        from sandbox_agent.pipeline.evaluator import _FULL_VALIDATION_GATES
        assert _FULL_VALIDATION_GATES["min_trades"] == 20, (
            "trade-count gate changed; it is a statistical floor, not a "
            "frequency target — see prem14a")

    def test_prem14a_metrics_now_pass(self, tmp_path):
        passed, msg = self._run(tmp_path, dict(self.PREM14A))
        assert passed, f"prem14a would still be rejected: {msg}"

    def test_thirty_trades_is_not_a_failure_reason(self, tmp_path):
        passed, msg = self._run(tmp_path, dict(self.PREM14A))
        assert "total_trades" not in msg, f"trade count cited as a failure: {msg}"

    def test_below_the_statistical_floor_still_fails(self, tmp_path):
        """The floor is real — 20 exists so a 3-trade fluke cannot pass."""
        m = dict(self.PREM14A, total_trades=19)
        passed, msg = self._run(tmp_path, m)
        assert not passed and "total_trades" in msg
