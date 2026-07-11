"""Tests for metrics_sanity — consistency-based validation (long AND short
strategies), canonical hashing, evaluator-owned stamping."""

import json
import math

import pytest

from sandbox_agent.pipeline.metrics_sanity import (
    canonical_hash,
    stamp_metrics_file,
    validate_metrics,
)


# --- universal rules -----------------------------------------------------------

def test_nan_inf_rejected():
    assert validate_metrics({"total_return": float("nan")})
    assert validate_metrics({"oos_sharpe": float("inf")})
    assert validate_metrics({"nested": {"dd_pct": float("-inf")}})


def test_clean_small_metrics_pass():
    assert validate_metrics({"return_pct": 12.5, "dd_pct": -8.2, "trades": 42}) == []


# --- unverifiable claims (no equity curve): crude ceilings ----------------------

def test_absurd_return_rejected_without_curve():
    errs = validate_metrics({"return_pct": 1_399_591.9, "dd_pct": -235.3})
    assert any("return" in e.lower() for e in errs)


def test_positive_drawdown_rejected():
    errs = validate_metrics({"return_pct": 10.0, "dd_pct": 5.0})
    assert any("drawdown" in e.lower() for e in errs)


def test_short_strategy_moderate_sub100_dd_allowed_without_curve():
    # Shorts can legitimately exceed -100% DD (equity goes negative). Without
    # a curve we can't verify, so only ABSURD magnitudes are rejected.
    assert validate_metrics({"return_pct": 80.0, "dd_pct": -150.0}) == []


def test_absurd_dd_magnitude_rejected_without_curve():
    errs = validate_metrics({"return_pct": 10.0, "dd_pct": -5000.0})
    assert any("drawdown" in e.lower() for e in errs)


def test_long_only_dd_beyond_100_rejected():
    errs = validate_metrics({"return_pct": 10.0, "dd_pct": -150.0},
                            strategy_type="long_only")
    assert any("long" in e.lower() for e in errs)


# --- verifiable path (equity curve present): consistency beats ceilings ---------

def _curve(values):
    return [[f"2026-01-{i+1:02d}", v] for i, v in enumerate(values)]


def test_curve_consistent_metrics_pass():
    # 100 -> 150: +50% return, max DD -20% (150->120 is later... build simply)
    eq = _curve([100, 120, 96, 150])
    m = {"equity_curve": eq, "total_return": 0.50, "max_drawdown": -0.20}
    assert validate_metrics(m) == []


def test_declared_return_contradicts_curve_rejected():
    eq = _curve([100, 120, 96, 150])
    m = {"equity_curve": eq, "total_return": 5.00, "max_drawdown": -0.20}
    errs = validate_metrics(m)
    assert any("match" in e.lower() or "curve" in e.lower() for e in errs)


def test_short_blowup_curve_passes_with_blown_up_flag():
    # Equity goes negative (short squeeze): DD < -100% is REAL here.
    eq = _curve([100, 130, -30, 10])
    dd = (-30 - 130) / 130          # -1.2308
    ret = (10 - 100) / 100
    m = {"equity_curve": eq, "total_return": ret, "max_drawdown": dd}
    errs = validate_metrics(m)
    assert errs == []
    assert m.get("blown_up") is True    # stamped in place


def test_sub100_dd_with_all_positive_curve_rejected():
    eq = _curve([100, 120, 110, 130])   # never negative
    m = {"equity_curve": eq, "total_return": 0.30, "max_drawdown": -1.5}
    errs = validate_metrics(m)
    assert any("contradict" in e.lower() or "curve" in e.lower() for e in errs)


def test_trades_after_blowup_rejected():
    eq = _curve([100, -20, 30, 60])
    m = {
        "equity_curve": eq,
        "total_return": -0.40,
        "max_drawdown": (-20 - 100) / 100,
        "trades_log": [
            {"ts": "2026-01-01", "symbol": "SOXS", "qty": -10},
            {"ts": "2026-01-03", "symbol": "SOXS", "qty": 5},   # AFTER blowup on 01-02
        ],
    }
    errs = validate_metrics(m)
    assert any("blowup" in e.lower() or "liquidat" in e.lower() for e in errs)


def test_huge_but_curve_consistent_return_passes():
    # Realism is stage-3's job; sanity only checks internal consistency.
    eq = _curve([100, 5000, 4000, 20000])
    m = {"equity_curve": eq, "total_return": 199.0, "max_drawdown": -0.20}
    assert validate_metrics(m) == []


def test_non_monotonic_timestamps_rejected():
    eq = [["2026-01-02", 100], ["2026-01-01", 110]]
    m = {"equity_curve": eq, "total_return": 0.10, "max_drawdown": 0.0}
    errs = validate_metrics(m)
    assert any("monotonic" in e.lower() or "timestamp" in e.lower() for e in errs)


# --- hashing + stamping ----------------------------------------------------------

def test_canonical_hash_stable_and_ignores_content_hash():
    m1 = {"a": 1, "b": [1, 2]}
    m2 = {"b": [1, 2], "a": 1, "content_hash": "whatever"}
    assert canonical_hash(m1) == canonical_hash(m2)
    assert canonical_hash({"a": 2}) != canonical_hash(m1)


def test_stamp_metrics_file(tmp_path):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps({"oos_sharpe": 1.2}))
    stamped = stamp_metrics_file(str(p), strategy_version="v10")
    on_disk = json.loads(p.read_text())
    assert on_disk["schema_version"] == 1
    assert on_disk["strategy_version"] == "v10"
    assert on_disk["content_hash"] == canonical_hash(on_disk)
    assert stamped["content_hash"] == on_disk["content_hash"]
