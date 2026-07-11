"""Known-answer tests for the vetted trading ledger + metrics.

This module is what strategies MUST use for accounting (evaluator lineage
gate) — its guarantees: correct long/short margin math, halt-at-blowup (no
post-liquidation trading), and metrics that pass metrics_sanity by
construction."""

import math

import pytest

from sandbox_agent.trading_accounting import Ledger, LedgerBlownUp, compute_metrics
from sandbox_agent.pipeline.metrics_sanity import validate_metrics


def test_long_roundtrip_exact_numbers():
    led = Ledger(cash=10_000)
    led.fill("2026-01-02", "XYZ", qty=100, price=50.0)          # cost 5000
    led.mark("2026-01-02", {"XYZ": 50.0})                        # equity 10000
    led.mark("2026-01-03", {"XYZ": 55.0})                        # +500 → 10500
    led.fill("2026-01-04", "XYZ", qty=-100, price=55.0)          # sell all
    led.mark("2026-01-04", {"XYZ": 55.0})                        # 10500 cash
    curve = led.equity_curve()
    assert [e for _, e in curve] == [10_000, 10_500, 10_500]
    assert led.cash == 10_500
    assert led.positions.get("XYZ", 0) == 0


def test_commission_reduces_equity():
    led = Ledger(cash=1_000)
    led.fill("2026-01-02", "XYZ", qty=10, price=50.0, commission=5.0)
    led.mark("2026-01-02", {"XYZ": 50.0})
    assert led.equity_curve()[-1][1] == pytest.approx(995.0)


def test_short_position_profits_when_price_falls():
    led = Ledger(cash=10_000)
    led.fill("2026-01-02", "SOXS", qty=-100, price=40.0)         # short: +4000 cash
    led.mark("2026-01-02", {"SOXS": 40.0})                       # 14000 - 4000 = 10000
    led.mark("2026-01-03", {"SOXS": 30.0})                       # 14000 - 3000 = 11000
    assert led.equity_curve()[-1][1] == pytest.approx(11_000)


def test_short_squeeze_blowup_halts_trading():
    led = Ledger(cash=10_000)
    led.fill("2026-01-02", "GME", qty=-1000, price=20.0)         # +20000 cash → 30000
    led.mark("2026-01-02", {"GME": 20.0})                        # 30000-20000 = 10000
    led.mark("2026-01-03", {"GME": 35.0})                        # 30000-35000 = -5000 → BLOWN UP
    assert led.blown_up
    curve = led.equity_curve()
    assert curve[-1][1] < 0
    with pytest.raises(LedgerBlownUp):
        led.fill("2026-01-04", "GME", qty=1000, price=35.0)      # no post-blowup trades
    # marks after blowup are allowed (observing the wreckage), no more fills.


def test_metrics_known_answer_and_sanity_pass():
    led = Ledger(cash=10_000)
    led.fill("2026-01-02", "XYZ", qty=100, price=50.0)
    for ts, px in [("2026-01-02", 50.0), ("2026-01-03", 55.0), ("2026-01-04", 44.0),
                   ("2026-01-05", 60.0)]:
        led.mark(ts, {"XYZ": px})
    m = compute_metrics(led)
    # equity: 10000, 10500, 9400, 11000
    assert m["total_return"] == pytest.approx(0.10)
    assert m["return_pct"] == pytest.approx(10.0)
    # peak 10500 → trough 9400 = -10.476%
    assert m["max_drawdown"] == pytest.approx((9400 - 10500) / 10500, rel=1e-6)
    assert m["total_trades"] == 1
    assert m["blown_up"] is False
    assert "equity_curve" in m and len(m["equity_curve"]) == 4
    # By construction, the vetted metrics pass the sanity gates.
    assert validate_metrics(m) == []


def test_blown_up_metrics_report_sub100_dd_and_pass_sanity():
    led = Ledger(cash=10_000)
    led.fill("2026-01-02", "GME", qty=-1000, price=20.0)
    led.mark("2026-01-02", {"GME": 20.0})
    led.mark("2026-01-03", {"GME": 35.0})       # equity -5000
    m = compute_metrics(led)
    assert m["blown_up"] is True
    assert m["max_drawdown"] < -1.0             # beyond -100%, legitimately
    assert validate_metrics(m) == []            # consistency: curve shows <= 0


def test_sharpe_and_tstat_shapes():
    led = Ledger(cash=10_000)
    led.fill("2026-01-02", "XYZ", qty=100, price=50.0)
    prices = [50, 51, 52, 51.5, 53, 54, 53.5, 55, 56, 57]
    for i, px in enumerate(prices):
        led.mark(f"2026-01-{i+2:02d}", {"XYZ": float(px)})
    m = compute_metrics(led)
    assert m["sharpe"] > 0
    assert m["t_stat_daily_returns"] > 0
    assert -0.5 <= m["deflated_sharpe"] <= 0.5
    assert math.isfinite(m["sharpe"])


def test_flat_curve_zero_sharpe_not_nan():
    led = Ledger(cash=5_000)
    for i in range(5):
        led.mark(f"2026-01-{i+2:02d}", {})
    m = compute_metrics(led)
    assert m["sharpe"] == 0.0
    assert m["total_return"] == 0.0
    assert validate_metrics(m) == []
