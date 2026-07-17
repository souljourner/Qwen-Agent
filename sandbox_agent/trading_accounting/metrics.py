"""Metrics computed from a vetted Ledger — sanity-clean by construction.

Emits exactly the keys the pipeline gates read (total_return/return_pct,
max_drawdown/dd_pct, sharpe, total_trades, t_stat_daily_returns,
deflated_sharpe) plus the equity_curve itself, so metrics_sanity can verify
declared values against the curve (the strong consistency path).

Conventions: total_return and max_drawdown are FRACTIONS (-0.35 = -35%);
*_pct twins are percentages. max_drawdown is peak-to-trough on the equity
curve and may legitimately be < -1.0 when equity went negative (short
blowup); the curve is included so the sanity gate can confirm that.

Pure stdlib (math/statistics) — no numpy/pandas required, so it imports in
any project venv.
"""

import math
from statistics import mean, pstdev, stdev
from typing import List, Optional, Sequence

from sandbox_agent.trading_accounting.ledger import Ledger

TRADING_DAYS_PER_YEAR = 252


def _period_returns(equities: Sequence[float]) -> List[float]:
    """Simple per-mark returns; stops at the first non-positive equity point
    (returns are undefined across a blowup — and trading halts there anyway)."""
    out = []
    for prev, cur in zip(equities, equities[1:]):
        if prev <= 0:
            break
        out.append((cur - prev) / prev)
    return out


def _max_drawdown(equities: Sequence[float]) -> float:
    peak = equities[0]
    worst = 0.0
    for e in equities:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (e - peak) / peak
            if dd < worst:
                worst = dd
    return worst


def _probabilistic_sharpe(sr_period: float, returns: Sequence[float]) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado) vs SR*=0, using
    per-period SR and higher moments. Returns Φ(z) in [0, 1]."""
    n = len(returns)
    if n < 3 or sr_period == 0:
        return 0.5
    mu = mean(returns)
    sd = pstdev(returns)
    if sd == 0:
        return 0.5
    skew = mean([((r - mu) / sd) ** 3 for r in returns])
    kurt = mean([((r - mu) / sd) ** 4 for r in returns])
    denom = math.sqrt(max(1e-12, 1 - skew * sr_period + ((kurt - 1) / 4.0) * sr_period ** 2))
    z = (sr_period * math.sqrt(n - 1)) / denom
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def compute_metrics(ledger: Ledger, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Compute the gate-facing metrics dict from a Ledger."""
    curve = ledger.equity_curve()
    if not curve:
        raise ValueError("ledger has no equity marks — call mark() once per bar")
    equities = [e for _, e in curve]
    first, last = equities[0], equities[-1]

    total_return = (last - first) / first if first > 0 else 0.0
    max_dd = _max_drawdown(equities)
    returns = _period_returns(equities)

    if len(returns) >= 2:
        sd = stdev(returns)
        mu = mean(returns)
        sharpe = (mu / sd) * math.sqrt(periods_per_year) if sd > 0 else 0.0
        t_stat = (mu / (sd / math.sqrt(len(returns)))) if sd > 0 else 0.0
        sr_period = (mu / sd) if sd > 0 else 0.0
    else:
        sharpe = t_stat = sr_period = 0.0

    # Deflated Sharpe reported as PSR - 0.5: positive iff the Sharpe is
    # statistically distinguishable from zero (given n, skew, kurtosis).
    deflated = _probabilistic_sharpe(sr_period, returns) - 0.5

    # Annualized Sortino (mean / downside deviation, MAR=0) — the KEY metric
    # for comparing strategies. Capped at 99 (JSON-safe) when there are no
    # losing periods.
    if len(returns) >= 2:
        dd_dev = math.sqrt(sum(min(r, 0.0) ** 2 for r in returns) / len(returns))
        if dd_dev > 0:
            sortino = min((mean(returns) / dd_dev) * math.sqrt(periods_per_year), 99.0)
        else:
            sortino = 99.0 if mean(returns) > 0 else 0.0
    else:
        sortino = 0.0

    # Annualized return (CAGR). None when equity crossed zero — a compound
    # rate is undefined across a sign change (short-strategy blowups).
    years = max(len(returns), 1) / periods_per_year
    if first > 0 and last > 0 and years > 0:
        annualized = (last / first) ** (1.0 / years) - 1.0
    else:
        annualized = None

    return {
        "total_return": total_return,
        "return_pct": total_return * 100.0,
        "annualized_return": annualized,
        "annualized_return_pct": annualized * 100.0 if annualized is not None else None,
        "max_drawdown": max_dd,
        "dd_pct": max_dd * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "t_stat_daily_returns": t_stat,
        "deflated_sharpe": deflated,
        "total_trades": len(ledger.trades),
        "blown_up": bool(ledger.blown_up),
        "equity_curve": [[ts, e] for ts, e in curve],
        "trades_log": ledger.trades,
        "accounting": "sandbox_agent.trading_accounting",
    }


def walk_forward_win_rate(strategy_returns: Sequence[float],
                          benchmark_returns: Sequence[float],
                          n_windows: int = 8) -> Optional[float]:
    """Fraction of contiguous windows where the strategy's cumulative return
    beats the benchmark's. Returns None when there's too little data."""
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < n_windows * 2:
        return None
    size = n // n_windows
    wins = 0
    for w in range(n_windows):
        s = slice(w * size, (w + 1) * size)
        strat = math.prod(1 + r for r in strategy_returns[s]) - 1
        bench = math.prod(1 + r for r in benchmark_returns[s]) - 1
        if strat > bench:
            wins += 1
    return wins / n_windows
