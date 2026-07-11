"""Vetted trading accounting — the ledger + metrics module strategies MUST use.

Strategies written by the research-loop agent import this instead of
hand-rolling equity/return/drawdown math (hand-rolled accounting produced
impossible numbers like -235% drawdowns from post-blowup trading). The
evaluator's lineage gate checks that backtest scripts import
`sandbox_agent.trading_accounting`.

Usage inside a backtest script (PYTHONPATH=/app is preset by exec):

    from sandbox_agent.trading_accounting import Ledger, compute_metrics
    led = Ledger(cash=100_000)
    led.fill(ts, symbol, qty, price)          # qty < 0 = short
    led.mark(ts, {symbol: close, ...})        # once per bar
    metrics = compute_metrics(led)            # sanity-clean by construction
"""

from sandbox_agent.trading_accounting.ledger import Ledger, LedgerBlownUp
from sandbox_agent.trading_accounting.metrics import compute_metrics, walk_forward_win_rate

__all__ = ["Ledger", "LedgerBlownUp", "compute_metrics", "walk_forward_win_rate"]
