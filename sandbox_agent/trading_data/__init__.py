"""Deterministic market-data layer for trading strategies (auram_data-backed).

Strategies MUST source US equity/ETF prices from here instead of ad-hoc
yfinance calls — the data comes from the local auram_data QuestDB store
(EODHD-fed, launchd-refreshed), so backtests are reproducible and identical
across pipeline iterations.
"""

from sandbox_agent.trading_data.client import (
    DataUnavailable,
    get_daily,
    get_dividends,
    get_intraday_5m,
    get_splits,
    get_universe,
    health,
)

__all__ = ["DataUnavailable", "get_daily", "get_dividends", "get_intraday_5m",
           "get_splits", "get_universe", "health"]
