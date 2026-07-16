# Trading Data — the local EODHD market-data store
> How to fetch US equity/ETF market data: the local auram/QuestDB store is the REQUIRED source (deterministic, split/dividend-adjusted); yfinance is banned for US equities/ETFs.

## What it is
A local QuestDB store fed by EODHD (~3,000 US tickers + leveraged ETFs), refreshed automatically (intraday every 2h during market hours, daily after close). It serves daily OHLCV, 5-minute intraday bars, dividends/splits, fundamentals, earnings, and news through an HTTP API on the host. Backtests and any price analysis MUST use it so results are reproducible — yfinance data changes between runs.

## Client API (inside code_interpreter or exec scripts)
```python
from sandbox_agent.trading_data import (
    get_daily, get_intraday_5m, get_universe, get_dividends, get_splits,
    health, request_backfill, DataUnavailable)

df = get_daily("AAPL", start="2020-01-01")   # ascending DatetimeIndex, ADJUSTED OHLC
raw = get_daily("AAPL", adjusted=False)       # exchange prints (rarely what you want)
bars5 = get_intraday_5m("AAPL", start="2026-06-01")
universe = get_universe()                     # symbol/name/exchange/sector/market_cap
```
- Prices are split/dividend-**adjusted** by default — raw prices misstate returns across splits.
- Every function returns a DataFrame or raises `DataUnavailable` (unknown symbol, empty range, service down) — never a silently-empty result. Check `health()` first if unsure the service is up.

## Missing symbol? Backfill it on demand
```python
try:
    df = get_daily("CRWD")
except DataUnavailable:
    request_backfill("CRWD")   # pulls from EODHD; blocks until bars are queryable
    df = get_daily("CRWD")
```
- `request_backfill` ingests daily + dividends/splits synchronously (seconds) and intraday/fundamentals/news in the background; the symbol permanently joins the daily-refresh universe.
- Call it at most ONCE per symbol; if data is still unavailable, the symbol doesn't exist at EODHD — skip it.
- It spends EODHD API credits (~20+/symbol): cap yourself at **25** new symbols per task/pipeline run; record what you skipped.
- Existing-but-stale symbols get topped up automatically (`status: "extended"`).

## Policy
- **yfinance is BANNED for US equities/ETFs.** Use it only for what the store lacks: futures, COT, FX, and index tickers like ^VIX.
- Inside trading pipelines the stage instructions repeat these rules — they are the same store and the same policy.
- The `stock_price` chat tool is fine for a single "what's AAPL at?" quote; anything analytical (returns, backtests, screens) goes through this client.
