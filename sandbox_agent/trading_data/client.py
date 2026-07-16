"""HTTP client for the auram_data market-data API (QuestDB + FastAPI).

Base URL from AURAM_DATA_URL (default http://host.docker.internal:18000 — the
service runs on the host under launchd; the agent container reaches it with no
credentials). Every function returns a pandas DataFrame or raises
DataUnavailable — NEVER a silently-empty result, so strategies can't quietly
backtest on missing data.

Prices are split/dividend-ADJUSTED by default (adjusted=True): OHLC are scaled
by the adjusted_close/close ratio per bar. Raw prices misstate returns across
splits — only pass adjusted=False when you specifically need exchange prints.

Deps: requests + pandas (install in the project venv: `uv pip install pandas requests`).
"""

import os
import time
from typing import Optional

import pandas as pd
import requests

DEFAULT_BASE_URL = "http://host.docker.internal:18000"
_TIMEOUT = 30


class DataUnavailable(RuntimeError):
    """No data for the request (unknown symbol, empty range, or service down)."""


def _base_url() -> str:
    return os.environ.get("AURAM_DATA_URL", DEFAULT_BASE_URL).rstrip("/")


def _get_json(path: str, params: Optional[dict] = None) -> dict:
    url = f"{_base_url()}{path}"
    try:
        resp = requests.get(url, params=params or {}, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001 — collapse transport errors into one type
        raise DataUnavailable(f"auram_data request failed: {url} — {e}") from e


def _bars_frame(payload: dict, symbol: str, adjusted: bool) -> pd.DataFrame:
    rows = payload.get("data") or []
    if not rows:
        raise DataUnavailable(
            f"no bars for {symbol!r} — symbol not in the auram universe or empty range. "
            f"Check get_universe() for coverage.")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")
    if adjusted and "adjusted_close" in df.columns:
        ratio = (df["adjusted_close"] / df["close"]).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * ratio
    return df.drop(columns=[c for c in ("symbol",) if c in df.columns])


def get_daily(symbol: str, start: Optional[str] = None, end: Optional[str] = None,
              adjusted: bool = True, limit: int = 20000) -> pd.DataFrame:
    """Daily OHLCV for one symbol, ascending DatetimeIndex. Dates YYYY-MM-DD."""
    params = {"limit": limit}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    payload = _get_json(f"/api/daily/{symbol.upper()}", params)
    return _bars_frame(payload, symbol, adjusted)


def get_intraday_5m(symbol: str, start: Optional[str] = None, end: Optional[str] = None,
                    limit: int = 10000) -> pd.DataFrame:
    """5-minute bars (raw — no adjusted_close at intraday granularity)."""
    params = {"limit": limit}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    payload = _get_json(f"/api/intraday/{symbol.upper()}", params)
    return _bars_frame(payload, symbol, adjusted=False)


def get_universe(limit: int = 10000) -> pd.DataFrame:
    """All tickers in the local store (symbol, name, exchange, sector,
    industry, market_cap) — deduped and sorted by symbol."""
    payload = _get_json("/api/symbols", {"limit": limit})
    rows = payload.get("data") or []
    if not rows:
        raise DataUnavailable("auram_data /api/symbols returned no tickers")
    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).sort_values("symbol")
    return df.reset_index(drop=True)


def get_dividends(symbol: str, limit: int = 200) -> pd.DataFrame:
    payload = _get_json(f"/api/dividends/{symbol.upper()}", {"limit": limit})
    rows = payload.get("data") or []
    if not rows:
        raise DataUnavailable(f"no dividends recorded for {symbol!r}")
    return pd.DataFrame(rows)


def get_splits(symbol: str, limit: int = 200) -> pd.DataFrame:
    payload = _get_json(f"/api/splits/{symbol.upper()}", {"limit": limit})
    rows = payload.get("data") or []
    if not rows:
        raise DataUnavailable(f"no splits recorded for {symbol!r}")
    return pd.DataFrame(rows)


def health() -> bool:
    try:
        return _get_json("/health").get("status") == "ok"
    except DataUnavailable:
        return False


_BACKFILL_TIMEOUT = 300  # sync path fetches a symbol's full daily history


def request_backfill(symbol: str, timeout: int = _BACKFILL_TIMEOUT,
                     wait_visible: float = 20.0) -> dict:
    """Ask the auram_data service to backfill a symbol from EODHD on demand.

    Synchronously ingests daily OHLCV + dividends/splits; intraday,
    fundamentals, and news follow in the background. The symbol permanently
    joins the daily-refresh universe. Blocks (up to `wait_visible` seconds)
    until the new daily bars are actually queryable — QuestDB commits large
    out-of-order merges asynchronously — so `get_daily` works immediately
    after this returns. Returns the service summary, e.g.
    {"symbol", "status": "backfilled"|"extended"|"exists", "counts", "background"}.
    Raises DataUnavailable if EODHD has no such symbol or the service is down.
    """
    sym = symbol.strip().upper()
    url = f"{_base_url()}/api/ingest/{sym}"
    try:
        resp = requests.post(url, timeout=timeout)
        resp.raise_for_status()
        summary = resp.json()
    except Exception as e:  # noqa: BLE001 — collapse transport errors into one type
        raise DataUnavailable(f"backfill failed: {url} — {e}") from e

    if summary.get("status") in ("backfilled", "extended") and wait_visible > 0:
        deadline = time.monotonic() + wait_visible
        while True:
            try:
                if _get_json(f"/api/daily/{sym}", {"limit": 1}).get("data"):
                    break
            except DataUnavailable:
                pass
            if time.monotonic() >= deadline:
                break
            time.sleep(1.0)
    return summary
