"""Tests for the auram_data HTTP client (mocked requests)."""

import pytest

import sandbox_agent.trading_data.client as tdc
from sandbox_agent.trading_data.client import DataUnavailable, get_daily, get_universe


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _bars(rows):
    return {"data": rows, "count": len(rows)}


BARS = [
    # API returns ts DESC
    {"ts": "2026-01-03 07:00:00", "symbol": "AAPL", "open": 102, "high": 104,
     "low": 101, "close": 103, "volume": 1000, "adjusted_close": 51.5},
    {"ts": "2026-01-02 07:00:00", "symbol": "AAPL", "open": 100, "high": 102,
     "low": 99, "close": 101, "volume": 900, "adjusted_close": 50.5},
]


def test_get_daily_ascending_and_adjusted(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        return _Resp(_bars(BARS))

    monkeypatch.setattr(tdc.requests, "get", fake_get)
    df = get_daily("AAPL", start="2026-01-01")
    assert list(df.index) == sorted(df.index)          # ascending
    # adjusted=True: OHLC scaled by adjusted_close/close ratio (0.5 here)
    assert df.iloc[0]["close"] == pytest.approx(50.5)
    assert df.iloc[0]["open"] == pytest.approx(50.0)
    assert df.iloc[-1]["close"] == pytest.approx(51.5)
    assert "volume" in df.columns
    assert "/api/daily/AAPL" in calls[0][0]


def test_get_daily_raw_when_adjusted_false(monkeypatch):
    monkeypatch.setattr(tdc.requests, "get", lambda *a, **k: _Resp(_bars(BARS)))
    df = get_daily("AAPL", adjusted=False)
    assert df.iloc[0]["close"] == pytest.approx(101)


def test_empty_result_raises_data_unavailable(monkeypatch):
    monkeypatch.setattr(tdc.requests, "get", lambda *a, **k: _Resp(_bars([])))
    with pytest.raises(DataUnavailable):
        get_daily("NOPE")


def test_connection_error_raises_data_unavailable(monkeypatch):
    def boom(*a, **k):
        raise tdc.requests.ConnectionError("down")
    monkeypatch.setattr(tdc.requests, "get", boom)
    with pytest.raises(DataUnavailable):
        get_daily("AAPL")


def test_get_universe_dedupes(monkeypatch):
    payload = {"data": [
        {"symbol": "A", "name": "Agilent", "exchange": "NYSE", "sector": "Healthcare",
         "industry": "Diag", "market_cap": 1.0, "currency_code": "USD", "country_code": "US"},
        {"symbol": "A", "name": "Agilent", "exchange": "NYSE", "sector": "Healthcare",
         "industry": "Diag", "market_cap": 1.0, "currency_code": "USD", "country_code": "US"},
        {"symbol": "AAPL", "name": "Apple", "exchange": "NASDAQ", "sector": "Tech",
         "industry": "Hardware", "market_cap": 2.0, "currency_code": "USD", "country_code": "US"},
    ], "count": 3}
    monkeypatch.setattr(tdc.requests, "get", lambda *a, **k: _Resp(payload))
    df = get_universe()
    assert list(df["symbol"]) == ["A", "AAPL"]         # deduped, sorted


def test_missing_adjusted_close_falls_back_to_raw(monkeypatch):
    rows = [dict(BARS[1], adjusted_close=None)]
    monkeypatch.setattr(tdc.requests, "get", lambda *a, **k: _Resp(_bars(rows)))
    df = get_daily("AAPL")                              # adjusted=True default
    assert df.iloc[0]["close"] == pytest.approx(101)    # graceful fallback


# --- request_backfill --------------------------------------------------------

def test_request_backfill_posts_and_returns_summary(monkeypatch):
    from sandbox_agent.trading_data.client import request_backfill
    captured = {}
    summary = {"symbol": "NVDA", "status": "backfilled",
               "counts": {"daily": 100, "dividends": 4, "splits": 1},
               "background": ["intraday", "fundamentals", "news"]}

    def fake_post(url, timeout=None, **kw):
        captured["url"] = url
        captured["timeout"] = timeout
        return _Resp(summary)

    monkeypatch.setattr(tdc.requests, "post", fake_post)
    monkeypatch.setattr(tdc.requests, "get", lambda url, **kw: _Resp(_bars(BARS)))
    out = request_backfill("nvda")
    assert out == summary
    assert captured["url"].endswith("/api/ingest/NVDA")  # uppercased
    assert captured["timeout"] == 300  # generous: sync path fetches full history


def test_request_backfill_waits_until_bars_visible(monkeypatch):
    # QuestDB commits big out-of-order merges asynchronously: the POST can
    # succeed before GET /api/daily sees rows. request_backfill must poll
    # until the data is queryable so callers can get_daily immediately.
    from sandbox_agent.trading_data.client import request_backfill
    summary = {"symbol": "NVDA", "status": "backfilled",
               "counts": {"daily": 100}, "background": []}
    polls = []

    def fake_get(url, params=None, timeout=None, **kw):
        polls.append(url)
        return _Resp(_bars([] if len(polls) < 3 else BARS))

    monkeypatch.setattr(tdc.requests, "post", lambda url, **kw: _Resp(summary))
    monkeypatch.setattr(tdc.requests, "get", fake_get)
    monkeypatch.setattr(tdc.time, "sleep", lambda s: None)
    out = request_backfill("NVDA")
    assert out == summary
    assert len(polls) == 3  # kept polling until bars appeared
    assert all("/api/daily/NVDA" in u for u in polls)


def test_request_backfill_exists_skips_visibility_poll(monkeypatch):
    from sandbox_agent.trading_data.client import request_backfill

    def no_get(url, **kw):
        raise AssertionError("no poll expected for status=exists")

    monkeypatch.setattr(tdc.requests, "post",
                        lambda url, **kw: _Resp({"symbol": "AAPL", "status": "exists",
                                                 "counts": {}, "background": []}))
    monkeypatch.setattr(tdc.requests, "get", no_get)
    assert request_backfill("AAPL")["status"] == "exists"


def test_request_backfill_failure_raises_data_unavailable(monkeypatch):
    from sandbox_agent.trading_data.client import request_backfill

    def fake_post(url, timeout=None, **kw):
        raise ConnectionError("service down")

    monkeypatch.setattr(tdc.requests, "post", fake_post)
    with pytest.raises(DataUnavailable):
        request_backfill("NVDA")


def test_request_backfill_404_raises_data_unavailable(monkeypatch):
    from sandbox_agent.trading_data.client import request_backfill
    monkeypatch.setattr(tdc.requests, "post",
                        lambda url, timeout=None, **kw: _Resp({"detail": "no data"}, status=404))
    with pytest.raises(DataUnavailable):
        request_backfill("ZZZZZZ")


def test_request_backfill_exported():
    import sandbox_agent.trading_data as td
    assert "request_backfill" in td.__all__
    assert callable(td.request_backfill)
