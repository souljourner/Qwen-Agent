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
