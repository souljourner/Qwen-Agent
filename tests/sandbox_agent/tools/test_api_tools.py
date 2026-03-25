"""Tests for api_tools — SSE parsing and tool registration."""

import json
from unittest.mock import MagicMock, patch

import pytest

from sandbox_agent.tools.api_tools import _call_tool_api


class FakeResponse:
    """Mock requests.Response with SSE-style iter_lines."""

    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line


class TestCallToolApi:

    @patch("sandbox_agent.tools.api_tools.requests.post")
    def test_parses_sse_result(self, mock_post):
        mock_post.return_value = FakeResponse([
            "event: progress",
            'data: {"message": "Searching..."}',
            "",
            "event: result",
            'data: {"results": [{"title": "Hello", "snippet": "World"}]}',
            "",
        ])
        result = _call_tool_api("web_search", {"query": "test"})
        parsed = json.loads(result)
        assert parsed[0]["title"] == "Hello"

    @patch("sandbox_agent.tools.api_tools.requests.post")
    def test_handles_error_result(self, mock_post):
        mock_post.return_value = FakeResponse([
            "event: result",
            'data: {"error": "Search failed"}',
            "",
        ])
        result = _call_tool_api("web_search", {"query": "test"})
        assert "Error: Search failed" in result

    @patch("sandbox_agent.tools.api_tools.requests.post")
    def test_handles_no_trailing_blank_line(self, mock_post):
        """SSE stream that ends without a trailing blank line."""
        mock_post.return_value = FakeResponse([
            "event: result",
            'data: {"results": {"price": 150.0}}',
        ])
        result = _call_tool_api("stock_price", {"symbol": "AAPL"})
        parsed = json.loads(result)
        assert parsed["price"] == 150.0

    @patch("sandbox_agent.tools.api_tools.requests.post")
    def test_handles_empty_stream(self, mock_post):
        mock_post.return_value = FakeResponse([])
        result = _call_tool_api("web_search", {"query": "test"})
        assert "no result" in result

    @patch("sandbox_agent.tools.api_tools.requests.post")
    def test_stock_price_structured_result(self, mock_post):
        stock_data = {"results": {"symbol": "AAPL", "price": 251.52, "change": 0.03}}
        mock_post.return_value = FakeResponse([
            "event: progress",
            'data: {"message": "Fetching stock data for AAPL..."}',
            "",
            "event: result",
            f"data: {json.dumps(stock_data)}",
            "",
        ])
        result = _call_tool_api("stock_price", {"symbol": "AAPL"})
        parsed = json.loads(result)
        assert parsed["symbol"] == "AAPL"
        assert parsed["price"] == 251.52


class TestToolRegistration:

    def test_web_search_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "web_search" in TOOL_REGISTRY

    def test_web_url_fetch_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "web_url_fetch" in TOOL_REGISTRY

    def test_stock_price_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "stock_price" in TOOL_REGISTRY

    def test_web_search_overwrites_builtin(self):
        """Our BraveWebSearch should have overwritten the built-in Serper-based one."""
        from qwen_agent.tools.base import TOOL_REGISTRY
        from sandbox_agent.tools.api_tools import BraveWebSearch
        tool_cls = TOOL_REGISTRY["web_search"]
        assert tool_cls is BraveWebSearch


class TestToolSchemas:

    def test_web_search_schema(self):
        from sandbox_agent.tools.api_tools import BraveWebSearch
        tool = BraveWebSearch()
        func = tool.function
        assert func["name"] == "web_search"
        assert "query" in func["parameters"]["properties"]

    def test_web_url_fetch_schema(self):
        from sandbox_agent.tools.api_tools import WebUrlFetch
        tool = WebUrlFetch()
        func = tool.function
        assert func["name"] == "web_url_fetch"
        assert "url" in func["parameters"]["properties"]

    def test_stock_price_schema(self):
        from sandbox_agent.tools.api_tools import StockPrice
        tool = StockPrice()
        func = tool.function
        assert func["name"] == "stock_price"
        assert "symbol" in func["parameters"]["properties"]
