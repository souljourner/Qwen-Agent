"""Custom tools that call the local API on port 8080 for web search, URL fetch, and stock data."""

import json
from typing import Union

import requests

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import TOOLS_API_BASE
from sandbox_agent.tools.sanitizer import sanitize_web_content

# Timeout for API calls to port 8080 (seconds)
API_TIMEOUT = 120


def _call_tool_api(tool_name: str, arguments: dict, timeout: int = API_TIMEOUT) -> str:
    """Call the local tools API via SSE and return the result.

    POST /api/tools/execute with {"name": tool_name, "arguments": {...}}
    Parses SSE stream: collects event:result data.
    """
    url = f"{TOOLS_API_BASE}/api/tools/execute"
    payload = {"name": tool_name, "arguments": arguments}

    try:
        response = requests.post(url, json=payload, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.ConnectionError:
        return f"Error: Tools API at {TOOLS_API_BASE} is unreachable"
    except requests.Timeout:
        return f"Error: Tools API request timed out after {timeout}s"
    except requests.HTTPError as e:
        return f"Error: Tools API returned HTTP {e.response.status_code}"

    result_data = None
    current_event = None
    current_data_lines = []

    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue

        if line == "":
            # Blank line = end of SSE event
            if current_event == "result" and current_data_lines:
                raw = "\n".join(current_data_lines)
                try:
                    result_data = json.loads(raw)
                except json.JSONDecodeError:
                    result_data = raw
            current_event = None
            current_data_lines = []
            continue

        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[len("data:"):].strip())

    # Handle case where stream ends without trailing blank line
    if current_event == "result" and current_data_lines and result_data is None:
        raw = "\n".join(current_data_lines)
        try:
            result_data = json.loads(raw)
        except json.JSONDecodeError:
            result_data = raw

    if result_data is None:
        return "(no result returned from tool API)"

    if isinstance(result_data, dict) and "error" in result_data:
        return f"Error: {result_data['error']}"

    if isinstance(result_data, dict) and "results" in result_data:
        return json.dumps(result_data["results"], ensure_ascii=False, indent=2)

    return json.dumps(result_data, ensure_ascii=False, indent=2)


@register_tool("web_search", allow_overwrite=True)
class BraveWebSearch(BaseTool):
    """Search the web using Brave Search via the local tools API."""

    name = "web_search"
    description = "Search the web for current information on any topic."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        raw_result = _call_tool_api("web_search", {"query": params["query"]})
        return sanitize_web_content(raw_result)


@register_tool("web_url_fetch")
class WebUrlFetch(BaseTool):
    """Fetch a URL and return its content as markdown."""

    name = "web_url_fetch"
    description = "Fetch a URL and return its content as markdown or plain text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            }
        },
        "required": ["url"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        raw_result = _call_tool_api("web_url_fetch", {"url": params["url"]})
        return sanitize_web_content(raw_result)


@register_tool("stock_price")
class StockPrice(BaseTool):
    """Fetch current stock price and market data."""

    name = "stock_price"
    description = "Fetch current stock price and market data for a ticker symbol."
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., AAPL, GOOGL, MSFT).",
            }
        },
        "required": ["symbol"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        raw_result = _call_tool_api("stock_price", {"symbol": params["symbol"]})
        return sanitize_web_content(raw_result)
