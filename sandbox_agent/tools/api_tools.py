"""Custom tools that call the local API on port 8080 for web search, URL fetch, and stock data."""

import json
import threading
import time
from typing import Union

import requests

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import TOOLS_API_BASE
from sandbox_agent.tools.sanitizer import sanitize_web_content

# Timeout for API calls to port 8080 (seconds)
API_TIMEOUT = 120

# Brave Search rate limit: 1 request per second
_brave_search_lock = threading.Lock()
_brave_search_last_call = 0.0
BRAVE_SEARCH_MIN_INTERVAL = 2.0  # seconds (Brave Search rate limit)


def _rate_limit_brave_search():
    """Enforce 1 request/second rate limit for Brave Search."""
    global _brave_search_last_call
    with _brave_search_lock:
        now = time.monotonic()
        elapsed = now - _brave_search_last_call
        if elapsed < BRAVE_SEARCH_MIN_INTERVAL:
            time.sleep(BRAVE_SEARCH_MIN_INTERVAL - elapsed)
        _brave_search_last_call = time.monotonic()


def _call_tool_api(tool_name: str,
                   arguments: dict,
                   timeout: int = API_TIMEOUT,
                   return_obj: bool = False) -> Union[str, dict, list]:
    """Call the local tools API via SSE and return the result.

    POST /api/tools/execute with {"name": tool_name, "arguments": {...}}
    Parses SSE stream: collects event:result data.

    If ``return_obj`` is True, success results are returned as the parsed
    ``results`` object (dict/list) instead of a JSON-dumped string. Errors are
    always returned as an "Error: ..." string regardless of ``return_obj``.
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
        if return_obj:
            return result_data["results"]
        return json.dumps(result_data["results"], ensure_ascii=False, indent=2)

    if return_obj:
        return result_data
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
        _rate_limit_brave_search()
        raw_result = _call_tool_api("web_search", {"query": params["query"]})
        return sanitize_web_content(raw_result)


def _format_fetch_result(results: dict) -> str:
    """Sanitize the fetched page content and append a pagination hint.

    The server returns {"content", "total_chars", "offset", "returned_chars",
    "has_more", ...}. Only the page content is run through sanitize_web_content
    (it's untrusted web text); the pagination hint we generate ourselves is
    appended after the [TOOL_OUTPUT] delimiters so it stays trusted.
    """
    content = results.get("content") or ""
    body = sanitize_web_content(content)

    if results.get("has_more"):
        offset = results.get("offset") or 0
        returned = results.get("returned_chars")
        if returned is None:
            returned = len(content)
        total = results.get("total_chars")
        next_offset = offset + returned
        remaining = f"{total - next_offset} more chars" if isinstance(total, int) else "more content"
        body += (f"\n\n[web_url_fetch: showing chars {offset}-{next_offset} of {total}. "
                 f"{remaining} available — call again with the same url and offset={next_offset} "
                 f"to continue.]")
    return body


@register_tool("web_url_fetch")
class WebUrlFetch(BaseTool):
    """Fetch a URL and return its content as markdown."""

    name = "web_url_fetch"
    description = (
        "Fetch a single web page and return its text content as markdown. Best for a quick "
        "lookup of one page. Long pages are paginated: the result reports how many characters "
        "are left and gives you the offset to continue from. For fetching many URLs or doing "
        "bulk/heavy processing, use requests.get() inside code_interpreter instead — that keeps "
        "the raw content out of your context.")
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "offset": {
                "type": "integer",
                "description": ("Character offset to start from within the page's content. "
                                "Use it to page through a long page (default 0)."),
            },
            "max_chars": {
                "type": "integer",
                "description": ("Maximum number of characters to return from the offset. Omit to "
                                "return as much as available; if the page is longer the result is "
                                "flagged so you can fetch the rest with a larger offset."),
            },
        },
        "required": ["url"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        args = {"url": params["url"]}
        if params.get("offset") is not None:
            args["offset"] = params["offset"]
        if params.get("max_chars") is not None:
            args["max_chars"] = params["max_chars"]
        result = _call_tool_api("web_url_fetch", args, return_obj=True)
        if isinstance(result, str):
            return result  # error string ("Error: ...") or "(no result ...)"
        if isinstance(result, dict):
            return _format_fetch_result(result)
        return sanitize_web_content(json.dumps(result, ensure_ascii=False, indent=2))


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
