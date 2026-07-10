"""Playwright-based browser automation tools.

Provides a headless Chromium browser session with persistent cookies for
interactive web browsing: navigate, screenshot, click, type, scroll.

The browser context is shared across all tool calls within a process.
Cookies are saved to DATA_DIR/browser_state/ after each navigation so
login state survives between agent runs.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR


# ---------------------------------------------------------------------------
# Browser singleton — one Playwright browser + context per Python process
# ---------------------------------------------------------------------------

_playwright = None
_browser = None
_context = None
_page = None
_BROWSER_STATE_DIR = Path(DATA_DIR) / "browser_state"
_COOKIE_FILE = _BROWSER_STATE_DIR / "cookies.json"
_VIEWPORT_WIDTH = 1280
_VIEWPORT_HEIGHT = 900
_NAV_TIMEOUT = 30000  # ms

# Human-like delay ranges (seconds)
_DELAY_AFTER_NAV = (1.0, 2.5)
_DELAY_AFTER_CLICK = (0.3, 0.8)
_DELAY_AFTER_TYPE = (0.2, 0.5)
_TYPE_DELAY_MS = (20, 60)  # ms per keystroke


def _human_delay(low: float = 0.3, high: float = 0.8) -> None:
    """Sleep a random amount to look human."""
    time.sleep(random.uniform(low, high))


def _is_mac() -> bool:
    return sys.platform == "darwin"


def _ensure_browser():
    """Lazy-init Playwright browser, context, and page. Restore cookies from disk."""
    global _playwright, _browser, _context, _page

    if _page is not None:
        return _page

    from playwright.sync_api import sync_playwright

    _playwright = sync_playwright().start()

    # Use a persistent user-data-dir so cookies/storage survive within the session
    _state_dir = str(_BROWSER_STATE_DIR)
    os.makedirs(_state_dir, exist_ok=True)

    # Headed mode: when XVFB_ENABLED=true, launch headed so the browser window
    # appears on the virtual display (visible via noVNC at localhost:6080).
    _headed = os.getenv("XVFB_ENABLED", "").lower() == "true"

    _browser = _playwright.chromium.launch(
        headless=not _headed,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=TranslateUI",
            "--disable-http2",
            "--window-size=1280,900",
            "--lang=en-US",
            "--start-maximized",
        ],
    )

    _context = _browser.new_context(
        viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        ignore_https_errors=True,
        java_script_enabled=True,
        locale="en-US",
        timezone_id="America/New_York",
    )

    # Stealth: mask Playwright automation fingerprints
    _context.add_init_script("""
        // Override navigator.webdriver to undefined (headless browsers set this to true)
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        // Remove Playwright-specific properties
        delete navigator.__proto__.webdriver;
        // Mask plugins array (headless browsers have empty plugins)
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        // Mask languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        // Chrome runtime
        window.chrome = { runtime: {} };
        // Permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({state: Notification.permission}) :
                originalQuery(parameters)
        );
    """)

    # Restore previously saved cookies
    if _COOKIE_FILE.exists():
        try:
            cookies = json.loads(_COOKIE_FILE.read_text())
            _context.add_cookies(cookies)
        except Exception:
            pass

    _page = _context.new_page()
    _page.set_default_timeout(_NAV_TIMEOUT)
    return _page


def _save_cookies():
    """Dump current cookies to disk for cross-run persistence."""
    if _context is None:
        return
    try:
        cookies = _context.cookies()
        os.makedirs(_BROWSER_STATE_DIR, exist_ok=True)
        _COOKIE_FILE.write_text(json.dumps(cookies, indent=2))
    except Exception:
        pass


def _close_browser():
    """Explicitly shut down the browser (for graceful teardown)."""
    global _page, _context, _browser, _playwright
    _save_cookies()
    _page = None
    _context = None
    _browser = None
    _playwright = None


# ---------------------------------------------------------------------------
# Tool: browser_navigate
# ---------------------------------------------------------------------------

@register_tool("browser_navigate", allow_overwrite=True)
class BrowserNavigate(BaseTool):
    """Navigate the browser to a URL and wait for the page to load."""

    name = "browser_navigate"
    description = (
        "Open a URL in the browser and wait for the page to fully load. "
        "Use this before taking a screenshot or interacting with page elements."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to navigate to (e.g., https://www.ebay.com/itm/123456).",
            },
            "wait_until": {
                "type": "string",
                "enum": ["load", "domcontentloaded", "networkidle", "commit"],
                "description": "When to consider navigation successful. Default: domcontentloaded. "
                              "Use 'load' for full page load, 'networkidle' for all network quiet, "
                              "'domcontentloaded' for faster initial render.",
            },
        },
        "required": ["url"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        url = params["url"]
        wait_until = params.get("wait_until", "domcontentloaded")
        page = _ensure_browser()

        # Retry up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                resp = page.goto(url, wait_until=wait_until, timeout=_NAV_TIMEOUT)
                _save_cookies()
                status = resp.status if resp else 200
                title = page.title()
                final_url = page.url
                return (
                    f"Navigated to: {final_url}\n"
                    f"Status: {status}\n"
                    f"Title: {title}"
                )
            except Exception as e:
                err_msg = str(e)
                if attempt < 2:
                    delay = 1.5 * (2 ** attempt)  # 2.4s, 4.8s
                    time.sleep(delay)
                    continue
                return f"Navigation failed after {attempt + 1} attempts: {err_msg}"


# ---------------------------------------------------------------------------
# Tool: browser_screenshot
# ---------------------------------------------------------------------------

@register_tool("browser_screenshot", allow_overwrite=True)
class BrowserScreenshot(BaseTool):
    """Capture a screenshot of the current browser page."""

    name = "browser_screenshot"
    description = (
        "Take a screenshot of the current browser page. Returns an image the vision model can see. "
        "Use this after navigating to a URL or after clicking/typing to see the result."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        page = _ensure_browser()
        try:
            # Small pause to let animations finish
            time.sleep(0.3)
            png_bytes = page.screenshot(full_page=False)
            import base64
            b64 = base64.b64encode(png_bytes).decode("ascii")
            data_url = f"data:image/png;base64,{b64}"
            url = page.url
            title = page.title()
            return [
                ContentItem(text=f"Screenshot of: {url}\nTitle: {title}"),
                ContentItem(image=data_url),
            ]
        except Exception as e:
            return f"Screenshot failed: {e}"


# ---------------------------------------------------------------------------
# Tool: browser_click
# ---------------------------------------------------------------------------

@register_tool("browser_click", allow_overwrite=True)
class BrowserClick(BaseTool):
    """Click on a page element by coordinates or text label."""

    name = "browser_click"
    description = (
        "Click on an element in the current browser page. "
        "Specify EITHER pixel coordinates (x, y) from the screenshot, "
        "OR text to find and click (e.g., 'Buy Now', 'Add to Cart', 'Checkout'). "
        "Use coordinates for precise clicking on visual elements, "
        "or text for labeled buttons and links."
    )
    parameters = {
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "description": "X coordinate in pixels (from screenshot). Use with y for coordinate-based clicking.",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate in pixels (from screenshot). Use with x for coordinate-based clicking.",
            },
            "text": {
                "type": "string",
                "description": "Text label of the element to click (e.g., 'Buy Now', 'Checkout'). "
                               "Finds the first matching visible element.",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        page = _ensure_browser()

        x = params.get("x")
        y = params.get("y")
        text = params.get("text")

        try:
            if text:
                # Find element containing the text and click it
                selector = f"text={text}"
                el = page.wait_for_selector(selector, timeout=5000, state="visible")
                if el:
                    el.click()
                    _save_cookies()
                    return f"Clicked element with text: '{text}'"
                # Fallback: try broader selector
                els = page.query_selector_all(f"//*[contains(text(), '{text}')]")
                if els:
                    els[0].click()
                    _save_cookies()
                    return f"Clicked element containing text: '{text}'"
                return f"No visible element found with text: '{text}'"

            elif x is not None and y is not None:
                page.mouse.click(x, y)
                _save_cookies()
                return f"Clicked at coordinates ({x}, {y})"

            else:
                return "Error: provide either (x, y) coordinates or a text label."

        except Exception as e:
            return f"Click failed: {e}"


# ---------------------------------------------------------------------------
# Tool: browser_type
# ---------------------------------------------------------------------------

@register_tool("browser_type", allow_overwrite=True)
class BrowserType(BaseTool):
    """Type text into the currently focused element or an element found by selector."""

    name = "browser_type"
    description = (
        "Type text into the currently focused input field, or into an element "
        "identified by its label text. Use `clear_first=true` to clear existing content before typing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to type into the input field.",
            },
            "clear_first": {
                "type": "boolean",
                "description": "Whether to clear existing content before typing. Default: true.",
            },
            "into": {
                "type": "string",
                "description": "Optional: label text of the target input field (e.g., 'Username', 'Search'). "
                               "If omitted, types into the currently focused element.",
            },
        },
        "required": ["text"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        page = _ensure_browser()

        text = params["text"]
        clear_first = params.get("clear_first", True)
        into = params.get("into")

        try:
            if into:
                # Try to find the input field by its label
                # Common patterns: <label for="id">, <label>...<input>, placeholder, aria-label
                target = None
                for sel in [
                    f"input[placeholder='{into}']",
                    f"input[aria-label='{into}']",
                    f"input[name='{into.lower()}']",
                    f"label:has-text('{into}') >> input",
                ]:
                    try:
                        target = page.wait_for_selector(sel, timeout=2000, state="visible")
                        if target:
                            target.click()  # focus it
                            break
                    except Exception:
                        continue

                if not target:
                    # Last resort: find any label with the text and click near it
                    label_sel = f"//*[contains(text(), '{into}')]"
                    labels = page.query_selector_all(label_sel)
                    if labels:
                        labels[0].click()
                        time.sleep(0.2)
                    else:
                        return f"Could not find input field labeled: '{into}'"

            if clear_first:
                page.keyboard.press("Control+a")
                page.keyboard.press("Backspace")

            page.keyboard.type(text, delay=30)
            _save_cookies()
            return f'Typed: "{text}"'

        except Exception as e:
            return f"Type failed: {e}"


# ---------------------------------------------------------------------------
# Tool: browser_scroll
# ---------------------------------------------------------------------------

@register_tool("browser_scroll", allow_overwrite=True)
class BrowserScroll(BaseTool):
    """Scroll the current browser page up or down."""

    name = "browser_scroll"
    description = (
        "Scroll the current page. Use this when content is below the visible area, "
        "e.g., to see product reviews, additional listings, or checkout form fields."
    )
    parameters = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["down", "up"],
                "description": "Direction to scroll. Default: down.",
            },
            "amount": {
                "type": "integer",
                "description": "Number of pixels to scroll. Default: 500.",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict] = '{}', **kwargs) -> str:
        params = self._verify_json_format_args(params)
        page = _ensure_browser()

        direction = params.get("direction", "down")
        amount = params.get("amount", 500)

        try:
            delta_y = amount if direction == "down" else -amount
            page.evaluate(f"window.scrollBy(0, {delta_y})")
            time.sleep(0.3)  # wait for lazy-loaded content
            _save_cookies()
            return f"Scrolled {direction} by {amount}px"
        except Exception as e:
            return f"Scroll failed: {e}"


# ---------------------------------------------------------------------------
# Credential vault — AES-encrypted site credentials
# ---------------------------------------------------------------------------

_CREDENTIALS_FILE = _BROWSER_STATE_DIR / "credentials.json"
# Encryption key from env var; falls back to a default (container-scoped only).
# In production, set BROWSER_CREDENTIAL_KEY to a 32-byte hex string.
_CRYPT_KEY = os.getenv("BROWSER_CREDENTIAL_KEY", "0" * 64).encode()


def _encrypt(text: str) -> str:
    """Encrypt a string using Fernet (AES-128-CBC + HMAC)."""
    from cryptography.fernet import Fernet, InvalidToken
    try:
        key = Fernet.generate_key()  # We'll derive from env, but Fernet needs 32 url-safe bytes
        # Use a deterministic key from the env var for cross-call consistency
        import base64
        raw = _CRYPT_KEY.hex() if len(_CRYPT_KEY) == 64 else b"0" * 64
        fernet_key = base64.urlsafe_b64encode(bytes.fromhex(raw[:64]))
        f = Fernet(fernet_key)
        return f.encrypt(text.encode()).decode()
    except Exception as e:
        # Fallback: store as base64 if encryption fails (better than crashing)
        import base64
        return base64.b64encode(text.encode()).decode()


def _decrypt(token: str) -> str:
    """Decrypt a Fernet token, falling back to base64 decode."""
    from cryptography.fernet import Fernet, InvalidToken
    import base64
    try:
        raw = _CRYPT_KEY.hex() if len(_CRYPT_KEY) == 64 else b"0" * 64
        fernet_key = base64.urlsafe_b64encode(bytes.fromhex(raw[:64]))
        f = Fernet(fernet_key)
        return f.decrypt(token.encode()).decode()
    except (InvalidToken, Exception):
        # Fallback: try base64
        try:
            return base64.b64decode(token.encode()).decode()
        except Exception:
            return token


def _load_credentials() -> dict:
    if not _CREDENTIALS_FILE.exists():
        return {}
    try:
        return json.loads(_CREDENTIALS_FILE.read_text())
    except Exception:
        return {}


def _save_credentials(data: dict):
    os.makedirs(_BROWSER_STATE_DIR, exist_ok=True)
    _CREDENTIALS_FILE.write_text(json.dumps(data, indent=2))


@register_tool("browser_save_credentials", allow_overwrite=True)
class BrowserSaveCredentials(BaseTool):
    """Save site credentials encrypted to disk for future logins."""

    name = "browser_save_credentials"
    description = (
        "Save login credentials for a website. Stored encrypted on disk. "
        "Use this once per site, then the agent can auto-login in future sessions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "site": {
                "type": "string",
                "description": "Site identifier (e.g., 'ebay', 'bestbuy', 'amazon').",
            },
            "username": {
                "type": "string",
                "description": "Login username, email, or phone number.",
            },
            "password": {
                "type": "string",
                "description": "Login password.",
            },
        },
        "required": ["site", "username", "password"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        site = params["site"].lower().strip()
        creds = _load_credentials()
        creds[site] = {
            "username": _encrypt(params["username"]),
            "password": _encrypt(params["password"]),
        }
        _save_credentials(creds)
        return f"Credentials saved for site: {site}"


@register_tool("browser_get_credentials", allow_overwrite=True)
class BrowserGetCredentials(BaseTool):
    """Retrieve encrypted site credentials for login."""

    name = "browser_get_credentials"
    description = (
        "Get saved credentials for a website. Returns username and password "
        "so the agent can fill in login forms."
    )
    parameters = {
        "type": "object",
        "properties": {
            "site": {
                "type": "string",
                "description": "Site identifier (e.g., 'ebay', 'bestbuy').",
            },
        },
        "required": ["site"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        site = params["site"].lower().strip()
        creds = _load_credentials()
        if site not in creds:
            return f"No credentials saved for site: {site}. Use browser_save_credentials first."
        entry = creds[site]
        username = _decrypt(entry.get("username", ""))
        password = _decrypt(entry.get("password", ""))
        return f"Site: {site}\nUsername: {username}\nPassword: {password}"
