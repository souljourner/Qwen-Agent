"""LLM bridge — exposes an HTTP endpoint so code_interpreter can make LLM calls.

Uses the backup model (qwen3.5-27b) on vLLM via OpenAI-compatible API.
Secured with a shared secret token generated at startup.
"""

import json
import logging
import secrets
import threading

import requests as http_requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from sandbox_agent.config import BACKGROUND_LLM_CFG

logger = logging.getLogger(__name__)

_server = None
_port = None
_auth_token = None

MAX_REQUEST_BODY = 1_000_000  # 1MB max request size


def _create_handler(llm_cfg: dict, auth_token: str):
    """Create a request handler that calls the LLM via OpenAI-compatible API."""

    class LLMHandler(BaseHTTPRequestHandler):

        def do_POST(self):
            if self.path != "/llm":
                self.send_error(404)
                return

            # Verify auth token
            provided_token = self.headers.get("X-Auth-Token", "")
            if provided_token != auth_token:
                self.send_error(403, "Invalid auth token")
                return

            # Enforce request size limit
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_REQUEST_BODY:
                self.send_error(413, f"Request body too large (max {MAX_REQUEST_BODY} bytes)")
                return

            body = json.loads(self.rfile.read(content_length))

            prompt = body.get("prompt", "")
            system = body.get("system", "")
            think = body.get("think", False)

            # Cap prompt to ~200k tokens (800k chars) to stay within model limits
            max_prompt_chars = 200000 * 4
            if len(prompt) > max_prompt_chars:
                prompt = prompt[:max_prompt_chars] + "\n... (truncated to fit token budget)"

            try:
                # Build messages
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                # Call vLLM OpenAI-compatible API
                resp = http_requests.post(
                    f"{llm_cfg['model_server']}/chat/completions",
                    json={
                        "model": llm_cfg["model"],
                        "messages": messages,
                        "max_tokens": 4096,
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                result_text = data["choices"][0]["message"].get("content", "")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"result": result_text}).encode())

            except BrokenPipeError:
                logger.debug("LLM bridge: client disconnected (BrokenPipeError)")
            except Exception as e:
                logger.warning(f"LLM bridge call failed: {e}")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                except BrokenPipeError:
                    pass  # Client already gone

        def log_message(self, format, *args):
            pass  # Suppress default HTTP logging

    return LLMHandler


def start_bridge(llm_cfg: Optional[dict] = None) -> int:
    """Start the LLM bridge server. Returns the port number."""
    global _server, _port, _auth_token
    if _server is not None:
        return _port

    cfg = llm_cfg or BACKGROUND_LLM_CFG
    _auth_token = secrets.token_hex(32)
    handler = _create_handler(cfg, _auth_token)

    # Find a free port, bind to localhost only
    _server = HTTPServer(("127.0.0.1", 0), handler)
    _port = _server.server_address[1]

    thread = threading.Thread(target=_server.serve_forever, daemon=True, name="llm-bridge")
    thread.start()
    logger.info(f"LLM bridge started on port {_port} -> {cfg['model']} @ {cfg['model_server']}")
    return _port


def get_port() -> Optional[int]:
    """Get the bridge port, or None if not started."""
    return _port


def get_auth_token() -> Optional[str]:
    """Get the auth token, or None if not started."""
    return _auth_token


def get_kernel_init_code(port: int) -> str:
    """Return Python code that injects llm_call() into the kernel namespace."""
    token = _auth_token
    return f"""\
import requests as _llm_requests

_LLM_BRIDGE_URL = "http://127.0.0.1:{port}/llm"
_LLM_AUTH_TOKEN = "{token}"

def llm_call(prompt, system="", think=False):
    \"\"\"Call the background LLM with a custom prompt.

    Args:
        prompt: The user message / instruction for the LLM.
        system: Optional system message to set the LLM's behavior.
        think: If True, enable chain-of-thought reasoning (slower but better for
            complex analysis, synthesis, and multi-step reasoning). Default False
            (fast mode, best for extraction, classification, formatting).

    Returns:
        The LLM's text response.

    Example:
        # Fast extraction — no thinking needed
        data = llm_call(
            f"Extract the price from this page:\\n{{html[:4000]}}",
            system="Return only the price as a number."
        )

        # Complex analysis — enable thinking
        analysis = llm_call(
            f"Analyze the market implications of this news:\\n{{article}}",
            system="Provide a structured analysis with bull/bear cases.",
            think=True
        )
    \"\"\"
    resp = _llm_requests.post(
        _LLM_BRIDGE_URL,
        json={{"prompt": prompt, "system": system, "think": think}},
        headers={{"X-Auth-Token": _LLM_AUTH_TOKEN}},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["result"]

print("llm_call() is available — call the background LLM with any prompt. Use think=True for complex reasoning.")
"""
