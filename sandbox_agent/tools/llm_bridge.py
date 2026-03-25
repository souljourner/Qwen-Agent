"""LLM bridge — exposes an HTTP endpoint so code_interpreter can make LLM calls.

Runs a lightweight HTTP server in a background thread. The Jupyter kernel
calls it via requests.post("http://127.0.0.1:{port}/llm", json={...}).
Secured with a shared secret token generated at startup.
"""

import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from sandbox_agent.config import BACKGROUND_LLM_CFG

logger = logging.getLogger(__name__)

_server = None
_port = None
_auth_token = None

MAX_REQUEST_BODY = 1_000_000  # 1MB max request size


def _create_handler(llm_cfg: dict, auth_token: str):
    """Create a request handler that uses the given LLM config."""

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

            # Cap prompt to ~200k tokens (800k chars) to stay within model limits
            max_prompt_chars = 200000 * 4
            if len(prompt) > max_prompt_chars:
                prompt = prompt[:max_prompt_chars] + "\n... (truncated to fit token budget)"

            try:
                from qwen_agent.llm import get_chat_model
                from qwen_agent.llm.schema import Message

                llm = get_chat_model(llm_cfg)
                messages = []
                # Prepend /no_think to suppress Qwen3's hidden reasoning tokens
                # This significantly speeds up responses for extraction/classification tasks
                system_content = "/no_think\n" + system if system else "/no_think"
                messages.append(Message(role="system", content=system_content))
                messages.append(Message(role="user", content=prompt))

                # Non-streaming call
                response = []
                for response in llm.chat(messages=messages, stream=False):
                    pass

                # Extract text
                result_text = ""
                if response:
                    for msg in response:
                        if hasattr(msg, "content") and isinstance(msg.content, str):
                            result_text += msg.content

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
    logger.info(f"LLM bridge started on port {_port}")
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

def llm_call(prompt, system=""):
    \"\"\"Call the background LLM with a custom prompt.

    Args:
        prompt: The user message / instruction for the LLM.
            Put your task-specific prompt here — e.g., "Extract all product names
            and prices from this text: ..." or "Classify this text as positive
            or negative: ..."
        system: Optional system message to set the LLM's behavior.

    Returns:
        The LLM's text response.

    Example:
        # Extract specific data from a web page
        import requests
        html = requests.get("https://example.com").text
        result = llm_call(
            f"Extract all email addresses from this text:\\n{{html}}",
            system="You are a precise data extractor. Return only the data requested, one item per line."
        )
        print(result)

        # Classify content
        result = llm_call(
            f"Is this article about technology, finance, or politics?\\n{{article_text}}",
            system="Respond with exactly one word: technology, finance, or politics."
        )
    \"\"\"
    resp = _llm_requests.post(
        _LLM_BRIDGE_URL,
        json={{"prompt": prompt, "system": system}},
        headers={{"X-Auth-Token": _LLM_AUTH_TOKEN}},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["result"]

print("llm_call() is available — call the background LLM with any prompt.")
"""
