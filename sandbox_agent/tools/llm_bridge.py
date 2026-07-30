"""LLM bridge — exposes an HTTP endpoint so code_interpreter can make LLM calls.

Falls back through the primary vLLM model (qwen3.6-27b-linux) then the backup
(qwen3.5 397B). Treats empty completions as failures with one retry per model.
Secured with a shared secret token generated at startup.
"""

import json
import logging
import secrets
import threading

import requests as http_requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from sandbox_agent.config import BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG

logger = logging.getLogger(__name__)

_server = None
_port = None
_auth_token = None

MAX_REQUEST_BODY = 1_000_000  # 1MB max request size


def _build_fallback_chain(llm_cfg: Optional[dict]) -> list:
    """Ordered (model, server) pairs tried in sequence on each /llm request.

    Starts with the SECONDARY tier (qwen3.6, 10 slots): bridge traffic is
    slot-unaware raw HTTP — often loops from exec scripts — and must not
    hammer the 3-slot MLX primary (laguna). A caller-supplied llm_cfg is only
    honored if it matches one of the two canonical models — this keeps the
    bridge from silently routing to a misconfigured model."""
    canonical = {
        BACKGROUND_LLM_CFG["model"]: BACKGROUND_LLM_CFG["model_server"],
        PRIMARY_LLM_CFG["model"]: PRIMARY_LLM_CFG["model_server"],
    }
    chain = []
    if llm_cfg and llm_cfg.get("model") in canonical:
        chain.append((llm_cfg["model"], llm_cfg["model_server"]))
    for m, s in canonical.items():
        if all(m != existing for existing, _ in chain):
            chain.append((m, s))
    return chain


def _create_handler(llm_cfg: Optional[dict], auth_token: str):
    """Create a request handler that calls the LLM via OpenAI-compatible API."""
    fallback_chain = _build_fallback_chain(llm_cfg)

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
                from sandbox_agent.model_tracker import model_start, model_done

                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                result_text = None
                last_error = None
                # Primary → backup chain. Retry each model once on empty content
                # before falling through (vLLM sometimes returns 200 with an empty
                # completion under context-window pressure).
                for model, server in fallback_chain:
                    succeeded = False
                    for attempt in (1, 2):
                        try:
                            model_start(model, f"llm_call: {prompt[:80]}")
                            resp = http_requests.post(
                                f"{server}/chat/completions",
                                json={
                                    "model": model,
                                    "messages": messages,
                                    "temperature": 0.6,
                                },
                                timeout=600,
                            )
                            resp.raise_for_status()
                            content = resp.json()["choices"][0]["message"].get("content") or ""
                            model_done(model)
                            if content.strip():
                                result_text = content
                                succeeded = True
                                break
                            last_error = Exception(f"{model}: empty completion (attempt {attempt})")
                        except Exception as e:
                            model_done(model)
                            last_error = e
                            logger.warning(f"llm_call failed on {model}: {e}")
                            break  # HTTP error — skip retry, try next model
                    if succeeded:
                        break

                if result_text is None:
                    raise last_error or Exception("All models failed")
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
    """Start the LLM bridge server. Returns the port number.

    `llm_cfg` is optional — when omitted (or when its model isn't in the
    canonical primary/backup set), the bridge falls through primary → backup
    on every request."""
    global _server, _port, _auth_token
    if _server is not None:
        return _port

    _auth_token = secrets.token_hex(32)
    handler = _create_handler(llm_cfg, _auth_token)
    chain = _build_fallback_chain(llm_cfg)

    # Find a free port, bind to localhost only
    _server = HTTPServer(("127.0.0.1", 0), handler)
    _port = _server.server_address[1]

    thread = threading.Thread(target=_server.serve_forever, daemon=True, name="llm-bridge")
    thread.start()
    chain_desc = " → ".join(f"{m}" for m, _ in chain)
    logger.info(f"LLM bridge started on port {_port} — chain: {chain_desc}")
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
"""
