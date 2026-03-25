"""Local code interpreter — runs Python via a subprocess Jupyter kernel (no Docker-in-Docker).

Based on the benchmark code_interpreter pattern from qwen_agent, adapted as a
registered tool that works inside an already-sandboxed Docker container.
"""

import atexit
import base64
import io
import json
import logging
import os
import queue
import re
import subprocess
import sys
import time
import uuid
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

WORK_DIR = os.path.join(DATA_DIR, "code_interpreter")

LAUNCH_KERNEL_PY = "from ipykernel import kernelapp as app\napp.launch_new_instance()\n"

INIT_CODE = """\
def input(*args, **kwargs):
    raise NotImplementedError('Python input() function is disabled.')

import os, math, re, json
import numpy as np
import pandas as pd
"""

# LLM bridge init code is appended after INIT_CODE when the kernel starts
_llm_init_code = None

_kernel_client = None
_kernel_process = None


def _escape_ansi(text: str) -> str:
    return re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]").sub("", text)


def _start_kernel():
    """Start a local Jupyter kernel as a subprocess and return a BlockingKernelClient."""
    global _kernel_process
    from jupyter_client import BlockingKernelClient

    os.makedirs(WORK_DIR, exist_ok=True)

    pid = os.getpid()
    connection_file = os.path.join(WORK_DIR, f"kernel_connection_{pid}.json")
    launch_script = os.path.join(WORK_DIR, f"launch_kernel_{pid}.py")

    # Clean up stale files
    for f in [connection_file, launch_script]:
        if os.path.exists(f):
            os.remove(f)

    with open(launch_script, "w") as fout:
        fout.write(LAUNCH_KERNEL_PY)

    _kernel_process = subprocess.Popen(
        [sys.executable, launch_script, "--IPKernelApp.connection_file", connection_file, "--quiet"],
        cwd=WORK_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info(f"Kernel started (PID={_kernel_process.pid})")

    # Wait for connection file to be written
    for _ in range(100):  # 10 seconds max
        if os.path.isfile(connection_file):
            try:
                with open(connection_file, "r") as fp:
                    json.load(fp)
                break
            except json.JSONDecodeError:
                pass
        time.sleep(0.1)
    else:
        raise RuntimeError("Kernel connection file not created within 10 seconds")

    kc = BlockingKernelClient(connection_file=connection_file)
    kc.load_connection_file()
    kc.start_channels()
    kc.wait_for_ready()

    # Run init code
    kc.execute(INIT_CODE)
    _drain_output(kc)

    # Inject llm_call() if bridge is running
    if _llm_init_code:
        kc.execute(_llm_init_code)
        _drain_output(kc)

    return kc


def _drain_output(kc, timeout_seconds: int = 5) -> None:
    """Drain all pending messages from the kernel."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            msg = kc.get_iopub_msg(timeout=0.5)
            if msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle":
                break
        except queue.Empty:
            break


def _get_kernel():
    """Get or start the kernel client."""
    global _kernel_client
    if _kernel_client is None:
        _kernel_client = _start_kernel()
    return _kernel_client


def _cleanup_kernel():
    """Clean up kernel on exit."""
    global _kernel_process, _kernel_client
    if _kernel_client:
        try:
            _kernel_client.stop_channels()
        except Exception:
            pass
    if _kernel_process:
        try:
            _kernel_process.terminate()
            _kernel_process.wait(timeout=5)
        except Exception:
            pass


atexit.register(_cleanup_kernel)


MIN_TIMEOUT = 600       # 10 minutes minimum
SECONDS_PER_TOKEN = 0.1  # ~10 tokens/sec generation speed as budget estimate
MAX_TIMEOUT = 7200      # 2 hour hard cap


def _compute_timeout(code: str) -> int:
    """Compute timeout based on code size. Longer code = more work = more time.

    Heuristic: 1 token ≈ 4 chars, budget ~0.1s per token of code.
    Minimum 10 minutes, max 2 hours.
    """
    estimated_tokens = len(code) // 4
    computed = int(estimated_tokens * SECONDS_PER_TOKEN)
    return max(MIN_TIMEOUT, min(computed, MAX_TIMEOUT))


def _execute_code(code: str, timeout: int = 0) -> str:
    """Execute Python code in the kernel and return the output."""
    if timeout <= 0:
        timeout = _compute_timeout(code)
    kc = _get_kernel()
    kc.wait_for_ready()

    # Add timeout guard
    wrapped = f"""\
import signal
def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out after {timeout} seconds")
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm({timeout})
{code}
signal.alarm(0)
"""
    kc.execute(wrapped)

    result_parts = []
    image_idx = 0

    while True:
        finished = False
        try:
            msg = kc.get_iopub_msg(timeout=timeout + 5)
            msg_type = msg["msg_type"]

            if msg_type == "status":
                if msg["content"].get("execution_state") == "idle":
                    finished = True
            elif msg_type == "execute_result":
                text = msg["content"]["data"].get("text/plain", "")
                if text:
                    result_parts.append(text)
                if "image/png" in msg["content"]["data"]:
                    image_idx += 1
                    path = _save_image(msg["content"]["data"]["image/png"])
                    result_parts.append(f"[Image saved: {path}]")
            elif msg_type == "display_data":
                if "image/png" in msg["content"]["data"]:
                    image_idx += 1
                    path = _save_image(msg["content"]["data"]["image/png"])
                    result_parts.append(f"[Image saved: {path}]")
                else:
                    text = msg["content"]["data"].get("text/plain", "")
                    if text:
                        result_parts.append(text)
            elif msg_type == "stream":
                text = msg["content"]["text"]
                if text.strip():
                    result_parts.append(text)
            elif msg_type == "error":
                tb = _escape_ansi("\n".join(msg["content"]["traceback"]))
                if "Code execution timed out" in tb:
                    result_parts.append(f"Timeout: no response after {timeout} seconds.")
                else:
                    result_parts.append(f"Error:\n{tb}")
                finished = True

        except queue.Empty:
            result_parts.append(f"Timeout: no response after {timeout} seconds.")
            finished = True
        except Exception as e:
            result_parts.append(f"Kernel error: {e}")
            finished = True

        if finished:
            break

    # Reset alarm
    try:
        kc.execute("signal.alarm(0)")
        _drain_output(kc, timeout_seconds=2)
    except Exception:
        pass

    result = "\n".join(result_parts).strip() or "(no output)"

    # Cap output to token budget
    from sandbox_agent.token_budget import truncate_output
    from sandbox_agent.config import MAX_CODE_OUTPUT_TOKENS
    return truncate_output(result, MAX_CODE_OUTPUT_TOKENS)


def _save_image(image_b64: str) -> str:
    """Save a base64-encoded PNG image to the work directory."""
    import PIL.Image
    filename = f"{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(WORK_DIR, filename)
    png_bytes = base64.b64decode(image_b64)
    PIL.Image.open(io.BytesIO(png_bytes)).save(filepath, "png")
    return filepath


@register_tool("code_interpreter", allow_overwrite=True)
class LocalCodeInterpreter(BaseTool):
    """Execute Python code in a local Jupyter kernel."""

    name = "code_interpreter"
    description = (
        "Execute Python code. Has numpy, pandas, requests, and llm_call(prompt, system='') "
        "for background LLM calls. Variables persist between calls. "
        "CRITICAL RULES: "
        "1) Write intermediate data to files (use DATA_DIR=os.getenv('DATA_DIR','data')), never to stdout. "
        "2) For multiple URLs: ONE script, for-loop, save raw content to .jsonl file, process from file with llm_call(). "
        "3) print() ONLY a 3-5 line summary. Save full results to a file. "
        "4) NEVER print raw HTML/page content. NEVER make separate calls per URL."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
        },
        "required": ["code"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        code = params["code"]
        return _execute_code(code)
