"""Local code interpreter — runs Python as a fresh subprocess per call.

Each `code_interpreter` call writes the user's code (after a fixed prelude that
imports numpy/pandas, sets DATA_DIR/PROJECTS_DIR, and — if the LLM bridge is up —
defines llm_call()) to a temp .py file and runs it with `subprocess.Popen(...,
start_new_session=True)` + `communicate(timeout=...)`, killing the whole process
group on timeout/error. No persistent interpreter: state does NOT carry over
between calls — persist anything you need to a file. This mirrors `exec_tool` and
matches how OpenClaw/Hermes run code; it eliminates the wedged-Jupyter-kernel hang
class. Persistence is the filesystem (DATA_DIR/PROJECTS_DIR) + the container's
installed packages.
"""

import logging
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR, MAX_CODE_OUTPUT_TOKENS
from sandbox_agent.token_budget import truncate_output
from sandbox_agent.tools.exec_tool import kill_process_group

logger = logging.getLogger(__name__)

WORK_DIR = os.path.join(DATA_DIR, "scratch")

INIT_CODE = f"""\
def input(*args, **kwargs):
    raise NotImplementedError('Python input() function is disabled.')

import os, math, re, json
try:
    import numpy as np
except ImportError:
    pass
try:
    import pandas as pd
except ImportError:
    pass

# Pre-configured paths
DATA_DIR = os.getenv('DATA_DIR', '{DATA_DIR}')
PROJECTS_DIR = os.path.join(DATA_DIR, 'projects')
"""

# LLM bridge init code (defines llm_call()) — set at bootstrap by main.py /
# chat_app.py via `code_interpreter._llm_init_code = get_kernel_init_code(port)`.
# Prepended to every script. If unset, scripts that call llm_call() get a
# NameError (same as the old kernel started without the bridge).
_llm_init_code = None


def _escape_ansi(text: str) -> str:
    return re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]").sub("", text)


MIN_TIMEOUT = 600        # 10 minutes minimum
SECONDS_PER_TOKEN = 0.1  # ~10 tokens/sec generation speed as budget estimate
MAX_TIMEOUT = 1200       # 20 minute hard cap (a longer job should be a scheduled task)


def _compute_timeout(code: str) -> int:
    """Compute timeout based on code size. Longer code = more work = more time.

    Heuristic: 1 token ≈ 4 chars, budget ~0.1s per token of code.
    Minimum 10 minutes, max 20 minutes.
    """
    estimated_tokens = len(code) // 4
    computed = int(estimated_tokens * SECONDS_PER_TOKEN)
    return max(MIN_TIMEOUT, min(computed, MAX_TIMEOUT))


def _build_script(code: str) -> str:
    """Assemble the full script: fixed prelude + llm_call() def + user code."""
    return INIT_CODE + "\n" + (_llm_init_code or "") + "\n\n# --- user code ---\n" + code + "\n"


def _execute_code(code: str, timeout: int = 0) -> str:
    """Run `code` as a fresh Python subprocess; return exec-style output."""
    if timeout <= 0:
        timeout = _compute_timeout(code)

    os.makedirs(WORK_DIR, exist_ok=True)
    script_path = os.path.join(WORK_DIR, f"_ci_{uuid.uuid4().hex[:8]}.py")
    with open(script_path, "w") as f:
        f.write(_build_script(code))

    # /app on PYTHONPATH so the script can `from sandbox_agent.tools.llm_client
    # import llm_batch` (the editable install usually covers this already, but
    # be explicit — same as exec_tool).
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "/app" + (os.pathsep + existing_pp if existing_pp else "")

    from sandbox_agent.cancellation import register_child_pgid, unregister_child_pgid
    start = time.monotonic()
    proc = None
    try:
        # `start_new_session=True`: the script (and anything it spawns) gets its
        # own process group / session, isolated from PID 1. A `kill 0` / `pkill`
        # inside the script can't reach the container's main process.
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=DATA_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        # Register the child's process group so cancel_task can SIGKILL it
        # (start_new_session=True → pgid == pid).
        register_child_pgid(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # SIGKILL the whole group — a `server &` may have left children.
            kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            partial = _escape_ansi(((stdout or "") + ("\n" + stderr if stderr else "")).strip())
            partial = truncate_output(partial, MAX_CODE_OUTPUT_TOKENS) if partial else "(no output before timeout)"
            return f"Timed out after {timeout}s | {len(partial):,} chars\n\n{partial}"

        duration = time.monotonic() - start
        output = (stdout or "")
        if stderr:
            output += ("\n" if output else "") + stderr
        # NOTE: on a clean exit we deliberately do NOT kill the process group —
        # a server the script started with `&` is reparented to init and keeps
        # listening, so the agent can curl it from the next call (matches exec).
        output = _escape_ansi(output.strip()) or "(no output)"
        output = truncate_output(output, MAX_CODE_OUTPUT_TOKENS)
        return f"Exit {proc.returncode} | {duration:.1f}s | {len(output):,} chars\n\n{output}"

    except Exception as e:  # noqa: BLE001
        if proc is not None:
            kill_process_group(proc)
        return f"Error: {e}"
    finally:
        if proc is not None:
            unregister_child_pgid(proc.pid)
        try:
            os.unlink(script_path)
        except OSError:
            pass


@register_tool("code_interpreter", allow_overwrite=True)
class LocalCodeInterpreter(BaseTool):
    """Execute Python code as a fresh subprocess (no persistent state)."""

    name = "code_interpreter"
    description = (
        "Execute Python code. Has numpy, pandas, requests, and "
        "llm_call(prompt, system='', think=False) for background LLM calls. "
        "Use think=False (default) for fast extraction/classification. "
        "Use think=True for complex analysis, synthesis, or multi-step reasoning. "
        "Each call runs in a FRESH Python process — no variables/imports carry over between "
        "calls; persist anything you need to a file (use DATA_DIR/PROJECTS_DIR). Use plt.savefig(path), "
        "not plt.show() — there is no inline display. A server started with `&` survives to the next "
        "call (curl it then). "
        "NOTE: Only llm_call() and standard Python are available inside code. "
        "Other agent tools (web_search, project_write_file, schedule_task, etc.) "
        "are NOT available — use them as separate tool calls after code_interpreter returns. "
        "Pre-configured path variables: DATA_DIR, PROJECTS_DIR. "
        "Use these for file access, e.g.: open(f'{PROJECTS_DIR}/flatsixai/research/analysis.md'). "
        "NEVER use relative paths or hardcode 'data/...' — always use the path variables. "
        "CRITICAL RULES: "
        '1) IMPORTANT: You must escape all inner double quotes (e.g., \\") or use single quotes for strings within the code to ensure valid JSON output. '
        "2) Write intermediate data to files (use DATA_DIR=os.getenv('DATA_DIR','data')), never to stdout. "
        "3) For multiple URLs: ONE script, for-loop, save raw content to .jsonl file, process from file with llm_call(). "
        "4) print() ONLY a 3-5 line summary. Save full results to a file. "
        "5) NEVER print raw HTML/page content. NEVER make separate calls per URL."
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
