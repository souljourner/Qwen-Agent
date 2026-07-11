"""Local code interpreter — runs Python as a fresh subprocess per call.

Each `code_interpreter` call writes the user's code (after a fixed prelude that
imports numpy/pandas, sets DATA_DIR/PROJECTS_DIR, and — if the LLM bridge is up —
defines llm_call()) to a temp .py file and runs it as a fresh `python` subprocess
in its own session, draining its output line-by-line so progress is visible while
it runs (no streaming = the multi-minute runs that looked frozen). Killed via the
whole process group on timeout/error. No persistent interpreter: state does NOT
carry over between calls — persist anything you need to a file. This mirrors
`exec_tool` and matches how OpenClaw/Hermes run code; it eliminates the
wedged-Jupyter-kernel hang class. Persistence is the filesystem
(DATA_DIR/PROJECTS_DIR) + the container's installed packages.
"""

import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from typing import Callable, Optional, Union

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

# Thread-ident → callable(preview_text). chat_app registers one for the worker
# thread running an on_message turn so a long code_interpreter call's stdout
# streams to that turn's Chainlit tool step instead of sitting on a silent
# spinner. No-op for cron/background runs (they don't register one).
_progress_hooks: "dict[int, Callable[[str], None]]" = {}

_PROGRESS_INTERVAL = 0.5   # seconds between live progress emits
_PROGRESS_TAIL = 4000      # chars of recent output to surface per emit


def register_progress_hook(fn: Callable[[str], None]) -> None:
    _progress_hooks[threading.get_ident()] = fn


def unregister_progress_hook() -> None:
    _progress_hooks.pop(threading.get_ident(), None)


def _escape_ansi(text: str) -> str:
    return re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]").sub("", text)


MIN_TIMEOUT = 600        # 10 minutes minimum
SECONDS_PER_TOKEN = 0.1  # ~10 tokens/sec generation speed as budget estimate
MAX_TIMEOUT = 3600       # 1 hour hard cap (a longer job should be a scheduled task)


def _compute_timeout(code: str) -> int:
    """Compute timeout based on code size. Longer code = more work = more time.

    Heuristic: 1 token ≈ 4 chars, budget ~0.1s per token of code.
    Minimum 10 minutes, max 1 hour.
    """
    estimated_tokens = len(code) // 4
    computed = int(estimated_tokens * SECONDS_PER_TOKEN)
    return max(MIN_TIMEOUT, min(computed, MAX_TIMEOUT))


def _build_script(code: str) -> str:
    """Assemble the full script: fixed prelude + llm_call() def + user code."""
    return INIT_CODE + "\n" + (_llm_init_code or "") + "\n\n# --- user code ---\n" + code + "\n"


def _emit_progress(hook: Optional[Callable[[str], None]], text: str, last_line: str = "") -> None:
    """Push a live-progress snapshot: the dashboard preview (best-effort) and,
    if a hook is registered for this run, the recent tail of output."""
    try:
        from sandbox_agent.model_tracker import set_current_preview
        snippet = (last_line or text).strip().splitlines()[-1] if (last_line or text).strip() else ""
        set_current_preview(f"code_interpreter · {snippet[:200]}")
    except Exception:  # noqa: BLE001
        pass
    if hook is None:
        return
    preview = text if len(text) <= _PROGRESS_TAIL else "…(earlier output truncated)…\n" + text[-_PROGRESS_TAIL:]
    try:
        hook(preview)
    except Exception:  # noqa: BLE001
        pass


def _execute_code(code: str, timeout: int = 0) -> str:
    """Run `code` as a fresh Python subprocess, streaming its output; return
    exec-style output (`Exit N | Ns | M chars\\n\\n<output>` or `Timed out …`)."""
    if timeout <= 0:
        timeout = _compute_timeout(code)

    os.makedirs(WORK_DIR, exist_ok=True)
    script_path = os.path.join(WORK_DIR, f"_ci_{uuid.uuid4().hex[:8]}.py")
    with open(script_path, "w") as f:
        f.write(_build_script(code))

    # /app on PYTHONPATH so the script can `from sandbox_agent.tools.llm_client
    # import llm_batch` (the editable install usually covers this already, but
    # be explicit — same as exec_tool). PYTHONUNBUFFERED so the child flushes
    # each line immediately — otherwise stdout-to-a-pipe is block-buffered and
    # we'd only see output at process exit (defeating the streaming).
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "/app" + (os.pathsep + existing_pp if existing_pp else "")
    env["PYTHONUNBUFFERED"] = "1"

    from sandbox_agent.cancellation import register_child_pgid, unregister_child_pgid
    hook = _progress_hooks.get(threading.get_ident())  # capture on the worker thread
    start = time.monotonic()
    proc = None
    lines: "list[str]" = []
    try:
        # `start_new_session=True`: the script (and anything it spawns) gets its
        # own process group / session, isolated from PID 1. A `kill 0` / `pkill`
        # inside the script can't reach the container's main process.
        # stderr→stdout: one stream, output in the order the script wrote it.
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=DATA_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered reads
            env=env,
            start_new_session=True,
        )
        # Register the child's process group so cancel_task / chat Stop can
        # SIGKILL it (start_new_session=True → pgid == pid).
        register_child_pgid(proc.pid)

        # Drain stdout line-by-line in a helper thread (also stops the pipe
        # buffer from filling and deadlocking the child on a chatty script).
        last_emit = [0.0]

        def _reader():
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    lines.append(line)
                    now = time.monotonic()
                    if len(lines) == 1 or now - last_emit[0] >= _PROGRESS_INTERVAL:
                        last_emit[0] = now
                        _emit_progress(hook, "".join(lines), last_line=line)
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    proc.stdout.close()  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    pass

        reader = threading.Thread(target=_reader, name="ci-reader", daemon=True)
        reader.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_group(proc)  # SIGKILL the whole group — incl. any `server &` children
            timed_out = True
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        reader.join(timeout=5)  # let it flush any buffered lines

        duration = time.monotonic() - start
        output = _escape_ansi("".join(lines).strip()) or "(no output)"
        output = truncate_output(output, MAX_CODE_OUTPUT_TOKENS)
        _emit_progress(hook, output)  # final snapshot
        # NOTE: on a clean exit we deliberately do NOT kill the process group —
        # a server the script started with `&` is reparented to init and keeps
        # listening, so the agent can curl it from the next call (matches exec).
        if timed_out:
            return f"Timed out after {timeout}s | {len(output):,} chars\n\n{output}"
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
        "call (curl it then). For long-running scripts, print() progress lines — they stream as the "
        "script runs (stdout is unbuffered). "
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
        "4) print() ONLY a 3-5 line summary at the end. Save full results to a file. "
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
        # Email policy: no hand-rolled SMTP from interpreter code.
        from sandbox_agent.email_policy import BLOCKED_EMAIL_MSG, contains_email_bypass, log_blocked
        if contains_email_bypass(code):
            log_blocked("code_interpreter", code)
            return f"Error: {BLOCKED_EMAIL_MSG} Use the send_email tool instead."
        return _execute_code(code)
