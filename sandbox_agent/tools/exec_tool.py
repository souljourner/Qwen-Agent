"""Shell execution tool — runs arbitrary commands inside the container.

Provides the agent with general shell access for build tools, package managers,
file operations, and non-Python CLIs. Commands run via subprocess with shell=True.
"""

import logging
import os
import re
import signal
import subprocess
import time
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR, PROJECT_VENV_ENABLED, UV_BIN
from sandbox_agent.token_budget import truncate_output

logger = logging.getLogger(__name__)


def _project_venv_env(base_env: dict, workdir: str) -> dict:
    """If `workdir` has a usable `.venv`, return a COPY of base_env with that
    venv activated — VIRTUAL_ENV set, its bin/ prepended to PATH, PYTHONHOME
    cleared — so `python`, `pip`, and `uv pip` target the project's own
    environment. Returns base_env unchanged if there's no venv."""
    bindir = os.path.join(workdir, ".venv", "bin")
    if not os.path.isdir(bindir):
        return base_env
    env = dict(base_env)
    env["VIRTUAL_ENV"] = os.path.join(workdir, ".venv")
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    return env


def _ensure_project_venv(workdir: str) -> None:
    """Create `workdir/.venv` with `uv venv` if it's missing. Best-effort:
    uv's hardlink cache makes this near-instant, but if uv is unavailable or
    fails we log and return — exec then falls back to the shared global env."""
    if not PROJECT_VENV_ENABLED:
        return
    venv_path = os.path.join(workdir, ".venv")
    if os.path.isdir(os.path.join(venv_path, "bin")):
        return  # already present
    try:
        subprocess.run([UV_BIN, "venv", venv_path], cwd=workdir,
                       capture_output=True, timeout=120, check=False)
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
        logger.warning("Could not create project venv at %s (%s) — using global env", venv_path, e)


def kill_process_group(proc: "subprocess.Popen") -> None:
    """SIGKILL the entire process group of `proc` (which was started with
    start_new_session=True, so it's a group leader). Reaches background
    children a plain proc.kill() would miss. Best-effort — swallows the
    races where the group is already gone."""
    if proc is None or proc.poll() is not None:
        # already exited — still try the group in case `&` children linger
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass

MAX_TIMEOUT = 3600  # 1 hour
DEFAULT_TIMEOUT = 120  # 2 minutes
MAX_OUTPUT_TOKENS = 16000  # Same as code_interpreter

# Commands blocked before execution (defense-in-depth)
BLOCKED_PATTERNS = [
    re.compile(r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/\s*$"),  # rm -rf /
    re.compile(r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+/\s*$"),  # rm -fr /
    re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;"),  # fork bomb
    re.compile(r"dd\s+if=/dev/zero"),  # disk fill
    re.compile(r"\bmkfs\b"),  # format filesystem
    re.compile(r"\bfdisk\b"),  # partition table
    re.compile(r"\bmount\b.*\bdev\b"),  # mount devices
    # Broad process kills — PID 1 in the container is `python` (chainlit), so
    # any of these would take the whole agent down. Process-group isolation
    # (start_new_session) already protects against `kill 0` / `kill -- -PGID`
    # spilling into PID 1's group, but name-targeted kills bypass that.
    re.compile(r"\bpkill\b[^|;&]*\bpython"),
    re.compile(r"\bkillall\b[^|;&]*\bpython"),
    re.compile(r"\bkill\b\s+-?(9|SIGKILL|TERM|SIGTERM)?\s*-1\b"),  # kill -1 / kill -9 -1
    re.compile(r"\bkill\b\s+(-\w+\s+)?\b1\b"),  # kill 1 / kill -9 1
]


def _is_blocked(command: str) -> bool:
    """Check if a command matches any blocked pattern."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return True
    return False


def _validate_workdir(workdir: str) -> str:
    """Validate workdir is under DATA_DIR. Returns resolved path or raises."""
    resolved = os.path.realpath(workdir)
    data_real = os.path.realpath(DATA_DIR)
    if not resolved.startswith(data_real + os.sep) and resolved != data_real:
        raise ValueError(f"Working directory must be under DATA_DIR: {workdir}")
    return resolved


@register_tool("exec")
class ExecTool(BaseTool):
    """Execute shell commands inside the container."""

    name = "exec"
    description = (
        "Run a shell command. Supports pipes, redirects, && chains. "
        "Use for: package installs, build tools, git, file operations, starting servers, "
        "running tests, and any CLI tool. "
        "For Python data work and llm_call(), prefer code_interpreter. "
        "For building apps: use exec for installs and builds, project_write_file for source code. "
        "ALWAYS pass 'project' for project work: it runs in that project's own .venv so "
        "dependencies are isolated from other projects (and pipelines don't clobber each "
        "other). Use uv for packages — `uv pip install <pkg>` and `uv venv` — it's the default "
        "package manager and is cache-backed (near-instant re-installs). The project venv is "
        "auto-created on first use and is already activated, so `python`/`pip` also target it. "
        "Commands run in DATA_DIR by default, or in the project directory if 'project' is set."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute (supports pipes, redirects, && chains).",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory (must be under DATA_DIR). Default: DATA_DIR.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 120, max 3600).",
            },
            "project": {
                "type": "string",
                "description": "Project name — sets workdir to the project directory.",
            },
        },
        "required": ["command"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        command = params["command"]
        timeout = min(int(params.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)

        # Resolve working directory
        is_project = bool(params.get("project"))
        if is_project:
            workdir = os.path.join(DATA_DIR, "projects", params["project"])
            if not os.path.isdir(workdir):
                return f"Error: project '{params['project']}' not found."
        elif params.get("workdir"):
            try:
                workdir = _validate_workdir(params["workdir"])
            except ValueError as e:
                return f"Error: {e}"
        else:
            workdir = DATA_DIR

        # Block dangerous commands
        if _is_blocked(command):
            logger.warning(f"Blocked exec command: {command}")
            from sandbox_agent.activity_log import log_event
            log_event("blocked_exec", detail=command[:300])
            return "Error: command blocked by safety policy."

        # Log execution
        from sandbox_agent.activity_log import log_event
        from sandbox_agent.model_tracker import set_current_tool
        set_current_tool("exec")
        log_event("exec", detail=command[:300])
        logger.info(f"exec: {command[:200]} (workdir={workdir}, timeout={timeout}s)")

        # PYTHONPATH includes /app so subprocesses can import sandbox_agent.*
        # (e.g. `from sandbox_agent.tools.llm_client import llm_call` from a
        # backtest script).
        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = "/app" + (os.pathsep + existing_pp if existing_pp else "")

        # Per-project dependency isolation: project-scoped exec runs inside the
        # project's own .venv (created on first use) so pip installs don't leak
        # across projects. Falls back to the global env if disabled/unavailable.
        if is_project:
            _ensure_project_venv(workdir)
            env = _project_venv_env(env, workdir)

        # Execute. `start_new_session=True` puts the shell (and everything it
        # spawns) in its own process group / session, isolated from PID 1.
        # Without this, an agent-issued command like `kill 0`, `pkill -f
        # python`, or a botched PID-extraction one-liner can SIGTERM the
        # container's main process and take the whole agent down.
        from sandbox_agent.cancellation import register_child_pgid, unregister_child_pgid
        start = time.monotonic()
        proc = None
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=workdir,
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
                # Kill the WHOLE process group — a `cmd &` may have left
                # background children that proc.kill() alone wouldn't reach.
                kill_process_group(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""
                duration = time.monotonic() - start
                partial = ((stdout or "") + ("\n" + stderr if stderr else "")).strip()
                partial = truncate_output(partial, MAX_OUTPUT_TOKENS) if partial else "(no output before timeout)"
                return f"Timed out after {timeout}s | {len(partial):,} chars\n\n{partial}"

            duration = time.monotonic() - start
            output = (stdout or "")
            if stderr:
                output += ("\n" if output else "") + stderr
            output = output.strip() or "(no output)"
            output = truncate_output(output, MAX_OUTPUT_TOKENS)
            return f"Exit {proc.returncode} | {duration:.1f}s | {len(output):,} chars\n\n{output}"

        except Exception as e:
            if proc is not None:
                kill_process_group(proc)
            return f"Error: {e}"
        finally:
            set_current_tool(None)
            if proc is not None:
                unregister_child_pgid(proc.pid)
