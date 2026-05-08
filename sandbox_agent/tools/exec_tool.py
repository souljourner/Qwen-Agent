"""Shell execution tool — runs arbitrary commands inside the container.

Provides the agent with general shell access for build tools, package managers,
file operations, and non-Python CLIs. Commands run via subprocess with shell=True.
"""

import logging
import os
import re
import subprocess
import time
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR
from sandbox_agent.token_budget import truncate_output

logger = logging.getLogger(__name__)

MAX_TIMEOUT = 600  # 10 minutes
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
        "Use for: npm/pip install, build tools, git, file operations, starting servers, "
        "running tests, and any CLI tool. "
        "For Python data work with persistent state and llm_call(), prefer code_interpreter. "
        "For building apps: use exec for installs and builds, project_write_file for source code. "
        "Commands run in DATA_DIR by default, or in a project directory if 'project' is set."
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
                "description": "Timeout in seconds (default 120, max 600).",
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
        if params.get("project"):
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

        # Execute
        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            duration = time.monotonic() - start
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            output = output.strip() or "(no output)"
            output = truncate_output(output, MAX_OUTPUT_TOKENS)

            header = f"Exit {result.returncode} | {duration:.1f}s | {len(output):,} chars"
            return f"{header}\n\n{output}"

        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start
            partial = ""
            if e.stdout:
                partial += e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace")
            if e.stderr:
                partial += e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace")
            partial = truncate_output(partial.strip(), MAX_OUTPUT_TOKENS) if partial.strip() else "(no output before timeout)"

            header = f"Timed out after {timeout}s | {len(partial):,} chars"
            return f"{header}\n\n{partial}"

        except Exception as e:
            return f"Error: {e}"
        finally:
            set_current_tool(None)
