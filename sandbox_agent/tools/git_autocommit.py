"""Auto-commit changes to the agent's data directory git repo."""

import logging
import os
import subprocess
from pathlib import Path

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)


def _run_git(*args: str, cwd: str = DATA_DIR) -> str:
    """Run a git command in DATA_DIR. Returns stdout or empty string on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(f"git {' '.join(args)} failed: {result.stderr.strip()}")
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"git command failed: {e}")
        return ""


def ensure_git_repo() -> None:
    """Initialize a git repo in DATA_DIR if one doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    git_dir = Path(DATA_DIR) / ".git"
    if not git_dir.exists():
        _run_git("init")
        _run_git("config", "user.email", "agent@sandbox")
        _run_git("config", "user.name", "Sandbox Agent")
        logger.info(f"Initialized git repo in {DATA_DIR}")


def autocommit(filename: str, message: str) -> None:
    """Stage and commit a file change in the DATA_DIR git repo."""
    # Validate filename is a simple basename (no path traversal)
    basename = os.path.basename(filename)
    if basename != filename or "/" in filename or "\\" in filename or ".." in filename:
        logger.warning(f"Rejected autocommit for invalid filename: {filename}")
        return
    ensure_git_repo()
    _run_git("add", basename)
    # Check if there are staged changes
    status = _run_git("diff", "--cached", "--name-only")
    if not status:
        logger.debug(f"No changes to commit for {filename}")
        return
    _run_git("commit", "-m", message)
    logger.info(f"Auto-committed: {message}")
