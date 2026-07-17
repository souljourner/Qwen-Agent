"""Auto-commit changes to the agent's data directory git repo."""

import logging
import os
import subprocess
import time
from pathlib import Path

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)


def _run_git(*args: str, cwd: str = None) -> str:
    """Run a git command in DATA_DIR. Returns stdout or empty string on failure.
    cwd defaults to DATA_DIR at CALL time (frozen defaults break test patching)."""
    cwd = cwd or DATA_DIR
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


_STALE_LOCK_SECONDS = 600


def _break_stale_lock(cwd: str = None) -> None:
    """Remove .git/index.lock when it's older than _STALE_LOCK_SECONDS — a
    crashed git process left one for 2 days (2026-07-15→17) and every
    autocommit failed until manual cleanup. Fresh locks (live git process)
    are never touched."""
    lock = Path(cwd or DATA_DIR) / ".git" / "index.lock"
    try:
        if lock.exists() and (time.time() - lock.stat().st_mtime) > _STALE_LOCK_SECONDS:
            lock.unlink()
            logger.warning("Removed stale git index.lock (older than 10 min)")
    except OSError:
        pass


def autocommit(filename: str, message: str) -> None:
    """Stage and commit a file change in the DATA_DIR git repo."""
    _break_stale_lock()
    # Block path traversal but allow relative subdirectory paths (e.g., projects/name/file.md)
    normalized = os.path.normpath(filename)
    if ".." in normalized or normalized.startswith("/") or normalized.startswith("\\"):
        logger.warning(f"Rejected autocommit for invalid filename: {filename}")
        return
    ensure_git_repo()
    _run_git("add", normalized)
    # Check if there are staged changes
    status = _run_git("diff", "--cached", "--name-only")
    if not status:
        logger.debug(f"No changes to commit for {filename}")
        return
    _run_git("commit", "-m", message)
    logger.info(f"Auto-committed: {message}")
