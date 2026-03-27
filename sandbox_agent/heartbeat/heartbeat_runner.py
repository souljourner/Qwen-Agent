"""Heartbeat runner — OpenClaw-inspired isolated-session heartbeat with HEARTBEAT_OK suppression."""

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from qwen_agent.llm.schema import Message

from sandbox_agent.config import DATA_DIR, HEARTBEAT_INTERVAL_SECONDS
from sandbox_agent.scheduler.task_queue import TaskQueue

logger = logging.getLogger(__name__)

HEARTBEAT_OK = "HEARTBEAT_OK"
HEARTBEAT_OK_MAX_LENGTH = 300

HEARTBEAT_SYSTEM_SUFFIX = (
    "Read the HEARTBEAT checklist below. Follow it strictly. "
    "Do not infer or repeat old tasks. "
    "If nothing needs attention, reply only with HEARTBEAT_OK."
)


def _load_heartbeat_md(data_dir: Optional[str] = None) -> str:
    """Load HEARTBEAT.md from data dir, falling back to the bundled default."""
    data_dir = data_dir or DATA_DIR
    custom = os.path.join(data_dir, "HEARTBEAT.md")
    if os.path.exists(custom):
        return Path(custom).read_text()
    # Fall back to the bundled default
    bundled = Path(__file__).parent / "HEARTBEAT.md"
    if bundled.exists():
        return bundled.read_text()
    return "# Heartbeat Checklist\n- [ ] Check for due scheduled tasks"


def _parse_pending_items(md_text: str) -> List[str]:
    """Extract unchecked items (- [ ] ...) from markdown checklist."""
    return re.findall(r"^- \[ \] (.+)$", md_text, re.MULTILINE)


def _is_heartbeat_ok(response_text: str) -> bool:
    """Check if response is a HEARTBEAT_OK (nothing needs attention)."""
    return HEARTBEAT_OK in response_text and len(response_text.strip()) <= HEARTBEAT_OK_MAX_LENGTH


def _extract_response_text(response: List[Message]) -> str:
    """Extract text content from agent response messages."""
    parts = []
    for msg in response:
        if msg.role == "assistant":
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if hasattr(item, "text") and item.text:
                        parts.append(item.text)
    return "\n".join(parts)


class HeartbeatRunner:
    """Periodically runs heartbeat checks in isolated agent sessions.

    Following the OpenClaw pattern:
    - Each heartbeat creates a fresh agent session (isolated from main conversation)
    - Reads HEARTBEAT.md checklist + due tasks from TaskQueue
    - If nothing needs attention: HEARTBEAT_OK (suppressed, silent)
    - If action needed: returns notification text
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        agent_factory: Optional[Callable] = None,
        runner: Optional[Callable[[List[Message]], List[Message]]] = None,
        data_dir: Optional[str] = None,
        interval: Optional[int] = None,
        on_alert: Optional[Callable[[str], None]] = None,
        work_lock: Optional["threading.Lock"] = None,
    ):
        """
        Args:
            task_queue: Shared TaskQueue for checking due tasks.
            agent_factory: Callable that returns a fresh Agent instance (isolated session).
                Used if `runner` is not provided.
            runner: Callable(messages) -> response. Runs messages on the best available model.
                Takes precedence over agent_factory if both are provided.
            data_dir: Directory containing HEARTBEAT.md.
            interval: Heartbeat interval in seconds (default from config).
            on_alert: Callback for when the heartbeat finds something that needs attention.
            work_lock: Optional lock shared with cron loop to prevent concurrent background work.
        """
        self.agent_factory = agent_factory
        self.runner = runner
        self.task_queue = task_queue
        self.data_dir = data_dir or DATA_DIR
        self.interval = interval or HEARTBEAT_INTERVAL_SECONDS
        self.on_alert = on_alert or (lambda msg: logger.info(f"HEARTBEAT ALERT: {msg}"))
        self.work_lock = work_lock

    def run_once(self) -> Optional[str]:
        """Run a single heartbeat check. Returns alert text or None if OK."""
        # Load checklist
        md_text = _load_heartbeat_md(self.data_dir)
        pending_items = _parse_pending_items(md_text)

        # Check for due tasks
        due_tasks = self.task_queue.get_due_tasks()

        # Build the heartbeat user message
        parts = ["## Heartbeat Check\n"]
        if pending_items:
            parts.append("### Checklist Items")
            for item in pending_items:
                parts.append(f"- {item}")
            parts.append("")

        if due_tasks:
            parts.append("### Due Scheduled Tasks")
            for task in due_tasks:
                parts.append(f"- [{task.id}] {task.name}: {task.description}")
                if task.checkpoint:
                    parts.append(f"  (checkpoint at step {task.current_step})")
            parts.append("")

        if not pending_items and not due_tasks:
            # Nothing to check — skip the LLM call entirely
            logger.debug("HEARTBEAT_OK (no items to check)")
            return None

        user_message = "\n".join(parts)

        messages = [Message(role="user", content=user_message)]

        # Run via runner (lock-aware) or agent_factory (fixed model)
        if self.runner:
            response = self.runner(messages)
        elif self.agent_factory:
            agent = self.agent_factory()
            response = []
            for response in agent.run(messages=messages):
                pass
        else:
            logger.error("HeartbeatRunner has no runner or agent_factory")
            return None

        if not response:
            logger.debug("HEARTBEAT_OK (empty response)")
            return None

        response_text = _extract_response_text(response)

        # HEARTBEAT_OK suppression (OpenClaw pattern)
        if _is_heartbeat_ok(response_text):
            logger.debug("HEARTBEAT_OK (agent confirmed nothing needs attention)")
            return None

        # Something needs attention — notify
        return response_text

    def loop(self) -> None:
        """Run heartbeat checks forever at the configured interval."""
        logger.info(f"Heartbeat loop started (interval: {self.interval}s)")
        # Wait before the first heartbeat to let services come up
        time.sleep(min(self.interval, 60))
        while True:
            try:
                if self.work_lock:
                    logger.debug("Heartbeat: waiting for background work lock")
                    with self.work_lock:
                        alert = self.run_once()
                else:
                    alert = self.run_once()
                if alert:
                    self.on_alert(alert)
            except Exception:
                logger.warning("Heartbeat check failed (will retry next interval)", exc_info=False)
            time.sleep(self.interval)
