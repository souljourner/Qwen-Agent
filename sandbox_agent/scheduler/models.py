"""Pydantic models for the task scheduling system."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A scheduled task with support for one-shot, interval, and cron scheduling."""

    id: str
    name: str
    description: str

    # Schedule type (OpenClaw-style)
    schedule_type: Literal["at", "every", "cron"] = "at"
    cron: Optional[str] = None             # cron expression (e.g., "*/30 * * * *")
    # IANA zone the cron is written in — cron fields are WALL-CLOCK time in
    # this zone, so "30 12 * * 5" + America/Los_Angeles is 12:30pm Pacific
    # year-round (DST handled automatically; no UTC conversion, no drift).
    timezone: str = "America/Los_Angeles"
    interval_seconds: Optional[int] = None  # for "every" type
    run_at: Optional[datetime] = None       # for "at" type (one-shot)
    next_run: Optional[datetime] = None     # computed next execution time

    # Project scope (optional — None means global task)
    project: Optional[str] = None

    # Dependencies
    depends_on: List[str] = Field(default_factory=list)

    # Status
    status: Literal["pending", "running", "completed", "failed", "paused", "cancelled"] = "pending"
    result: Optional[str] = None
    priority: int = 0  # higher = more urgent

    # Long-running task support
    total_steps: Optional[int] = None
    current_step: int = 0
    checkpoint: Optional[Dict[str, Any]] = None

    # Retry with exponential backoff (OpenClaw pattern)
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None

    # Bumped by the stuck-detector when it abandons a worker it cannot kill.
    # Workers capture this at start and pass it to
    # update_task(expected_generation=...); a stale generation is ignored, so
    # an abandoned-but-alive worker can't double-complete the task later.
    run_generation: int = 0

    # Chat origin that scheduled this task — `{session_id, thread_id}` of the
    # Chainlit session whose agent run created it. None for tasks created
    # outside a chat (cron-spawned subtasks without parent origin, heartbeat,
    # REPL). The notifier uses this to route the completion notice + synthetic
    # follow-up agent turn only to the originating session, instead of
    # broadcasting to every connected tab.
    origin: Optional[Dict[str, Any]] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
