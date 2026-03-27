"""JSON-file-backed task queue with cron, interval, dependencies, and exponential backoff."""

import json
import os
import threading
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from croniter import croniter

from sandbox_agent.config import DATA_DIR
from sandbox_agent.scheduler.models import Task

# Exponential backoff delays for failed tasks (OpenClaw pattern)
BACKOFF_DELAYS = [30, 60, 300, 900, 3600]  # 30s, 1m, 5m, 15m, 60m


class TaskQueue:
    """Thread-safe, JSON-file-backed task queue with scheduling support."""

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = data_dir or DATA_DIR
        os.makedirs(self._data_dir, exist_ok=True)
        self._file_path = os.path.join(self._data_dir, "tasks.json")
        self._lock = threading.Lock()
        self._tasks: List[Task] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._file_path):
            with open(self._file_path, "r") as f:
                raw = json.load(f)
            self._tasks = [Task(**t) for t in raw]
        else:
            self._tasks = []

    def _save(self) -> None:
        with open(self._file_path, "w") as f:
            json.dump([t.model_dump(mode="json") for t in self._tasks], f, ensure_ascii=False, indent=2, default=str)
        try:
            from sandbox_agent.tools.git_autocommit import autocommit
            autocommit("tasks.json", "Update task queue")
        except Exception:
            pass  # Git commit is best-effort

    def add_task(
        self,
        name: str,
        description: str,
        schedule_type: str = "at",
        cron: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        run_at: Optional[datetime] = None,
        depends_on: Optional[List[str]] = None,
        priority: int = 0,
        project: Optional[str] = None,
    ) -> Task:
        """Create and enqueue a new task."""
        task_id = uuid.uuid4().hex[:12]
        now = datetime.now()

        # Compute next_run based on schedule type
        if schedule_type == "at":
            next_run = run_at or now
        elif schedule_type == "every":
            next_run = now + timedelta(seconds=interval_seconds or 60)
        elif schedule_type == "cron":
            if not cron:
                raise ValueError("cron expression required for schedule_type='cron'")
            next_run = croniter(cron, now).get_next(datetime)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        task = Task(
            id=task_id,
            name=name,
            description=description,
            schedule_type=schedule_type,
            cron=cron,
            interval_seconds=interval_seconds,
            run_at=run_at,
            next_run=next_run,
            depends_on=depends_on or [],
            priority=priority,
            project=project,
            created_at=now,
        )

        with self._lock:
            self._tasks.append(task)
            self._save()
        return task

    def get_due_tasks(self) -> List[Task]:
        """Return tasks that are due to run now, respecting dependencies."""
        now = datetime.now()
        completed_ids = {t.id for t in self._tasks if t.status == "completed"}

        due = []
        with self._lock:
            for task in self._tasks:
                if task.status not in ("pending", "failed"):
                    continue
                if task.next_run and task.next_run > now:
                    continue
                # Check dependencies
                if task.depends_on and not all(dep_id in completed_ids for dep_id in task.depends_on):
                    continue
                # Check backoff for failed tasks
                if task.status == "failed" and task.retry_count >= task.max_retries:
                    if task.schedule_type == "at":
                        continue  # One-shot tasks stop after max_retries
                due.append(task)

        # Sort by priority (higher first), then by next_run (earlier first)
        due.sort(key=lambda t: (-t.priority, t.next_run or now))
        return due

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        result: Optional[str] = None,
        last_error: Optional[str] = None,
        current_step: Optional[int] = None,
        checkpoint: Optional[dict] = None,
    ) -> Optional[Task]:
        """Update a task's status and/or result."""
        with self._lock:
            task = self._find_task(task_id)
            if not task:
                return None

            now = datetime.now()
            task.updated_at = now

            if status:
                task.status = status

            if result is not None:
                task.result = result

            if last_error is not None:
                task.last_error = last_error

            if current_step is not None:
                task.current_step = current_step

            if checkpoint is not None:
                task.checkpoint = checkpoint

            # Handle status transitions
            if status == "failed":
                task.retry_count += 1
                # Compute next retry with exponential backoff
                delay_idx = min(task.retry_count - 1, len(BACKOFF_DELAYS) - 1)
                delay = BACKOFF_DELAYS[delay_idx]
                task.next_run = now + timedelta(seconds=delay)
                # Reset status to pending so it can be retried
                if task.retry_count < task.max_retries or task.schedule_type != "at":
                    task.status = "pending"

            elif status == "completed":
                # Compute next run for recurring tasks
                task.next_run = self._compute_next_run(task, now)
                if task.next_run:
                    # Recurring task: reset for next execution
                    task.status = "pending"
                    task.retry_count = 0
                    task.current_step = 0
                    task.checkpoint = None

            self._save()
            return task

    def list_tasks(self, status: Optional[str] = None, project: Optional[str] = None) -> List[Task]:
        """List tasks, optionally filtered by status and/or project."""
        with self._lock:
            tasks = list(self._tasks)
            if status:
                tasks = [t for t in tasks if t.status == status]
            if project:
                tasks = [t for t in tasks if t.project == project]
            return tasks

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a single task by ID."""
        with self._lock:
            return self._find_task(task_id)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue entirely."""
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t.id != task_id]
            if len(self._tasks) < before:
                self._save()
                return True
            return False

    def _find_task(self, task_id: str) -> Optional[Task]:
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    @staticmethod
    def _compute_next_run(task: Task, now: datetime) -> Optional[datetime]:
        """Compute the next run time for recurring tasks. Returns None for completed one-shot tasks."""
        if task.schedule_type == "at":
            return None  # One-shot: done
        elif task.schedule_type == "every":
            return now + timedelta(seconds=task.interval_seconds or 60)
        elif task.schedule_type == "cron":
            return croniter(task.cron, now).get_next(datetime)
        return None
