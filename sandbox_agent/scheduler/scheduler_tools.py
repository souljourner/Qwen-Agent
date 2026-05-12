"""Registered tools that allow the agent to self-schedule tasks."""

import json
from datetime import datetime
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.scheduler.checkpoint import load_checkpoint, save_checkpoint
from sandbox_agent.scheduler.task_queue import TaskQueue

# Shared task queue instance (initialized lazily)
_task_queue = None


def get_task_queue() -> TaskQueue:
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


def set_task_queue(tq: TaskQueue) -> None:
    """Allow injection of a custom TaskQueue (for testing or custom data dirs)."""
    global _task_queue
    _task_queue = tq


@register_tool("schedule_task")
class ScheduleTask(BaseTool):
    """Schedule a new task for future execution."""

    name = "schedule_task"
    description = (
        "Schedule a background task for execution. The task will run in a separate agent session "
        "with full access to all tools (web_search, code_interpreter, llm_call, etc). "
        "The description field is the prompt that the background agent receives — write it as "
        "detailed step-by-step instructions for what the agent should do."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short name for the task.",
            },
            "description": {
                "type": "string",
                "description": (
                    "Detailed step-by-step instructions for the background agent. "
                    "This is the prompt the agent will receive. Be specific about which tools to use "
                    "(e.g., 'Use code_interpreter to fetch URLs with requests.get(), then use llm_call() "
                    "to analyze each page'). The background agent has all the same tools you do. "
                    'IMPORTANT: You must escape all inner double quotes (e.g., \\") or use single quotes for strings within the code to ensure valid JSON output.'
                ),
            },
            "schedule_type": {
                "type": "string",
                "enum": ["at", "every", "cron"],
                "description": "Schedule type: 'at' for one-shot (use run_at='now' for immediate), 'every' for interval, 'cron' for cron expression.",
            },
            "cron": {
                "type": "string",
                "description": "Cron expression (e.g., '0 */1 * * *' for every hour). Required if schedule_type is 'cron'.",
            },
            "interval_seconds": {
                "type": "integer",
                "description": "Interval in seconds for 'every' schedule type.",
            },
            "run_at": {
                "type": "string",
                "description": "When to run: 'now' for immediate, or ISO 8601 datetime (e.g., '2024-01-15T10:00:00'). Only for schedule_type 'at'.",
            },
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of task IDs that must complete before this task can run.",
            },
            "priority": {
                "type": "integer",
                "description": "Priority (higher = more urgent). Default 0.",
            },
            "project": {
                "type": "string",
                "description": "Project name to scope this task to (optional). Task results can reference project files.",
            },
        },
        "required": ["name", "description", "schedule_type"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        tq = get_task_queue()

        run_at = None
        if params.get("run_at"):
            raw = params["run_at"]
            if raw.lower() in ("now", "immediately"):
                run_at = datetime.now()
            else:
                run_at = datetime.fromisoformat(raw)

        task = tq.add_task(
            name=params["name"],
            description=params["description"],
            schedule_type=params["schedule_type"],
            cron=params.get("cron"),
            interval_seconds=params.get("interval_seconds"),
            run_at=run_at,
            depends_on=params.get("depends_on"),
            priority=params.get("priority", 0),
            project=params.get("project"),
        )
        return json.dumps({
            "status": "scheduled",
            "task_id": task.id,
            "name": task.name,
            "next_run": str(task.next_run),
        }, ensure_ascii=False)


@register_tool("list_tasks")
class ListTasks(BaseTool):
    """List tasks in the queue."""

    name = "list_tasks"
    description = (
        "List tasks. Default shows current tasks (pending, running, paused, failed). "
        "Use category='completed' or 'cancelled' to see archived tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["current", "completed", "cancelled"],
                "description": "Which tasks to list: 'current' (default — active/paused/failed), 'completed', or 'cancelled'.",
            },
            "project": {
                "type": "string",
                "description": "Filter by project name.",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        tq = get_task_queue()
        tasks = tq.list_tasks(category=params.get("category"), project=params.get("project"))
        return json.dumps([{
            "id": t.id,
            "name": t.name,
            "project": t.project,
            "status": t.status,
            "schedule_type": t.schedule_type,
            "next_run": str(t.next_run) if t.next_run else None,
            "priority": t.priority,
            "current_step": t.current_step,
            "total_steps": t.total_steps,
        } for t in tasks], ensure_ascii=False, indent=2)


@register_tool("complete_task")
class CompleteTask(BaseTool):
    """Mark a task as completed."""

    name = "complete_task"
    description = "Mark a task as completed with an optional result summary."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to mark as completed.",
            },
            "result": {
                "type": "string",
                "description": "Optional result summary.",
            },
        },
        "required": ["task_id"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        tq = get_task_queue()
        task = tq.update_task(
            task_id=params["task_id"],
            status="completed",
            result=params.get("result"),
        )
        if not task:
            return json.dumps({"error": f"Task {params['task_id']} not found"})
        return json.dumps({
            "status": task.status,
            "task_id": task.id,
            "name": task.name,
            "next_run": str(task.next_run) if task.next_run else None,
        }, ensure_ascii=False)


@register_tool("update_task_checkpoint")
class UpdateTaskCheckpoint(BaseTool):
    """Save progress for a long-running task."""

    name = "update_task_checkpoint"
    description = (
        "Save a checkpoint for a long-running task so it can resume after interruption. "
        "Include the current step number and any state needed to continue."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to checkpoint.",
            },
            "step": {
                "type": "integer",
                "description": "Current step number (0-indexed).",
            },
            "checkpoint": {
                "type": "object",
                "description": "Arbitrary state dict to persist.",
            },
        },
        "required": ["task_id", "step"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        task_id = params["task_id"]
        step = params["step"]
        state = params.get("checkpoint", {})

        tq = get_task_queue()
        task = tq.update_task(
            task_id=task_id,
            current_step=step,
            checkpoint=state,
        )
        if not task:
            return json.dumps({"error": f"Task {task_id} not found"})

        # Also persist to disk for crash recovery
        save_checkpoint(task_id, step, state)

        return json.dumps({
            "status": "checkpointed",
            "task_id": task_id,
            "step": step,
        }, ensure_ascii=False)


@register_tool("cancel_task")
class CancelTask(BaseTool):
    """Cancel a task permanently — removes it from the queue."""

    name = "cancel_task"
    description = (
        "Cancel and remove a task from the queue. Works for both one-shot and recurring tasks. "
        "Use this to stop recurring cron/interval tasks that are no longer needed. "
        "If the task is currently running, it is interrupted: the agent loop stops at its next step "
        "and any exec/code_interpreter subprocess it's wedged in is killed (response includes killed_running)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to cancel.",
            },
        },
        "required": ["task_id"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        tq = get_task_queue()
        task = tq.get_task(params["task_id"])
        if not task:
            return json.dumps({"error": f"Task {params['task_id']} not found"})
        name = task.name
        tq.remove_task(params["task_id"])
        # If this task is currently executing, interrupt it: set its cancel
        # flag (the agent loop raises RunCancelled at the next step) and SIGKILL
        # any exec/code_interpreter subprocess it's wedged in.
        killed_running = False
        try:
            from sandbox_agent.cancellation import cancel as _cancel_run
            killed_running = _cancel_run(params["task_id"])
        except Exception:  # noqa: BLE001
            pass
        return json.dumps({
            "status": "cancelled",
            "task_id": params["task_id"],
            "name": name,
            "killed_running": killed_running,
        })


@register_tool("pause_task")
class PauseTask(BaseTool):
    """Pause a task — it stays in the queue but won't run until resumed."""

    name = "pause_task"
    description = (
        "Pause a task. It stays in the queue but won't be picked up by the cron loop. "
        "Use resume_task to unpause it. Works for both one-shot and recurring tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to pause.",
            },
        },
        "required": ["task_id"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        tq = get_task_queue()
        task = tq.update_task(params["task_id"], status="paused")
        if not task:
            return json.dumps({"error": f"Task {params['task_id']} not found"})
        return json.dumps({"status": "paused", "task_id": task.id, "name": task.name})


@register_tool("resume_task")
class ResumeTask(BaseTool):
    """Resume a paused task."""

    name = "resume_task"
    description = "Resume a paused task so it will run again on its next schedule."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to resume.",
            },
        },
        "required": ["task_id"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        tq = get_task_queue()
        task = tq.update_task(params["task_id"], status="pending")
        if not task:
            return json.dumps({"error": f"Task {params['task_id']} not found"})
        return json.dumps({"status": "pending", "task_id": task.id, "name": task.name, "next_run": str(task.next_run)})
