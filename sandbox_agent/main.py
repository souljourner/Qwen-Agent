"""Main entry point for the sandboxed Qwen Agent.

Lane-based concurrency (OpenClaw pattern):
- Main lane: interactive UI (Gradio WebUI or REPL) using primary vLLM model
- Heartbeat lane: periodic checks using background Ollama model
- Cron lane: scheduled task execution using background Ollama model
"""

import json
import logging
import os
import sys
import time
from threading import Lock, Thread
from typing import Iterator, List, Optional

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message

from sandbox_agent.config import (
    BACKGROUND_LLM_CFG,
    DATA_DIR,
    HEARTBEAT_INTERVAL_SECONDS,
    PRIMARY_LLM_CFG,
    TOOL_LIST,
    load_system_message,
    session_metadata,
)
from sandbox_agent.activity_log import clear_state, get_recent_events, log_event, set_state
from sandbox_agent.chat_logger import log_background_task, log_turn
from sandbox_agent.daily_digest import add_digest_entry, cleanup_old_digests
from sandbox_agent.heartbeat.heartbeat_runner import HeartbeatRunner
from sandbox_agent.scheduler.task_queue import TaskQueue
from sandbox_agent.token_budget import compute_request_timeout, estimate_messages_tokens, trim_to_budget

# Import tools to trigger @register_tool decorators
import sandbox_agent.tools.api_tools  # noqa: F401
import sandbox_agent.tools.self_edit_tools  # noqa: F401
import sandbox_agent.tools.code_interpreter  # noqa: F401
import sandbox_agent.chat_logger  # noqa: F401 (registers list_chat_logs, read_chat_log)
import sandbox_agent.tools.project_tools  # noqa: F401
import sandbox_agent.tools.notification_tools  # noqa: F401
import sandbox_agent.pipeline.pipeline_tools  # noqa: F401
import sandbox_agent.scheduler.scheduler_tools  # noqa: F401

logger = logging.getLogger(__name__)

# Lock for the primary vLLM model — held while the user is chatting.
# Background tasks check this to decide: vLLM (idle) or Ollama (busy).
_primary_model_lock = Lock()

# Lock for background work — ensures heartbeat and cron tasks run one at a time, never colliding.
_background_work_lock = Lock()


def create_agent(system_message: str, llm_cfg: dict, name: str = "SandboxAgent") -> Assistant:
    """Create a fresh Agent instance. Used as factory for isolated sessions."""
    agent = Assistant(
        llm=llm_cfg,
        function_list=TOOL_LIST,
        system_message=system_message,
        name=name,
        description="A research and task management assistant with web search, scheduling, and self-editing capabilities.",
    )

    # Wrap _call_tool to log every tool invocation
    original_call_tool = agent._call_tool

    def _logged_call_tool(tool_name, tool_args='{}', **kwargs):
        args_preview = str(tool_args)[:200]
        logger.info(f"Tool call: {tool_name}({args_preview})")
        set_state(current_tool=tool_name)
        log_event("tool_call", tool_name=tool_name, tool_args=args_preview)
        result = original_call_tool(tool_name, tool_args, **kwargs)
        result_preview = str(result)[:200]
        logger.info(f"Tool result: {tool_name} -> {result_preview}")
        log_event("tool_result", tool_name=tool_name, tool_result=result_preview)
        set_state(current_tool=None)
        return result

    agent._call_tool = _logged_call_tool
    return agent


class LockingAgent(Assistant):
    """Wraps an Assistant with smart model routing.

    Tries the primary vLLM model first. If a background task is currently
    using vLLM (lock is held), routes to the Ollama backup agent instead
    so the user gets an immediate response.
    """

    def __init__(self, inner: Assistant, backup: Assistant, lock: Lock):
        self._inner = inner      # Primary agent (vLLM)
        self._backup = backup    # Backup agent (Ollama)
        self._lock = lock

    def run(self, *args, **kwargs) -> Iterator[List[Message]]:
        messages = args[0] if args else kwargs.get("messages", [])
        user_msg = None
        if messages:
            msg_list = [m if isinstance(m, Message) else Message(**m) for m in messages]
            timeout = compute_request_timeout(msg_list)
            self._inner.llm.generate_cfg["request_timeout"] = timeout
            self._backup.llm.generate_cfg["request_timeout"] = timeout

            for m in reversed(msg_list):
                if m.role == "user":
                    user_msg = m
                    break

        # Try primary model first, fall back to 27B backup if busy
        got_lock = self._lock.acquire(blocking=False)
        if got_lock:
            agent = self._inner
            model_name = PRIMARY_LLM_CFG["model"]
            log_event("chat_start", detail=f"[primary] {str(user_msg.content)[:180]}" if user_msg else "[primary]")
        else:
            # Primary busy with background task — use 27B backup (reserved for user)
            agent = self._backup
            model_name = BACKGROUND_LLM_CFG["model"]
            log_event("chat_start", detail=f"[27b fallback] {str(user_msg.content)[:160]}" if user_msg else "[27b fallback]")
            logger.info(f"Chat routed to {model_name} (primary busy with background task)")

        set_state(status="chatting", model_in_use=model_name)
        try:
            response = []
            for response in agent.run(*args, **kwargs):
                yield response
        finally:
            if got_lock:
                self._lock.release()

        log_event("chat_complete")
        clear_state()

        if user_msg and response:
            try:
                log_turn(user_msg, response)
            except Exception:
                pass

    def run_nonstream(self, *args, **kwargs):
        got_lock = self._lock.acquire(blocking=False)
        agent = self._inner if got_lock else self._backup
        try:
            return agent.run_nonstream(*args, **kwargs)
        finally:
            if got_lock:
                self._lock.release()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _summarize_task_result(task_name: str, result_text: str, tool_calls: List[dict], project: str = "") -> str:
    """Use Ollama to write a concise summary of a completed task for the daily digest."""
    try:
        import requests as _req
        ollama_base = BACKGROUND_LLM_CFG["model_server"].replace("/v1", "")
        project_ctx = f" (project: {project})" if project else ""

        # Build a factual record of what happened from tool calls
        searches = []
        files_written = []
        files_read = []
        code_results = []
        scheduled = []
        other = []

        for tc in tool_calls:
            tool = tc.get("tool", "")
            args = tc.get("args", "")
            result = tc.get("result", "")
            if tool == "web_search":
                # Extract query from args
                try:
                    q = json.loads(args).get("query", args)
                except Exception:
                    q = args
                searches.append(q[:80])
            elif tool == "web_url_fetch":
                try:
                    u = json.loads(args).get("url", args)
                except Exception:
                    u = args
                searches.append(f"Fetched: {u[:80]}")
            elif tool == "project_write_file":
                try:
                    parsed = json.loads(args)
                    files_written.append(f"{parsed.get('project','')}/{parsed.get('path','')}")
                except Exception:
                    files_written.append(args[:80])
            elif tool == "project_read_file":
                try:
                    parsed = json.loads(args)
                    files_read.append(f"{parsed.get('project','')}/{parsed.get('path','')}")
                except Exception:
                    files_read.append(args[:80])
            elif tool == "code_interpreter":
                if result:
                    code_results.append(result[:120])
            elif tool == "schedule_task":
                scheduled.append(result[:100])
            elif tool:
                other.append(f"{tool}: {args[:60]}")

        # Build structured summary input
        parts = [f"Task: {task_name}{project_ctx}", ""]
        parts.append(f"Tool calls: {len(tool_calls)} total")
        if searches:
            parts.append(f"Searches ({len(searches)}): {'; '.join(searches[:5])}")
        if files_read:
            parts.append(f"Files read: {', '.join(files_read[:5])}")
        if files_written:
            parts.append(f"Files written: {', '.join(files_written[:5])}")
        if code_results:
            parts.append(f"Code outputs: {' | '.join(code_results[:3])}")
        if scheduled:
            parts.append(f"Scheduled: {'; '.join(scheduled[:3])}")
        if other:
            parts.append(f"Other: {'; '.join(other[:3])}")

        actions_text = "\n".join(parts)

        resp = _req.post(
            f"{ollama_base}/api/chat",
            json={
                "model": BACKGROUND_LLM_CFG["model"],
                "messages": [
                    {"role": "system", "content": (
                        "You are writing a digest entry. Given the facts below, write exactly 2-3 sentences. "
                        "State what was searched, what files were created, and any key findings or numbers. "
                        "Use the FACTS ONLY — do not invent information. Do not say 'the task initiated' or "
                        "'the system is prepared' — say what actually happened. If files were written, name them. "
                        "If searches were done, mention the topics. If nothing substantive happened, say that."
                    )},
                    {"role": "user", "content": actions_text},
                ],
                "think": False,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        summary = resp.json().get("message", {}).get("content", "")
        return summary if summary else result_text[:500]
    except Exception as e:
        logger.warning(f"Failed to summarize task result: {e}")
        return result_text[:500] if result_text else "Completed"


def run_on_best_available(system_message: str, messages: List[Message]) -> List[Message]:
    """Run a background agent session on the primary model.

    Background tasks always use the primary model (qwen3.5).
    The backup model (qwen3.5-27b) is reserved for user chat fallback only.
    Holds the lock for the entire run so the user gets routed to the 27B backup.
    """
    timeout = compute_request_timeout(messages)

    # Blocking acquire — background tasks wait for the primary model
    _primary_model_lock.acquire(blocking=True)
    try:
        logger.info(f"Background task using primary model (timeout={timeout}s)")
        set_state(model_in_use=PRIMARY_LLM_CFG["model"])
        log_event("model_select", detail="primary (background)", model=PRIMARY_LLM_CFG["model"])
        agent = create_agent(system_message, llm_cfg=PRIMARY_LLM_CFG)
        agent.llm.generate_cfg["request_timeout"] = timeout
        response: List[Message] = []
        for response in agent.run(messages=messages):
            pass
        return response
    finally:
        _primary_model_lock.release()


def cron_loop(system_message: str, task_queue: TaskQueue, poll_interval: int = 60) -> None:
    """Background loop that checks for due scheduled tasks and runs them."""
    logger.info(f"Cron loop started (poll interval: {poll_interval}s)")
    while True:
        try:
            due_tasks = task_queue.get_due_tasks()
            for task in due_tasks:
                # Acquire background work lock — prevents collision with heartbeat
                logger.info(f"Cron: waiting for background work lock for [{task.id}] {task.name}")
                with _background_work_lock:
                    logger.info(f"Cron: executing task [{task.id}] {task.name}")
                    set_state(status="cron_task", current_task=f"[{task.id}] {task.name}")
                    log_event("cron_start", task_id=task.id, task_name=task.name)
                    task_queue.update_task(task.id, status="running")
                    # Snapshot event count before execution to capture tool calls
                    events_before = len(get_recent_events(500))
                    try:
                        # Pipeline tasks get special handling
                        if task.name.startswith("pipeline:"):
                            from sandbox_agent.pipeline.stage_runner import run_pipeline_stage
                            result_text = run_pipeline_stage(task, system_message)
                            task_queue.update_task(task.id, status="completed", result=result_text[:1000])
                            log_background_task(task.name, task.id, result_text[:1000])
                            log_event("cron_complete", task_id=task.id, task_name=task.name)
                            add_digest_entry(
                                project=task.project,
                                task_name=task.name,
                                summary=result_text[:500],
                                source="pipeline",
                            )
                            logger.info(f"Cron: pipeline task [{task.id}] completed")
                            clear_state()
                            continue

                        # Regular tasks: build prompt and run
                        prompt_parts = [f"Execute this scheduled task:\n\n**{task.name}**: {task.description}"]
                        if task.project:
                            prompt_parts.append(
                                f"\nThis task belongs to project '{task.project}'. "
                                f"Save any results to the project using project_write_file. "
                                f"You can read existing project files with project_read_file."
                            )
                        if task.checkpoint:
                            prompt_parts.append(
                                f"\nThis task was previously interrupted at step {task.current_step}. "
                                f"Resume from checkpoint: {task.checkpoint}"
                            )
                        messages = [Message(role="user", content="\n".join(prompt_parts))]
                        logger.info(f"Cron: task [{task.id}] prompt: {prompt_parts[0][:200]}...")
                        response = run_on_best_available(system_message, messages)
                        # Extract result text
                        result_text = ""
                        for msg in response:
                            if msg.role == "assistant" and isinstance(msg.content, str):
                                result_text += msg.content
                        logger.info(f"Cron: task [{task.id}] result ({len(result_text)} chars): {result_text[:200]}...")
                        task_queue.update_task(task.id, status="completed", result=result_text[:1000])
                        log_background_task(task.name, task.id, result_text[:1000])
                        log_event("cron_complete", task_id=task.id, task_name=task.name)
                        # Collect tool calls that happened during this task
                        all_events = get_recent_events(500)
                        task_tool_calls = [
                            e for e in all_events[events_before:]
                            if e.get("type") == "tool_call"
                        ]
                        # Summarize using tool call record + result text
                        digest_summary = _summarize_task_result(task.name, result_text, task_tool_calls, task.project or "")
                        add_digest_entry(
                            project=task.project,
                            task_name=task.name,
                            summary=digest_summary,
                            source="cron",
                        )
                        logger.info(f"Cron: task [{task.id}] completed")
                        clear_state()
                    except Exception as e:
                        logger.exception(f"Cron: task [{task.id}] failed")
                        log_event("cron_failed", task_id=task.id, task_name=task.name, detail=str(e)[:300])
                        add_digest_entry(
                            project=task.project,
                            task_name=task.name,
                            summary=f"FAILED: {str(e)[:400]}",
                            source="cron",
                        )
                        task_queue.update_task(task.id, status="failed", last_error=str(e)[:500])
                        clear_state()
        except Exception:
            logger.exception("Cron loop error")
        time.sleep(poll_interval)


def _start_background_lanes(system_message: str, task_queue: TaskQueue) -> None:
    """Start heartbeat and cron background threads."""
    def bg_runner(messages: List[Message]) -> List[Message]:
        return run_on_best_available(system_message, messages)

    heartbeat = HeartbeatRunner(
        task_queue=task_queue,
        runner=bg_runner,
        on_alert=lambda msg: (
            logger.warning(f"HEARTBEAT ALERT: {msg}"),
            add_digest_entry(summary=msg[:500], source="heartbeat"),
        ),
        work_lock=_background_work_lock,
    )
    heartbeat_thread = Thread(target=heartbeat.loop, daemon=True, name="heartbeat")
    heartbeat_thread.start()
    logger.info(f"Heartbeat started (interval: {HEARTBEAT_INTERVAL_SECONDS}s)")

    cron_thread = Thread(target=cron_loop, args=(system_message, task_queue), daemon=True, name="cron")
    cron_thread.start()
    logger.info("Cron loop started")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sandbox Agent")
    parser.add_argument("--mode", choices=["gradio", "repl"], default="gradio",
                        help="UI mode: 'gradio' for web UI (default), 'repl' for terminal")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio server host (default: 0.0.0.0)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Ensure data directory exists and has a git repo
    os.makedirs(DATA_DIR, exist_ok=True)
    from sandbox_agent.tools.git_autocommit import ensure_git_repo
    ensure_git_repo()
    cleanup_old_digests(keep_days=3)

    # Reset any tasks stuck in "running" from a previous crash
    try:
        tq_init = TaskQueue()
        stuck = tq_init.list_tasks(status="running")
        for task in stuck:
            logger.warning(f"Resetting stuck task: [{task.id}] {task.name} (was running at shutdown)")
            tq_init.update_task(task.id, status="pending")
    except Exception:
        pass

    # Clear stale pipeline lock and reset stuck pipeline stages
    from sandbox_agent.pipeline.orchestrator import clear_lock_on_startup
    clear_lock_on_startup()

    # Load system messages
    # Base system message (static, used for background sessions — no metadata for KV cache stability)
    system_message = load_system_message()
    # Main agent gets one-time metadata (date, time, location) appended
    main_system_message = system_message + session_metadata()
    logger.info("System message loaded")

    # Shared task queue
    task_queue = TaskQueue()

    # Inject task queue into scheduler tools
    from sandbox_agent.scheduler.scheduler_tools import set_task_queue
    set_task_queue(task_queue)

    # Start LLM bridge so code_interpreter can make LLM calls to the background model
    from sandbox_agent.tools.llm_bridge import start_bridge, get_kernel_init_code
    from sandbox_agent.config import LLM_CALL_CFG
    import sandbox_agent.tools.code_interpreter as ci
    # llm_call() uses gemma4:26b on Ollama for per-item processing
    bridge_port = start_bridge(LLM_CALL_CFG)
    ci._llm_init_code = get_kernel_init_code(bridge_port)
    logger.info(f"LLM bridge started on port {bridge_port} -> {BACKGROUND_LLM_CFG['model']} @ {BACKGROUND_LLM_CFG['model_server']}")

    # Main agent on primary vLLM model
    # Main agent uses system message WITH metadata (date/time/location)
    # Background sessions use the base system_message WITHOUT metadata (static for KV cache)
    inner_agent = create_agent(main_system_message, llm_cfg=PRIMARY_LLM_CFG)
    logger.info(f"Main agent created (model: {PRIMARY_LLM_CFG['model']})")

    # Start background lanes
    _start_background_lanes(system_message, task_queue)

    # Backup agent on Ollama for when vLLM is busy
    backup_agent = create_agent(main_system_message, llm_cfg=BACKGROUND_LLM_CFG)
    logger.info(f"Backup agent created (model: {BACKGROUND_LLM_CFG['model']})")

    if args.mode == "gradio":
        _run_gradio(inner_agent, backup_agent, args.host, args.port)
    else:
        _run_repl(inner_agent, backup_agent)


def _run_gradio(inner_agent: Assistant, backup_agent: Assistant, host: str, port: int) -> None:
    """Launch the Gradio WebUI."""
    from qwen_agent.gui import WebUI

    # Wrap with smart routing: vLLM when free, Ollama when vLLM is busy
    agent = LockingAgent(inner_agent, backup_agent, _primary_model_lock)

    chatbot_config = {
        "user.name": "User",
        "input.placeholder": "Ask me anything... (Activity monitor: http://localhost:7861)",
        "prompt.suggestions": [
            "Search the web for latest Python news",
            "What's the current price of AAPL?",
            "Schedule a task to check stock prices every hour",
            "Show me my scheduled tasks",
            "Read my current heartbeat checklist",
        ],
    }

    logger.info(f"Starting Gradio WebUI on {host}:{port}")

    # Start a separate lightweight status API on port+1
    from sandbox_agent.status_server import start_status_server
    start_status_server(port + 1)
    logger.info(f"Status API started on {host}:{port + 1}/status")

    webui = WebUI(agent, chatbot_config=chatbot_config)
    webui.run(server_name=host, server_port=port)


def _run_repl(inner_agent: Assistant, backup_agent: Assistant) -> None:
    """Run the terminal REPL."""

    class StreamPrinter:
        def __init__(self):
            self._printed = 0

        def update(self, response: List[Message]) -> None:
            if not response:
                return
            last = response[-1]
            if last.role == "assistant" and isinstance(last.content, str):
                content = last.content
                if len(content) > self._printed:
                    sys.stdout.write(content[self._printed:])
                    sys.stdout.flush()
                    self._printed = len(content)

        def finish(self) -> None:
            if self._printed > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()

    print("Sandbox Agent ready. Type your message (Ctrl+C to exit).")
    print(f"  Primary model: {PRIMARY_LLM_CFG['model']} @ {PRIMARY_LLM_CFG['model_server']}")
    print(f"  Background model: {BACKGROUND_LLM_CFG['model']} @ {BACKGROUND_LLM_CFG['model_server']}")
    print()

    messages: List[Message] = []
    try:
        while True:
            try:
                user_input = input("> ")
            except EOFError:
                break

            if not user_input.strip():
                continue

            messages.append(Message(role="user", content=user_input))

            # Trim old turns if approaching token budget
            messages = trim_to_budget(messages)

            printer = StreamPrinter()
            got_lock = _primary_model_lock.acquire(blocking=False)
            agent = inner_agent if got_lock else backup_agent
            if not got_lock:
                print("  (vLLM busy — using Ollama)")
            try:
                response: List[Message] = []
                for response in agent.run(messages=messages):
                    printer.update(response)
            finally:
                if got_lock:
                    _primary_model_lock.release()
            printer.finish()

            log_turn(messages[-1], response)
            messages.extend(response)
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
