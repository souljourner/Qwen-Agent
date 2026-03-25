"""Main entry point for the sandboxed Qwen Agent.

Lane-based concurrency (OpenClaw pattern):
- Main lane: interactive UI (Gradio WebUI or REPL) using primary vLLM model
- Heartbeat lane: periodic checks using background Ollama model
- Cron lane: scheduled task execution using background Ollama model
"""

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
from sandbox_agent.chat_logger import log_background_task, log_turn
from sandbox_agent.heartbeat.heartbeat_runner import HeartbeatRunner
from sandbox_agent.scheduler.task_queue import TaskQueue
from sandbox_agent.token_budget import compute_request_timeout, estimate_messages_tokens, trim_to_budget

# Import tools to trigger @register_tool decorators
import sandbox_agent.tools.api_tools  # noqa: F401
import sandbox_agent.tools.self_edit_tools  # noqa: F401
import sandbox_agent.tools.code_interpreter  # noqa: F401
import sandbox_agent.chat_logger  # noqa: F401 (registers list_chat_logs, read_chat_log)
import sandbox_agent.tools.project_tools  # noqa: F401
import sandbox_agent.scheduler.scheduler_tools  # noqa: F401

logger = logging.getLogger(__name__)

# Lock for the primary vLLM model — held while any lane is actively running on it.
# Background lanes try to acquire non-blocking: if free, use primary; if held, fall back to Ollama.
_primary_model_lock = Lock()


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
        result = original_call_tool(tool_name, tool_args, **kwargs)
        result_preview = str(result)[:200]
        logger.info(f"Tool result: {tool_name} -> {result_preview}")
        return result

    agent._call_tool = _logged_call_tool
    return agent


class LockingAgent(Assistant):
    """Wraps an Assistant so that run() holds the primary model lock.

    This lets the Gradio WebUI (which calls agent.run() directly) automatically
    hold the lock, causing background tasks to fall back to Ollama.
    """

    def __init__(self, inner: Assistant, lock: Lock):
        # Bypass Assistant.__init__ — we delegate everything to inner
        self._inner = inner
        self._lock = lock

    def run(self, *args, **kwargs) -> Iterator[List[Message]]:
        # Extract the user message for logging (Gradio passes history as first positional arg)
        messages = args[0] if args else kwargs.get("messages", [])
        user_msg = None
        if messages:
            # Set dynamic request timeout based on payload size
            msg_list = [m if isinstance(m, Message) else Message(**m) for m in messages]
            timeout = compute_request_timeout(msg_list)
            self._inner.llm.generate_cfg["request_timeout"] = timeout

            for m in reversed(msg_list):
                if m.role == "user":
                    user_msg = m
                    break

        with self._lock:
            response = []
            for response in self._inner.run(*args, **kwargs):
                yield response

        # Log the turn after completion
        if user_msg and response:
            try:
                log_turn(user_msg, response)
            except Exception:
                pass  # Logging is best-effort

    def run_nonstream(self, *args, **kwargs):
        with self._lock:
            return self._inner.run_nonstream(*args, **kwargs)

    # Delegate attribute access to inner agent
    def __getattr__(self, name):
        return getattr(self._inner, name)


def run_on_best_available(system_message: str, messages: List[Message]) -> List[Message]:
    """Run an isolated agent session, preferring the primary model if it's free.

    Tries to acquire the primary model lock non-blocking:
    - If free: runs on primary vLLM (better quality)
    - If busy (main lane is chatting): falls back to Ollama
    Sets dynamic request timeout based on message payload size.
    """
    timeout = compute_request_timeout(messages)

    if _primary_model_lock.acquire(blocking=False):
        try:
            logger.info(f"Background task using primary model (idle, timeout={timeout}s)")
            agent = create_agent(system_message, llm_cfg=PRIMARY_LLM_CFG)
            agent.llm.generate_cfg["request_timeout"] = timeout
            response: List[Message] = []
            for response in agent.run(messages=messages):
                pass
            return response
        finally:
            _primary_model_lock.release()
    else:
        logger.info(f"Background task using Ollama (primary model busy, timeout={timeout}s)")
        agent = create_agent(system_message, llm_cfg=BACKGROUND_LLM_CFG)
        agent.llm.generate_cfg["request_timeout"] = timeout
        response: List[Message] = []
        for response in agent.run(messages=messages):
            pass
        return response


def cron_loop(system_message: str, task_queue: TaskQueue, poll_interval: int = 60) -> None:
    """Background loop that checks for due scheduled tasks and runs them."""
    logger.info(f"Cron loop started (poll interval: {poll_interval}s)")
    while True:
        try:
            due_tasks = task_queue.get_due_tasks()
            for task in due_tasks:
                logger.info(f"Cron: executing task [{task.id}] {task.name}")
                task_queue.update_task(task.id, status="running")
                try:
                    # Build task prompt with project and checkpoint context
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
                    logger.info(f"Cron: task [{task.id}] completed")
                except Exception as e:
                    logger.exception(f"Cron: task [{task.id}] failed")
                    task_queue.update_task(task.id, status="failed", last_error=str(e)[:500])
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
        on_alert=lambda msg: logger.warning(f"HEARTBEAT ALERT: {msg}"),
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
    import sandbox_agent.tools.code_interpreter as ci
    bridge_port = start_bridge(BACKGROUND_LLM_CFG)
    ci._llm_init_code = get_kernel_init_code(bridge_port)
    logger.info(f"LLM bridge started on port {bridge_port} -> {BACKGROUND_LLM_CFG['model']} @ {BACKGROUND_LLM_CFG['model_server']}")

    # Pre-warm Ollama model (lazy-loads into GPU on first call)
    try:
        import requests as _req
        logger.info(f"Pre-warming Ollama model ({BACKGROUND_LLM_CFG['model']})...")
        _req.post(
            f"{BACKGROUND_LLM_CFG['model_server']}/chat/completions",
            json={"model": BACKGROUND_LLM_CFG["model"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=120,
        )
        logger.info("Ollama model pre-warmed")
    except Exception as e:
        logger.warning(f"Failed to pre-warm Ollama model: {e}")

    # Main agent on primary vLLM model
    # Main agent uses system message WITH metadata (date/time/location)
    # Background sessions use the base system_message WITHOUT metadata (static for KV cache)
    inner_agent = create_agent(main_system_message, llm_cfg=PRIMARY_LLM_CFG)
    logger.info(f"Main agent created (model: {PRIMARY_LLM_CFG['model']})")

    # Start background lanes
    _start_background_lanes(system_message, task_queue)

    if args.mode == "gradio":
        _run_gradio(inner_agent, args.host, args.port)
    else:
        _run_repl(inner_agent)


def _run_gradio(inner_agent: Assistant, host: str, port: int) -> None:
    """Launch the Gradio WebUI."""
    from qwen_agent.gui import WebUI

    # Wrap agent with lock so background tasks fall back to Ollama while user is chatting
    agent = LockingAgent(inner_agent, _primary_model_lock)

    chatbot_config = {
        "user.name": "User",
        "input.placeholder": "Ask me anything...",
        "prompt.suggestions": [
            "Search the web for latest Python news",
            "What's the current price of AAPL?",
            "Schedule a task to check stock prices every hour",
            "Show me my scheduled tasks",
            "Read my current heartbeat checklist",
        ],
    }

    logger.info(f"Starting Gradio WebUI on {host}:{port}")
    webui = WebUI(agent, chatbot_config=chatbot_config)
    webui.run(server_name=host, server_port=port)


def _run_repl(inner_agent: Assistant) -> None:
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
            with _primary_model_lock:
                response: List[Message] = []
                for response in inner_agent.run(messages=messages):
                    printer.update(response)
            printer.finish()

            log_turn(messages[-1], response)
            messages.extend(response)
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
