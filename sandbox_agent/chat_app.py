"""Chainlit entrypoint for sandbox_agent.

Phase 1 scope: skeleton + persistence.
- Plain text streaming from the agent into a single `cl.Message`.
- Conversation history persisted via the SQLAlchemy data layer at
  `sandbox_agent/chat_data_layer.py`.
- On thread resume, history is rebuilt from the persisted messages so the
  next turn has full context.

NOT YET wired up (deliberately deferred):
- Tool-call and reasoning_content visualization as `cl.Step` (Phase 2).
- Background lanes (cron, heartbeat, LLM bridge, status server) — Phase 3.

Run from the host venv:
    chainlit run sandbox_agent/chat_app.py --port 7862

Or from the container after Phase 4 cutover:
    chainlit run /app/sandbox_agent/chat_app.py --host 0.0.0.0 --port 7860
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import chainlit as cl

# Import tool modules to trigger @register_tool registrations. Same set as
# sandbox_agent/main.py — the agent expects these in TOOL_REGISTRY when it
# constructs the Assistant from `function_list=TOOL_LIST`.
import sandbox_agent.tools.api_tools  # noqa: F401
import sandbox_agent.tools.self_edit_tools  # noqa: F401
import sandbox_agent.tools.code_interpreter  # noqa: F401
import sandbox_agent.tools.exec_tool  # noqa: F401
import sandbox_agent.chat_logger  # noqa: F401
import sandbox_agent.tools.project_tools  # noqa: F401
import sandbox_agent.tools.notification_tools  # noqa: F401
import sandbox_agent.pipeline.pipeline_tools  # noqa: F401
import sandbox_agent.scheduler.scheduler_tools  # noqa: F401

from qwen_agent.llm.schema import Message

from sandbox_agent.chat_data_layer import make_data_layer
from sandbox_agent.chat_logger import log_turn
from sandbox_agent.config import (
    BACKGROUND_LLM_CFG,
    PRIMARY_LLM_CFG,
    load_system_message,
)
from sandbox_agent.main import (
    LockingAgent,
    _primary_model_lock,
    bootstrap_background,
    create_agent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-load bootstrap: start cron + heartbeat + LLM bridge + (optionally)
# status server. Idempotent — bootstrap_background() guards against re-entry.
#
# STATUS_SERVER_PORT controls the status-server lane:
#   - default: 0 (do NOT start; appropriate when a separate container or
#     a previously-running process already owns port 7861)
#   - set to 7861 (or another port) to start it from this process
# ---------------------------------------------------------------------------

_STATUS_PORT = int(os.environ.get("STATUS_SERVER_PORT", "0"))
try:
    bootstrap_background(status_server_port=_STATUS_PORT)
    logger.info("Background lanes ready (status_server_port=%d)", _STATUS_PORT)
except Exception:  # noqa: BLE001
    # We do NOT want a bootstrap failure to take down the chat app entirely;
    # log and continue. Pipelines / cron will be unavailable but the chat
    # surface still works on whatever lanes did come up.
    logger.exception("bootstrap_background failed; chat will start without full background lanes")


# ---------------------------------------------------------------------------
# Data layer (Chainlit calls this factory at startup)
# ---------------------------------------------------------------------------

@cl.data_layer
def _get_data_layer():
    return make_data_layer()


# ---------------------------------------------------------------------------
# Single-user header auth — required for Chainlit's data layer to persist
# threads. We're a single-user deployment; every request gets the same user.
# Swap for `@cl.password_auth_callback` later if you ever expose this beyond
# localhost.
# ---------------------------------------------------------------------------

_SINGLE_USER_ID = "default"


@cl.header_auth_callback
def _header_auth(headers) -> Optional[cl.User]:
    return cl.User(identifier=_SINGLE_USER_ID, metadata={"role": "owner"})


# ---------------------------------------------------------------------------
# Agent (process-global, constructed once on first chat)
# ---------------------------------------------------------------------------

_AGENT: Optional[LockingAgent] = None
_AGENT_LOCK = threading.Lock()


def _get_agent() -> LockingAgent:
    """Return the process-shared LockingAgent. Lazily constructed so we don't
    pay the import + agent-creation cost at chainlit startup."""
    global _AGENT
    if _AGENT is None:
        with _AGENT_LOCK:
            if _AGENT is None:
                system_message = load_system_message()
                inner = create_agent(system_message, llm_cfg=PRIMARY_LLM_CFG)
                backup = create_agent(system_message, llm_cfg=BACKGROUND_LLM_CFG)
                _AGENT = LockingAgent(inner, backup, _primary_model_lock)
                logger.info(
                    "Agent constructed (primary=%s, backup=%s)",
                    PRIMARY_LLM_CFG["model"], BACKGROUND_LLM_CFG["model"],
                )
    return _AGENT


# ---------------------------------------------------------------------------
# Bridge: stream the agent's sync generator into Chainlit's async event loop
# ---------------------------------------------------------------------------

_DONE = object()


def _run_agent_in_thread(agent, messages, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Worker that drains `agent.run(messages)` and pushes each yielded chunk
    onto `queue` for the asyncio side to consume.

    Sentinel values:
    - `("chunk", List[Message])` — agent yielded a cumulative thread state
    - `("done", None)`           — generator exhausted normally
    - `("error", Exception)`     — generator raised; coroutine should re-raise
    """
    try:
        for chunk in agent.run(messages=messages):
            asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop)
        asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
    except Exception as e:  # noqa: BLE001 — surface anything to the user
        logger.exception("Agent run raised")
        asyncio.run_coroutine_threadsafe(queue.put(("error", e)), loop)


# ---------------------------------------------------------------------------
# Chainlit handlers
# ---------------------------------------------------------------------------

HISTORY_KEY = "history"


@cl.on_chat_start
async def on_chat_start():
    """Fresh chat — empty history."""
    cl.user_session.set(HISTORY_KEY, [])


_FEEDBACK_LOG = os.path.join(os.environ.get("DATA_DIR", "/app/data"), "feedback.jsonl")


def _lookup_step_output(step_id: str) -> str:
    """Synchronously read a step's text from chat.db (the message being rated)."""
    if not step_id:
        return ""
    try:
        from sandbox_agent.chat_data_layer import resolve_conninfo, _sqlite_path_from_conninfo
        db_path = _sqlite_path_from_conninfo(resolve_conninfo())
        if not db_path or not os.path.exists(db_path):
            return ""
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT output FROM steps WHERE id = ?", (step_id,)).fetchone()
        return (row[0] or "") if row else ""
    except Exception:  # noqa: BLE001
        logger.debug("feedback: could not look up step output for %s", step_id, exc_info=True)
        return ""


@cl.on_feedback
async def on_feedback(feedback):
    """Record a 👍/👎 vote so it's visible to the agent.

    Writes a structured line to DATA_DIR/feedback.jsonl AND a human-readable
    note to today's chat log (which the agent reads via the read_chat_log
    tool). Without this, the feedback buttons are write-only sinks Chainlit
    persists to the `feedbacks` table but nothing ever reads.
    """
    value = getattr(feedback, "value", None)
    sentiment = "👍 positive" if value == 1 else "👎 negative" if value == 0 else f"value={value}"
    comment = (getattr(feedback, "comment", None) or "").strip()
    for_id = getattr(feedback, "forId", None) or ""
    thread_id = getattr(feedback, "threadId", None)

    rated_text = _lookup_step_output(for_id)
    excerpt = " ".join(rated_text.split())[:300]

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "for_id": for_id,
        "value": value,
        "sentiment": sentiment,
        "comment": comment,
        "rated_excerpt": excerpt,
    }
    try:
        os.makedirs(os.path.dirname(_FEEDBACK_LOG) or ".", exist_ok=True)
        with open(_FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        logger.exception("feedback: failed to write %s", _FEEDBACK_LOG)

    try:
        from sandbox_agent.chat_logger import log_feedback_note
        log_feedback_note(sentiment, comment, excerpt)
    except Exception:  # noqa: BLE001
        logger.debug("feedback: failed to append to daily chat log", exc_info=True)

    logger.info("Feedback recorded: %s%s", sentiment, f" — {comment!r}" if comment else "")


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resumed chat — rebuild history from the persisted thread so the agent
    has full conversation context on the next turn.

    Chainlit's thread dict has a `steps` list. User messages and assistant
    messages live there with `type` discriminators that vary by Chainlit
    version. We accept either `user_message`/`assistant_message` or the more
    generic `run`/`text` pattern, and inspect the role-relevant fields.
    """
    history: List[Message] = []
    steps = thread.get("steps", []) if isinstance(thread, dict) else []
    for step in steps:
        step_type = (step.get("type") or "").lower()
        output = step.get("output") or ""
        if not output:
            continue
        if step_type in ("user_message", "user"):
            history.append(Message(role="user", content=output))
        elif step_type in ("assistant_message", "assistant", "ai", "llm"):
            history.append(Message(role="assistant", content=output))
        # Tool steps and reasoning are intentionally NOT replayed into the
        # agent's input — the agent re-derives those when it processes the
        # next user turn.
    cl.user_session.set(HISTORY_KEY, history)
    logger.info("Resumed thread with %d messages of history", len(history))


class _StreamBridge:
    """Maps a cumulative `List[Message]` stream from `agent.run()` into
    Chainlit primitives (`cl.Message` for text, `cl.Step` for tool calls and
    reasoning).

    State is keyed by message index in the cumulative thread. Indices stay
    stable within an LLM call's streaming and new indices appear when the
    agent executes tools and continues. Each entry tracks how much of that
    message we've already pushed to the UI so we send only deltas.
    """

    def __init__(self):
        # Per-message-index UI state.
        # entry shape: {"kind": "text"|"tool"|"thought", "obj": cl.Message|cl.Step,
        #               "streamed": int (chars already pushed), "name": str}
        self._by_index: Dict[int, dict] = {}
        # Allow result messages to find their tool step by function_id when present.
        self._tool_by_function_id: Dict[str, "cl.Step"] = {}

    @staticmethod
    def _function_id(msg) -> Optional[str]:
        extra = getattr(msg, "extra", None) or {}
        return extra.get("function_id")

    @staticmethod
    def _parent_id() -> Optional[str]:
        """ID of the on_message run step, so the cl.Step objects we create
        nest UNDER it — same parent as the assistant messages. Without this
        they're orphaned (parentId=NULL) and Chainlit renders them as
        standalone items at the bottom of the chat, after the answer."""
        try:
            run = cl.context.current_run
            return run.id if run is not None else None
        except Exception:  # noqa: BLE001
            return None

    async def consume(self, messages: List[Message]) -> None:
        for i, msg in enumerate(messages):
            try:
                await self._consume_one(i, msg)
            except Exception:  # noqa: BLE001 — never let a UI render kill the run
                logger.exception("StreamBridge: failed to render message at index %d", i)

    async def _consume_one(self, i: int, msg: Message) -> None:
        role = getattr(msg, "role", None)
        if role == "function":
            await self._close_tool_with_result(msg)
            return
        if role != "assistant":
            return  # system/user messages don't render here

        # Reasoning content lives on its own logical channel — render even if
        # `content` is also populated on the same message.
        rc = getattr(msg, "reasoning_content", None)
        if rc:
            await self._stream_thought(i, rc)

        content = getattr(msg, "content", None)
        function_call = getattr(msg, "function_call", None)
        if function_call is not None:
            await self._stream_tool_call(i, msg, function_call)
        elif isinstance(content, str) and content.strip():
            # `.strip()` so the model emitting a couple of leading newlines
            # before a tool call doesn't create an empty-looking bubble.
            await self._stream_text(i, content)
        # Empty / whitespace-only assistant messages with no function_call are
        # streaming placeholders — ignore until they grow real content.

    async def _stream_text(self, i: int, text: str) -> None:
        state = self._by_index.get(i)
        first = False
        if state is None or state["kind"] != "text":
            cl_msg = cl.Message(content="")
            # Send IMMEDIATELY to claim the message's position in the chat
            # history (relative to any cl.Step we've already sent). Without
            # this, the message renders only when streaming finishes — and
            # ends up out-of-order relative to tool Steps that were sent
            # mid-stream. Also: stream_token does NOT auto-persist; the
            # initial send() is required for the data layer to track it.
            await cl_msg.send()
            state = {"kind": "text", "obj": cl_msg, "streamed": 0}
            self._by_index[i] = state
            first = True
        if len(text) > state["streamed"]:
            delta = text[state["streamed"]:]
            state["streamed"] = len(text)
            if first:
                delta = delta.lstrip("\n")  # trim leading newlines from the bubble
                if not delta:
                    return
            await state["obj"].stream_token(delta)

    async def finalize(self) -> None:
        """End-of-run cleanup. Calls update() on each streamed cl.Message so
        the final content is persisted to the data layer (Chainlit's
        stream_token does not persist on its own — explicit update is
        required after the last token)."""
        for state in self._by_index.values():
            obj = state.get("obj")
            if obj is None:
                continue
            try:
                await obj.update()
            except Exception:  # noqa: BLE001
                logger.debug("finalize: update failed for %s", state.get("kind"), exc_info=True)

    async def _stream_thought(self, i: int, reasoning: str) -> None:
        # Use a dedicated key so a single message-index can carry both a
        # thought step (for reasoning_content) and a text/tool element.
        if not reasoning.strip():
            return  # whitespace-only reasoning — don't create an empty step
        key = -(i + 1)  # negative key namespace for thoughts
        state = self._by_index.get(key)
        if state is None:
            step = cl.Step(
                name="thinking", type="thought", default_open=False,
                parent_id=self._parent_id(),
            )
            await step.send()
            state = {"kind": "thought", "obj": step, "streamed": 0}
            self._by_index[key] = state
        if len(reasoning) > state["streamed"]:
            state["obj"].output = reasoning  # cumulative; chainlit renders the whole thing
            await state["obj"].update()
            state["streamed"] = len(reasoning)

    async def _stream_tool_call(self, i: int, msg, function_call) -> None:
        name = getattr(function_call, "name", None) or "tool"
        args = getattr(function_call, "arguments", "") or ""
        function_id = self._function_id(msg)

        state = self._by_index.get(i)
        if state is None or state["kind"] != "tool":
            step = cl.Step(
                name=name, type="tool", show_input="json",
                parent_id=self._parent_id(),
            )
            await step.send()
            state = {
                "kind": "tool", "obj": step, "streamed": 0,
                "name": name, "function_id": function_id,
            }
            self._by_index[i] = state
            if function_id:
                self._tool_by_function_id[function_id] = step

        # Stream growing arguments as the tool-call's input.
        if len(args) > state["streamed"]:
            state["obj"].input = args
            await state["obj"].update()
            state["streamed"] = len(args)

    async def _close_tool_with_result(self, msg) -> None:
        function_id = self._function_id(msg)
        target_step = None
        if function_id:
            target_step = self._tool_by_function_id.get(function_id)
        if target_step is None:
            # Fallback: most recent open tool step with matching name.
            tool_name = getattr(msg, "name", None)
            for state in reversed(list(self._by_index.values())):
                if state.get("kind") == "tool" and state.get("name") == tool_name:
                    target_step = state["obj"]
                    break
        if target_step is None:
            return
        result = getattr(msg, "content", "")
        if not isinstance(result, str):
            result = str(result)
        target_step.output = result
        try:
            await target_step.update()
        except Exception:  # noqa: BLE001
            logger.debug("StreamBridge: failed to update closed tool step", exc_info=True)


@cl.on_message
async def on_message(message: cl.Message):
    history: List[Message] = cl.user_session.get(HISTORY_KEY) or []
    history.append(Message(role="user", content=message.content))

    agent = _get_agent()
    bridge = _StreamBridge()

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    threading.Thread(
        target=_run_agent_in_thread,
        args=(agent, history, queue, loop),
        daemon=True,
        name="chainlit-agent-run",
    ).start()

    final_response: List[Message] = []
    while True:
        kind, payload = await queue.get()
        if kind == "done":
            break
        if kind == "error":
            await bridge.finalize()  # persist whatever did stream before the error
            await cl.ErrorMessage(
                content=f"Agent error: {type(payload).__name__}: {payload}"
            ).send()
            return
        # kind == "chunk"
        final_response = payload
        await bridge.consume(payload)

    # Finalize: persist all streamed cl.Messages to the data layer. Without
    # this, stream_token tokens reach the UI but no chat.db row is written.
    await bridge.finalize()

    # Persist response into agent-side history for the next turn. We feed back
    # the full thread (assistant text + function calls + function results) so
    # the agent has continuity. Reasoning_content is intentionally NOT carried
    # forward — it inflates the prompt and the model re-derives it on demand.
    user_msg = history[-1]  # the user message we appended at the top
    for msg in final_response:
        if msg.role in ("assistant", "function") and (
            (msg.content and (isinstance(msg.content, str) or isinstance(msg.content, list)))
            or getattr(msg, "function_call", None)
        ):
            history.append(msg)
    cl.user_session.set(HISTORY_KEY, history)

    # Append to the daily markdown chat log — separate from Chainlit's SQLite
    # persistence; markdown is the human-readable, exportable archive.
    try:
        log_turn(user_msg, final_response)
    except Exception:  # noqa: BLE001 — never let logging kill the chat
        logger.exception("log_turn failed (chat already delivered)")
