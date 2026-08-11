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
import base64
import json
import logging
import mimetypes
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import chainlit as cl
from chainlit.utils import utc_now  # ISO timestamp helper Chainlit uses for step start/end

from sandbox_agent import cancellation
from sandbox_agent.ui_output import _cap_step_output

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
import sandbox_agent.tools.display_tools  # noqa: F401
import sandbox_agent.tools.skill_tools  # noqa: F401
import sandbox_agent.tools.session_search_tools  # noqa: F401
import sandbox_agent.tools.browser_tools  # noqa: F401

from qwen_agent.llm.schema import ContentItem, FunctionCall, Message

from sandbox_agent.chat_data_layer import make_data_layer
from sandbox_agent.chat_logger import log_turn
from sandbox_agent.tools.code_interpreter import register_progress_hook, unregister_progress_hook
from sandbox_agent.tools.display_tools import register_display_hook, unregister_display_hook
from qwen_agent.llm.oai import register_usage_hook, unregister_usage_hook
from sandbox_agent.config import (
    BACKGROUND_LLM_CFG,
    PRIMARY_LLM_CFG,
    format_datetime,
    load_system_message,
    session_metadata,
)
from sandbox_agent.main import (
    LockingAgent,
    bootstrap_background,
    create_agent,
)

logger = logging.getLogger(__name__)


# Serve persisted Chainlit element blobs (display_doc images / PDFs / files) so
# they survive a page reload. LocalFsStorageClient writes them under
# DATA_DIR/.cl_elements and hands the data layer /cl-elements/<key> URLs; this
# route serves those bytes — same origin as Chainlit (:7860), so no CORS. The
# {object_key:path} matcher captures the slash in Chainlit's "<user>/<elem>" keys;
# resolve_object_path() rejects traversal.
try:
    from chainlit.server import app as _cl_app
    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import FileResponse as _FileResponse
    from sandbox_agent.chat_storage import resolve_object_path as _resolve_object_path

    @_cl_app.get("/cl-elements/{object_key:path}")
    async def _serve_cl_element(object_key: str):
        try:
            full = _resolve_object_path(object_key)
        except ValueError:
            raise _HTTPException(status_code=400, detail="bad object key")
        if not os.path.isfile(full):
            raise _HTTPException(status_code=404, detail="not found")
        return _FileResponse(full)

    # The decorator APPENDS the route — after Chainlit's included router, whose
    # SPA catch-all (`/{full_path:path}`, chainlit/server.py) matches first and
    # returns index.html for /cl-elements/* (the browser then can't decode HTML
    # as an image → display_doc images/PDFs render as an empty box). Starlette
    # matches routes in list order, so hoist ours to the front.
    _cl_app.router.routes.insert(0, _cl_app.router.routes.pop())
except Exception:  # noqa: BLE001 — never let route setup break app import
    logger.exception("Failed to mount /cl-elements route; persisted element reload may not work")


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
                _AGENT = LockingAgent(inner, backup)
                logger.info(
                    "Agent constructed (primary=%s, backup=%s)",
                    PRIMARY_LLM_CFG["model"], BACKGROUND_LLM_CFG["model"],
                )
    return _AGENT


# ---------------------------------------------------------------------------
# Bridge: stream the agent's sync generator into Chainlit's async event loop
# ---------------------------------------------------------------------------

_DONE = object()


def _run_agent_in_thread(
    agent,
    messages,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    run_id: str,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    frozen_metadata: str = "",
):
    """Worker that drains `agent.run(messages)` and pushes each yielded chunk
    onto `queue` for the asyncio side to consume.

    Runs inside `cancellation.begin_run(run_id)` so the chat Stop button works:
    when the on_message coroutine is cancelled it calls `cancellation.cancel(run_id)`,
    which sets this run's cancel flag (the agent loop raises RunCancelled at its
    next yield, via `cancellation.guard` in `_compacting_run`) and SIGKILLs any
    exec/code_interpreter subprocess the run is wedged in. The begin_run context
    tags THIS thread — where `agent.run` and the tools execute — so
    `register_child_pgid` and the loop's cancellation check see it.

    Sentinel values:
    - `("chunk", List[Message])`   — agent yielded a cumulative thread state
    - `("tool_progress", str)`     — live stdout from a running code_interpreter
    - `("usage_summary", dict)`    — per-turn token totals (prompt/completion/calls)
    - `("done", None)`             — generator exhausted normally (or cancelled)
    - `("error", Exception)`       — generator raised; coroutine should re-raise
    """
    def _push(item):
        # call_soon_threadsafe + put_nowait is much lighter than
        # run_coroutine_threadsafe(queue.put(...)) — the latter creates a
        # concurrent.futures.Future *and* schedules a coroutine Task on the
        # event loop for every push. At ~70 chunks/sec from a streaming LLM,
        # that's ~thousands of put-tasks per turn competing with the drain
        # coroutine for the event loop, which makes the UI lag behind. The
        # queue is asyncio.Queue (unbounded) so put_nowait never raises.
        loop.call_soon_threadsafe(queue.put_nowait, item)

    # While this thread runs a code_interpreter call, its stdout streams here
    # as ("tool_progress", text) so the on_message coroutine can show it on the
    # live tool step. Cleared in `finally` so it doesn't leak to a later run.
    register_progress_hook(lambda text: _push(("tool_progress", text)))
    # Slow-compaction heads-up: summarize-tier compactions can run for minutes
    # (cold prefill of the whole history through the summarizer) — surface a
    # notice instead of a silent hang. Same per-thread lifecycle as above.
    from sandbox_agent.compaction import register_compaction_notice_hook
    register_compaction_notice_hook(lambda payload: _push(("compaction_notice", payload)))
    # display_doc pushes ("display_doc", payload) so the doc is rendered to the
    # user out-of-band — its content never enters the agent's message stream.
    register_display_hook(lambda payload: _push(("display_doc", payload)))

    # Aggregate per-call vLLM usage across all _call_llm invocations in this turn
    # (a turn often makes several — reasoning, then per-tool, then the final answer).
    # Track per-turn: the LARGEST single prompt the model processed (= the
    # context-window high-water mark — what we'd hit the 256k cap with) and
    # the SUM of completions (= total work generated this turn). Per-call
    # prompt sums (a.k.a. "billable tokens") aren't useful here since we run
    # against a local vLLM, not a paid API — what we want to see is how close
    # any single call got to the context limit.
    turn_usage = {"max_prompt": 0, "completion_total": 0, "calls": 0}

    def _on_usage(info: dict) -> None:
        p = int(info.get("prompt_tokens", 0) or 0)
        c = int(info.get("completion_tokens", 0) or 0)
        if p > turn_usage["max_prompt"]:
            turn_usage["max_prompt"] = p
        turn_usage["completion_total"] += c
        turn_usage["calls"] += 1

    register_usage_hook(_on_usage)
    # Stamp this worker thread with the originating chat so that any tool the
    # agent calls (schedule_task, start_pipeline, …) can read it via
    # chat_origin.current_origin() and tag the task it creates. The notifier
    # then routes that task's completion notice back to THIS session only.
    from sandbox_agent import chat_origin
    if session_id or thread_id:
        chat_origin.set_current_origin({"session_id": session_id, "thread_id": thread_id})
    # Freeze date/time ONCE per conversation and bake into both agents'
    # system_message. The timestamp is captured on the asyncio side and passed
    # in as frozen_metadata → vLLM KV prefix cache stays stable.
    if frozen_metadata:
        fresh_sys = load_system_message() + frozen_metadata
        agent._inner.system_message = fresh_sys
        agent._backup.system_message = fresh_sys

    try:
        with cancellation.begin_run(run_id):
            for chunk in agent.run(messages=messages):
                _push(("chunk", chunk))
        _push(("usage_summary", dict(turn_usage)))
        _push(("done", None))
    except cancellation.RunCancelled:
        logger.info("chat run %s cancelled — agent halted", run_id)
        _push(("usage_summary", dict(turn_usage)))
        _push(("done", None))
    except Exception as e:  # noqa: BLE001 — surface anything to the user
        logger.exception("Agent run raised")
        _push(("error", e))
    finally:
        unregister_progress_hook()
        from sandbox_agent.compaction import unregister_compaction_notice_hook
        unregister_compaction_notice_hook()
        unregister_display_hook()
        unregister_usage_hook()
        chat_origin.set_current_origin(None)


# ---------------------------------------------------------------------------
# Background-task completion notices (Hermes-style alert)
# ---------------------------------------------------------------------------
# When a cron task / pipeline stage finishes, it pushes a record onto
# task_notify's queue (the producer side, in main.py). A single background loop
# here drains that queue and pushes a notice into every connected Chainlit
# session — so the agent/user learns the job is done without having to ask. If
# no session is connected the events stay queued and get delivered as soon as
# one connects (the loop is the sole drainer).

_NOTIFIER_POLL_SECONDS = 4.0
_notifier_started = False


def _format_task_notice(evt: dict) -> str:
    """Markdown notice for the UI (shows in the chat bubble)."""
    name = evt.get("name") or evt.get("task_id") or "task"
    ok = evt.get("ok", True)
    result = (evt.get("result") or "").strip()
    head = (f"✅ Background task **{name}** finished."
            if ok else f"⚠️ Background task **{name}** failed.")
    if result:
        snippet = result if len(result) <= 600 else result[:600].rstrip() + " …"
        return f"{head}\n\n{snippet}\n\n_(ask me about it for the full result — I'll look it up with `list_tasks`)_"
    return head


def _format_task_notice_for_agent(evt: dict) -> str:
    """The directive the agent sees in its conversation history.

    Injected as role='user' with a leading `[system event]` tag — vLLM's chat
    completions endpoint rejects mid-thread role='system' messages with HTTP
    400 ("System message must be at the beginning"), even though the chat
    template would render them; the API validation runs first. The
    `[system event]` prefix tells the model this isn't a real user message
    but a notice from the runtime."""
    name = evt.get("name") or evt.get("task_id") or "task"
    ok = evt.get("ok", True)
    result = (evt.get("result") or "").strip()
    verb = "finished" if ok else "failed"
    body = f": {result[:600]}" if result else "."
    return (
        f"[{format_datetime()}] [system event] A background task '{name}' {verb}{body} "
        f"If a follow-up is warranted (read result files via project_read_file, "
        f"schedule the next step, send a notification, alert the user, etc.), "
        f"do it now; otherwise stay silent."
    )


async def _completion_notifier_loop() -> None:
    from chainlit.session import ws_sessions_id
    from chainlit.context import init_ws_context
    from sandbox_agent import task_notify

    while True:
        try:
            await asyncio.sleep(_NOTIFIER_POLL_SECONDS)
            # Only sessions that have a thread (been through on_chat_start/resume)
            # can receive a persisted message.
            sessions = [s for s in list(ws_sessions_id.values()) if getattr(s, "thread_id", None)]
            if not sessions:
                continue  # nobody to deliver to — leave events queued
            events = task_notify.drain()
            for evt in events:
                ui_content = _format_task_notice(evt)
                agent_line = _format_task_notice_for_agent(evt)
                # Route by origin: if the task was scheduled from a specific
                # chat session, deliver only there. Match session_id first
                # (same tab still open), then thread_id (reconnected tab on
                # the same thread). If origin is set but no session matches,
                # fall back to broadcast so the user still sees it somewhere.
                target_sessions = sessions
                origin = evt.get("origin") if isinstance(evt, dict) else None
                if origin:
                    by_session = [s for s in sessions if origin.get("session_id") and getattr(s, "id", None) == origin["session_id"]]
                    by_thread = [s for s in sessions if origin.get("thread_id") and getattr(s, "thread_id", None) == origin["thread_id"]]
                    matched = by_session or by_thread
                    if matched:
                        target_sessions = matched
                    else:
                        logger.info(
                            "notifier: origin session/thread no longer open (task=%s) — broadcasting",
                            evt.get("task_id"),
                        )
                for session in target_sessions:
                    try:
                        init_ws_context(session)
                        # 1. Show the user (live message in the chat).
                        await cl.Message(content=ui_content, author="background task").send()
                        # 2. Tell the agent (append to this session's HISTORY_KEY).
                        # cl.user_session stores the list by reference; both this
                        # loop and on_message mutate the same list, so this is a
                        # plain append, no clobber.
                        hist = cl.user_session.get(HISTORY_KEY)
                        if hist is None:
                            hist = []
                            cl.user_session.set(HISTORY_KEY, hist)
                        # extra marks this as machine chatter, not the
                        # user speaking — the retention ladder protects
                        # real user text far more aggressively.
                        hist.append(Message(role="user", content=agent_line,
                                            extra={"synthetic": "notifier"}))
                        # 3. Trigger a synthetic agent turn right now so it can
                        # follow up (read a result file, schedule the next step,
                        # send a notification) without waiting for the user.
                        # Serialized with on_message via the per-session lock.
                        # Damped: repeated notifications from the same task (a
                        # failing stage rescheduling itself) and post-Stop
                        # notifications deliver to UI + history but do NOT
                        # spawn agent turns — see notify_damper.
                        from sandbox_agent.notify_damper import should_trigger_synthetic_turn
                        if not should_trigger_synthetic_turn(str(evt.get("name") or evt.get("task_id") or "?")):
                            logger.info("notifier: synthetic turn suppressed (cooldown/mute) for %s",
                                        evt.get("name"))
                            continue
                        try:
                            async with _get_turn_lock():
                                await _execute_agent_turn(_get_agent(), hist, log_user_msg=None)
                        except asyncio.CancelledError:
                            raise
                        except Exception:  # noqa: BLE001
                            logger.debug("notifier: synthetic agent turn failed",
                                         exc_info=True)
                    except asyncio.CancelledError:
                        raise
                    except Exception:  # noqa: BLE001
                        logger.debug("notifier: deliver to session %s failed",
                                     getattr(session, "id", "?"), exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("completion notifier loop error")
            await asyncio.sleep(5)


def _ensure_notifier() -> None:
    """Start the completion-notifier loop once (idempotent). Called from
    on_chat_start / on_chat_resume so there's always a running event loop."""
    global _notifier_started
    if _notifier_started:
        return
    _notifier_started = True
    asyncio.create_task(_completion_notifier_loop())


# ---------------------------------------------------------------------------
# Chainlit handlers
# ---------------------------------------------------------------------------

HISTORY_KEY = "history"

# Model context window — used to render "max ctx N (P% of 256k)" in the
# per-turn footer so you can see at a glance how close any single LLM call
# this turn got to the limit. The primary model (qwen3.6-27b-linux) and
# backup (qwen3.5) both support 256k; bump this if the deployed model changes.
_CONTEXT_WINDOW_TOKENS = 256_000

# UI footer appended to each assistant message by on_message (token-usage line
# right above the feedback buttons). Persisted in chat.db.output as part of the
# message, but the agent must NOT see it in its conversation context on resume.
_STATS_FOOTER_MARKER = "\n\n---\n📊 _last turn:"


def _strip_stats_footer(text: str) -> str:
    i = text.find(_STATS_FOOTER_MARKER)
    return text[:i] if i != -1 else text


_active_run_id: Optional[str] = None


@cl.on_stop
async def on_stop():
    """Stop button: cancel the ACTIVE agent run (user turn or synthetic
    notifier turn) and mute further synthetic turns briefly — without this a
    notification storm was unstoppable from the UI (2026-07-16)."""
    from sandbox_agent.notify_damper import mute_on_stop
    mute_on_stop()
    if _active_run_id:
        cancelled = cancellation.cancel(_active_run_id)
        logger.info("Stop pressed: cancel(%s) -> %s; synthetic turns muted", _active_run_id, cancelled)
    else:
        logger.info("Stop pressed: no active run; synthetic turns muted")


@cl.on_chat_start
async def on_chat_start():
    """Fresh chat — empty history."""
    cl.user_session.set(HISTORY_KEY, [])
    _ensure_notifier()


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


# Char budget for replayed tool-call/result pairs during chat.db
# reconstruction (~100k tokens). Newest tools are kept; older history stays
# text-only and the per-turn compactor manages the rest.
RESUME_TOOL_CONTEXT_MAX_CHARS = 400_000


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resumed chat — restore the agent's context for the next turn.

    Preferred path: load the per-thread history sidecar (exact agent-side
    Message list incl. tool results — see sandbox_agent.chat_history).
    Fallback for threads predating the sidecar: reconstruct from chat.db
    steps, replaying tool-call/result pairs newest-first up to
    RESUME_TOOL_CONTEXT_MAX_CHARS."""
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    from sandbox_agent.chat_history import load_history
    history = load_history(thread_id)
    if history is not None:
        cl.user_session.set(HISTORY_KEY, history)
        logger.info("Resumed thread %s from sidecar (%d messages)", thread_id, len(history))
        _ensure_notifier()
        return

    history = _rebuild_history_from_thread(thread)
    cl.user_session.set(HISTORY_KEY, history)
    logger.info("Resumed thread %s via chat.db reconstruction (%d messages)",
                thread_id, len(history))
    _ensure_notifier()


def _rebuild_history_from_thread(thread, tool_budget_chars: int = RESUME_TOOL_CONTEXT_MAX_CHARS) -> List[Message]:
    """Rebuild agent history from a persisted Chainlit thread dict.

    Chainlit's thread dict has a `steps` list. User messages and assistant
    messages live there with `type` discriminators that vary by Chainlit
    version. Tool steps are replayed as function_call/result Message pairs,
    NEWEST-first within `tool_budget_chars` (older tools drop; text always
    stays) — restoring the evidence the agent gathered, not just its prose.
    """
    history: List[Message] = []
    steps = thread.get("steps", []) if isinstance(thread, dict) else []

    # Decide which tool steps fit the replay budget (newest-first selection,
    # rendered in original order).
    tool_indices = [i for i, s in enumerate(steps) if (s.get("type") or "").lower() == "tool"]
    included_tools = set()
    budget = tool_budget_chars
    for i in reversed(tool_indices):
        s = steps[i]
        size = len(s.get("output") or "") + len(s.get("input") or "")
        if size <= budget:
            included_tools.add(i)
            budget -= size
    # Index thread elements by the step they belong to, so a user_message
    # rebuild can re-attach any persisted images (Chainlit saved them to
    # disk; we re-encode as data URLs the same way on_message does).
    elements_by_step: Dict[str, List[dict]] = {}
    for el in (thread.get("elements") or []) if isinstance(thread, dict) else []:
        fid = el.get("forId") or el.get("for_id")
        if fid:
            elements_by_step.setdefault(fid, []).append(el)
    for i, step in enumerate(steps):
        step_type = (step.get("type") or "").lower()
        output = step.get("output") or ""
        if step_type == "tool":
            if i not in included_tools:
                continue
            tool_name = step.get("name") or "tool"
            history.append(Message(role="assistant", content="", function_call=FunctionCall(
                name=tool_name, arguments=step.get("input") or "{}")))
            history.append(Message(role="function", name=tool_name, content=output))
            continue
        if not output and not elements_by_step.get(step.get("id")):
            continue
        if step_type in ("user_message", "user"):
            # Pull the per-message timestamp persisted on cl.Message.metadata
            # (see on_message). Handle dict or json-string hydration depending
            # on data-layer behavior. Empty / missing → no prefix (graceful
            # degradation for messages saved before this feature).
            md = step.get("metadata") or {}
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except Exception:  # noqa: BLE001
                    md = {}
            sent_at = (md or {}).get("sent_at") or ""
            # Try to rebuild a multimodal user message if this step had image
            # elements persisted alongside.
            attached = elements_by_step.get(step.get("id") or "", [])
            image_parts: List[ContentItem] = []
            for el in attached:
                mime = (el.get("mime") or "").lower()
                if not mime.startswith("image/"):
                    continue
                path = el.get("path")
                if not path or not os.path.exists(path):
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                image_parts.append(ContentItem(
                    image=f"data:{mime};base64,{base64.b64encode(data).decode()}"
                ))
            if image_parts:
                parts: List[ContentItem] = []
                if output:
                    parts.append(ContentItem(text=output))
                parts.extend(image_parts)
                msg = Message(role="user", content=parts)
            else:
                msg = Message(role="user", content=output)
            _prepend_ts_to_agent_msg(msg, sent_at)
            history.append(msg)
        elif step_type in ("assistant_message", "assistant", "ai", "llm"):
            author = (step.get("name") or "").lower()
            # display_doc output is persisted (so it reloads in the UI) but must
            # NEVER re-enter the agent's context — skip it from the rebuild.
            if author == "document":
                continue
            # Background-task completion notices are persisted with
            # author="background task". Symmetric with the live path: rebuild
            # them as role="user" with a `[system event]` tag (vLLM rejects
            # mid-thread role="system" with HTTP 400 — see the live-path
            # comment in _format_task_notice_for_agent).
            if author == "background task":
                first_line = output.strip().splitlines()[0] if output.strip() else ""
                # strip leading emoji + markdown bold for a clean system line
                clean = first_line.replace("**", "").lstrip("✅⚠️ ").strip()
                history.append(Message(role="user", content=f"[system event] {clean}"))
            else:
                # Strip the UI-only token-usage footer and timestamp prefix so
                # the agent context stays clean on resume.
                history.append(Message(role="assistant", content=_strip_stats_footer(output)))
        # `thought` / `run` steps are never replayed (reasoning is re-derived).
    return history


class _StreamBridge:
    """Maps a cumulative `List[Message]` stream from `agent.run()` into
    Chainlit primitives (`cl.Message` for text, `cl.Step` for tool calls and
    reasoning).

    State is keyed by message index in the cumulative thread. Indices stay
    stable within an LLM call's streaming and new indices appear when the
    agent executes tools and continues. Each entry tracks how much of that
    message we've already pushed to the UI so we send only deltas.
    """

    # cl.Step.update() sends the FULL step dict (incl. cumulative reasoning/args)
    # over the socket + persists it. Calling it per chunk from a chatty LLM made
    # the UI lag many seconds behind the model. Throttle the per-chunk updates
    # to ~5/s; the in-memory step object is always current, and the close-time
    # update (via _close_open_thoughts / _close_tool_with_result) + finalize()
    # flush the final state with no throttle.
    _UPDATE_THROTTLE_S = 0.2
    # Server-side stream_token batching. The drain is fast (~66/sec) but Chainlit's
    # React frontend re-parses markdown per stream_token event — for long messages
    # that's ~300ms/event on the client. So 70 events/sec from the backend get
    # rendered at ~3/sec, you watch the catch-up after the LLM finishes. Solution:
    # accumulate per-token deltas and emit one stream_token per BATCH_INTERVAL
    # (or sooner if the unflushed delta gets large). Each batched event = one
    # client render, but a chunkier one — net renders go from 70/sec → ~12/sec
    # while the visible streaming experience stays smooth.
    _STREAM_TOKEN_BATCH_INTERVAL = 0.08   # ~12 emits/sec
    _STREAM_TOKEN_BATCH_MAX_CHARS = 200   # also flush if unflushed delta exceeds this

    def __init__(self):
        # Per-message-index UI state.
        # entry shape: {"kind": "text"|"tool"|"thought", "obj": cl.Message|cl.Step,
        #               "streamed": int (chars already pushed), "name": str}
        self._by_index: Dict[int, dict] = {}
        # Allow result messages to find their tool step by function_id when present.
        self._tool_by_function_id: Dict[str, "cl.Step"] = {}
        # key (matches _by_index key) → monotonic time of last awaited update()
        self._last_update: Dict = {}
        # Per-text-message: unflushed delta + last emit time. See _stream_text.
        self._text_pending: Dict[int, str] = {}
        self._text_last_emit: Dict[int, float] = {}
        # Tool steps whose result has already been applied. consume() re-scans
        # the full cumulative thread on every chunk, so without this a finished
        # tool result would be re-update()d (socket emit + DB persist of the
        # whole step) on every subsequent chunk — O(tool count) wasted writes
        # per chunk that bog the event loop down as a multi-tool turn grows.
        self._resolved_tool_steps: set = set()

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

    def last_text_message(self) -> Optional["cl.Message"]:
        """The most recently created assistant `cl.Message` (text bubble). Used
        by on_message to append the per-turn token-usage footer right before
        finalize persists everything."""
        last = None
        for state in self._by_index.values():
            if state.get("kind") == "text":
                last = state.get("obj")
        return last

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

    async def _close_open_thoughts(self) -> None:
        """Mark every still-open reasoning step as done so its spinner clears
        immediately, instead of all of them clearing together at the end of the
        run. Called whenever the model emits content or a tool call — that means
        the thinking up to that point is finished. We close *all* open thought
        steps rather than the one for the current message index, because
        qwen-agent sometimes emits the reasoning as its own assistant message,
        separate from the one carrying the tool call (so an index-keyed lookup
        misses it). `finalize()` is still the backstop for a turn that ends on
        a reasoning-only message."""
        for state in list(self._by_index.values()):
            if state.get("kind") != "thought":
                continue
            obj = state.get("obj")
            if obj is None or getattr(obj, "end", None):
                continue
            obj.end = utc_now()
            try:
                await obj.update()
            except Exception:  # noqa: BLE001
                logger.debug("StreamBridge: failed to close thought step", exc_info=True)

    async def _stream_text(self, i: int, text: str) -> None:
        await self._close_open_thoughts()
        state = self._by_index.get(i)
        if state is None or state["kind"] != "text":
            # New message — compute what we'd stream first. Don't create the
            # cl.Message / store state until we know there's real content.
            # (Models often emit leading newlines before a tool call; creating
            # the message for whitespace-only content leaves empty DB rows.)
            delta = text.lstrip("\n")
            if not delta:
                return  # whitespace-only — skip, no message created
            cl_msg = cl.Message(content="")
            # Send IMMEDIATELY to claim the message's position in the chat
            # history (relative to any cl.Step we've already sent). Without
            # this, the message renders only when streaming finishes — and
            # ends up out-of-order relative to tool Steps that were sent
            # mid-stream. Also: stream_token does NOT auto-persist; the
            # initial send() is required for the data layer to track it.
            await cl_msg.send()
            state = {"kind": "text", "obj": cl_msg, "streamed": len(text)}
            self._by_index[i] = state
            # Feed the stripped delta into the batcher for immediate emit.
            self._text_pending[i] = delta
            now = time.monotonic()
            if (now - self._text_last_emit.get(i, 0.0)) < self._STREAM_TOKEN_BATCH_INTERVAL and \
               len(self._text_pending[i]) < self._STREAM_TOKEN_BATCH_MAX_CHARS:
                return
            batched = self._text_pending[i]
            self._text_pending[i] = ""
            self._text_last_emit[i] = now
            await cl_msg.stream_token(batched)
            return
        if len(text) > state["streamed"]:
            delta = text[state["streamed"]:]
            state["streamed"] = len(text)
            # Batch into ~12 emits/sec (or sooner if the buffer is large) — the
            # client's per-event markdown re-parse can't keep up with 70/sec.
            self._text_pending[i] = self._text_pending.get(i, "") + delta
            now = time.monotonic()
            last = self._text_last_emit.get(i, 0.0)
            if (now - last) < self._STREAM_TOKEN_BATCH_INTERVAL and \
               len(self._text_pending[i]) < self._STREAM_TOKEN_BATCH_MAX_CHARS:
                return  # keep accumulating; finalize() will flush leftovers
            batched = self._text_pending[i]
            self._text_pending[i] = ""
            self._text_last_emit[i] = now
            await state["obj"].stream_token(batched)

    async def flush_pending_text(self) -> None:
        """Emit any text deltas the batcher is still holding back (small/fresh
        deltas accumulated in _stream_text). Must be called before the per-turn
        stats footer is appended to the last message — otherwise the footer is
        inserted ahead of the still-pending tail and ends up spliced into the
        middle of the model's final sentence."""
        for i, pending in list(self._text_pending.items()):
            if not pending:
                continue
            state = self._by_index.get(i)
            if state and state.get("kind") == "text":
                try:
                    await state["obj"].stream_token(pending)
                except Exception:  # noqa: BLE001
                    logger.debug("flush_pending_text: flushing pending text failed", exc_info=True)
            self._text_pending[i] = ""

    async def finalize(self) -> None:
        """End-of-run cleanup. Calls update() on each streamed element so the
        final content is persisted (Chainlit's stream_token does not persist
        on its own — explicit update is required after the last token), and
        sets `end` on any cl.Step that's still "running" so its spinner clears
        — covers thought steps (no natural close event) and tool steps whose
        result never came back."""
        # Flush any text deltas still batched (no-op if already flushed before
        # the footer was appended).
        await self.flush_pending_text()

        for state in self._by_index.values():
            obj = state.get("obj")
            if obj is None:
                continue
            # cl.Step → mark done if not already; cl.Message has no spinner concept.
            if state.get("kind") in ("tool", "thought") and not getattr(obj, "end", None):
                obj.end = utc_now()
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
            step.start = utc_now()  # mark "running" — finalize() sets `end` so the spinner clears
            await step.send()
            state = {"kind": "thought", "obj": step, "streamed": 0}
            self._by_index[key] = state
        if len(reasoning) > state["streamed"]:
            state["obj"].output = _cap_step_output(reasoning)  # cumulative; capped for the DOM
            state["streamed"] = len(reasoning)
            # Throttle the actual network round-trip — update_step sends the
            # entire cumulative `output` every call, so per-chunk updates flood
            # the socket. The close-time update flushes the final state.
            now = time.monotonic()
            if now - self._last_update.get(key, 0.0) >= self._UPDATE_THROTTLE_S:
                self._last_update[key] = now
                await state["obj"].update()

    async def _stream_tool_call(self, i: int, msg, function_call) -> None:
        await self._close_open_thoughts()
        name = getattr(function_call, "name", None) or "tool"
        args = getattr(function_call, "arguments", "") or ""
        function_id = self._function_id(msg)

        state = self._by_index.get(i)
        if state is None or state["kind"] != "tool":
            step = cl.Step(
                name=name, type="tool", show_input="json",
                parent_id=self._parent_id(),
            )
            step.start = utc_now()  # opens the step in "running" state (spinner)
            await step.send()
            state = {
                "kind": "tool", "obj": step, "streamed": 0,
                "name": name, "function_id": function_id,
            }
            self._by_index[i] = state
            if function_id:
                self._tool_by_function_id[function_id] = step

        # Stream growing arguments as the tool-call's input. Throttle the
        # update for the same reason as _stream_thought; close-time flushes.
        if len(args) > state["streamed"]:
            state["obj"].input = args
            state["streamed"] = len(args)
            now = time.monotonic()
            if now - self._last_update.get(i, 0.0) >= self._UPDATE_THROTTLE_S:
                self._last_update[i] = now
                await state["obj"].update()

    async def update_running_tool(self, text: str) -> None:
        """Live stdout from a still-running tool (code_interpreter) — show it on
        that tool's open cl.Step so a multi-minute run isn't a silent spinner.
        There's exactly one running tool at a time (the agent loop is sequential);
        the result message later replaces this with the final output via
        `_close_tool_with_result`."""
        target = None
        for state in self._by_index.values():
            if state.get("kind") == "tool":
                obj = state.get("obj")
                if obj is not None and not getattr(obj, "end", None):
                    target = obj
        if target is None:
            return
        target.output = _cap_step_output(text, keep="tail")
        try:
            await target.update()
        except Exception:  # noqa: BLE001
            logger.debug("StreamBridge: update_running_tool failed", exc_info=True)

    async def display_document(self, payload: dict) -> None:
        """Render a display_doc payload to the user — out-of-band from the agent's
        message stream, so the content never enters the agent's context. Sent with
        author='document' so on_chat_resume skips it on reload (shown to the user,
        never fed back to the model). Text persists as message content (no storage
        client needed); image/pdf/file ride as elements (persisted via
        LocalFsStorageClient → reloadable)."""
        kind = payload.get("kind")
        name = payload.get("name") or "document"
        path = payload.get("path")
        try:
            if kind == "text":
                await cl.Message(content=payload.get("text") or "(empty file)",
                                 author="document").send()
            elif kind == "image":
                await cl.Message(content=f"📄 {name}", author="document",
                                 elements=[cl.Image(path=path, name=name, display="inline")]).send()
            elif kind == "pdf":
                await cl.Message(content=f"📄 {name}", author="document",
                                 elements=[cl.Pdf(path=path, name=name, display="inline")]).send()
            else:
                await cl.Message(content=f"📄 {name}", author="document",
                                 elements=[cl.File(path=path, name=name)]).send()
        except Exception:  # noqa: BLE001 — a render failure shouldn't kill the turn
            logger.exception("display_document: failed to render %s (kind=%s)", name, kind)

    async def _close_tool_with_result(self, msg) -> None:
        function_id = self._function_id(msg)
        target_step = None
        if function_id:
            target_step = self._tool_by_function_id.get(function_id)
        if target_step is None:
            # Fallback: most recent matching-name tool step we haven't resolved
            # yet (skip already-resolved ones so repeat calls of the same-named
            # tool each pair with their own step).
            tool_name = getattr(msg, "name", None)
            for state in reversed(list(self._by_index.values())):
                if (state.get("kind") == "tool" and state.get("name") == tool_name
                        and id(state["obj"]) not in self._resolved_tool_steps):
                    target_step = state["obj"]
                    break
        if target_step is None:
            return
        # Apply the result exactly once. consume() re-scans the whole thread per
        # chunk; re-running update() here every time is what made long multi-tool
        # turns crawl to ~2 tok/s.
        if id(target_step) in self._resolved_tool_steps:
            return
        self._resolved_tool_steps.add(id(target_step))
        result = getattr(msg, "content", "")
        if not isinstance(result, str):
            result = str(result)
        target_step.output = _cap_step_output(result)
        # Mark the step done so its spinner clears immediately (not at end of run).
        if not getattr(target_step, "end", None):
            target_step.end = utc_now()
        try:
            await target_step.update()
        except Exception:  # noqa: BLE001
            logger.debug("StreamBridge: failed to update closed tool step", exc_info=True)


def _get_turn_lock() -> asyncio.Lock:
    """Per-session lock that serializes agent turns — both user-initiated
    (`on_message`) and notifier-initiated (synthetic system-event turns). Without
    it the two could interleave UI output on the same session. Stored lazily on
    cl.user_session so each session gets its own."""
    lock = cl.user_session.get("_turn_lock")
    if lock is None:
        lock = asyncio.Lock()
        cl.user_session.set("_turn_lock", lock)
    return lock


def _compaction_budgets():
    """Thresholds for the tier that will serve the NEXT turn.

    TRIGGER comes from the primary tier: compaction starts at 80% of its real
    window (PRIMARY_COMPACTION_TRIGGER_FRACTION), so the larger window is
    usable rather than capped at the smaller tier's threshold.

    TARGET deliberately does NOT scale with the window. Scaling it too would
    leave ~250k tokens in context after every compaction — a floor re-sent on
    every subsequent turn, minutes of prefill each. The bigger window buys
    HEADROOM, not a permanently larger prompt.

    Correctness comes from pinning: a history above SPILLABLE_CONTEXT_TOKENS
    (the secondary's hard budget) is routed to the primary and never offered
    to the smaller tier — see `_acquire_turn_slot(only=...)` in main.py.
    """
    from sandbox_agent.compaction.budget import Budgets, derive_budgets
    from sandbox_agent.config import (BACKGROUND_LLM_CFG,
                                      PRIMARY_COMPACTION_TRIGGER_FRACTION,
                                      PRIMARY_LLM_CFG)
    primary = derive_budgets(PRIMARY_LLM_CFG["context_window_tokens"],
                             PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"],
                             PRIMARY_COMPACTION_TRIGGER_FRACTION)
    secondary = derive_budgets(BACKGROUND_LLM_CFG["context_window_tokens"],
                               BACKGROUND_LLM_CFG["generate_cfg"]["max_input_tokens"])
    return Budgets(hard=primary.hard, trigger=primary.trigger,
                   target=min(primary.target, secondary.target))


def _persist_compaction(thread_id, history) -> None:
    """Compact the stored history in place, once it exceeds the trigger.

    Transactional: the ladder is pure. Nothing needs a private archive —
    chat.db already holds every step (Chainlit persists them during the turn,
    unconditionally, via @cl.data_layer), and that is exactly what
    `session_search` reads. Compaction shrinks what the agent CARRIES, not
    what it can REACH. `history` is mutated IN PLACE
    (slice-assign) because the list is aliased into cl.user_session and the
    notifier loop; rebinding here would silently drop anything appended
    while summarization was running.
    """
    from sandbox_agent.compaction.policy import compact_for_persistence
    from sandbox_agent.token_budget import estimate_messages_tokens

    try:
        budgets = _compaction_budgets()
        snapshot = list(history)
        if estimate_messages_tokens(snapshot) <= budgets.trigger:
            return

        result = compact_for_persistence(snapshot, target_tokens=budgets.target)
        if not result.changed:
            if result.error:
                logger.warning("Compaction made no progress: %s", result.error)
            return
        # Splice: anything the notifier appended while we were compacting
        # lives past the snapshot and must survive.
        appended_since = history[len(snapshot):]
        history[:] = result.messages + appended_since
        logger.info("Persisted compaction for %s: %d -> %d tokens (%s); "
                    "%d messages folded into the digest — their full text stays "
                    "in chat.db and is retrievable via session_search",
                    thread_id, result.before_tokens, result.after_tokens,
                    "+".join(result.levels) or "none", len(result.archived))
    except Exception:  # noqa: BLE001 — never break a turn over housekeeping
        logger.exception("Persisted compaction failed; history left intact")


async def _execute_agent_turn(agent, history: List[Message], *, log_user_msg: Optional[Message] = None) -> None:
    """Run one agent turn against `history` (last item = the triggering message —
    user message, or a synthesized [system: …] line from the notifier).

    Streams output into the active Chainlit session, persists everything,
    appends the agent's response to `history` (in place) and to
    cl.user_session[HISTORY_KEY], and (if log_user_msg given) appends the
    user-facing exchange to the daily markdown log. Caller must already hold
    `_get_turn_lock()` and have a valid cl.context set."""
    bridge = _StreamBridge()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    # Tag this turn so the Stop button (which cancels the on_message coroutine)
    # can halt the agent — see _run_agent_in_thread. cl.context.current_run is
    # set when called from on_message; for notifier-initiated turns it's None
    # and we fall back to a uuid (Stop doesn't reach notifier turns anyway).
    try:
        run_id = cl.context.current_run.id if cl.context.current_run else None
    except Exception:  # noqa: BLE001
        run_id = None
    run_id = run_id or f"chat-{uuid.uuid4().hex}"
    # Expose to @on_stop: Stop cancels whatever turn is active — including
    # notifier-initiated synthetic turns the coroutine-cancel path never saw.
    global _active_run_id
    _active_run_id = run_id

    # Capture the originating session/thread on the asyncio side (cl.context is
    # per-asyncio-Task — invalid on the worker thread). Passed into the worker
    # which stamps `chat_origin` so any task it schedules carries this origin.
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    try:
        if cl.context.session is not None:
            session_id = getattr(cl.context.session, "id", None)
            thread_id = getattr(cl.context.session, "thread_id", None)
    except Exception:  # noqa: BLE001
        pass

    # Capture frozen metadata on the asyncio side (cl.context is per-Task, not
    # available on the worker thread). Freeze once per conversation for KV cache
    # stability — first turn generates it, subsequent turns reuse the cached value.
    frozen_meta = cl.user_session.get("_frozen_metadata")
    if not frozen_meta:
        frozen_meta = session_metadata()
        cl.user_session.set("_frozen_metadata", frozen_meta)

    threading.Thread(
        target=_run_agent_in_thread,
        args=(agent, history, queue, loop, run_id, session_id, thread_id, frozen_meta),
        daemon=True,
        name="chainlit-agent-run",
    ).start()

    # held_back lets us coalesce queued items (each `chunk` and each
    # `tool_progress` is fully cumulative, so older ones are subsumed by the
    # latest); when we encounter a non-coalescable item while peeking we hold
    # it for the next iteration to keep FIFO ordering.
    final_response: List[Message] = []
    held_back: Optional[tuple] = None
    turn_usage: Optional[dict] = None  # populated by worker's ("usage_summary", …)
    try:
        while True:
            if held_back is not None:
                kind, payload = held_back
                held_back = None
            else:
                kind, payload = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                await bridge.finalize()  # persist whatever did stream before the error
                await cl.ErrorMessage(
                    content=f"Agent error: {type(payload).__name__}: {payload}"
                ).send()
                return
            if kind == "usage_summary":
                turn_usage = payload  # stash; applied just before finalize
                continue
            if kind == "display_doc":
                # display_doc tool — render the file to the user out-of-band.
                await bridge.display_document(payload)
                continue
            if kind == "compaction_notice":
                # Slow-compaction heads-up (display-only; never enters the
                # agent's context). ⏳ on start with a rough estimate, ✅ done.
                try:
                    if payload.get("phase") == "start":
                        await cl.Message(
                            author="background task",
                            content=(f"⏳ **Compacting conversation history** "
                                     f"(~{payload.get('est_tokens', 0):,} tokens, "
                                     f"≈{payload.get('est_minutes', 1)} min). "
                                     f"One-off housekeeping — your reply continues automatically."),
                        ).send()
                    elif payload.get("phase") == "done":
                        secs = int(payload.get("elapsed_s", 0))
                        pretty = f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"
                        await cl.Message(author="background task",
                                         content=f"✅ Compaction done in {pretty}.").send()
                except Exception:  # noqa: BLE001 — a notice must never break the turn
                    logger.debug("compaction notice display failed", exc_info=True)
                continue
            if kind == "tool_progress":
                # Live stdout from a running code_interpreter call. Drain any
                # later progress snapshots — only the most recent matters.
                while True:
                    try:
                        nk, np = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nk == "tool_progress":
                        payload = np
                        continue
                    held_back = (nk, np)
                    break
                await bridge.update_running_tool(payload)
                continue
            # kind == "chunk" — coalesce ahead. Each chunk is the full cumulative
            # thread state, so newer ones supersede older ones; the bridge's
            # delta computation (state["streamed"] char counters) handles a
            # "jumped forward" payload correctly. This stops the drain from
            # falling minutes behind a chatty model during streaming.
            while True:
                try:
                    nk, np = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if nk == "chunk":
                    payload = np
                    continue
                held_back = (nk, np)
                break
            final_response = payload
            await bridge.consume(payload)

        # Flush any text the batcher is still holding back BEFORE appending the
        # footer — otherwise the footer is spliced ahead of the pending tail and
        # lands in the middle of the model's last sentence.
        await bridge.flush_pending_text()

        # Append the context-usage footer to the last assistant text bubble
        # (right above the feedback buttons). finalize() below sends the
        # update. What we show is "how close any single call this turn got to
        # the model's context window" — NOT a billing sum. Per-call max prompt
        # is what trips the 256k cap; cumulative billable tokens aren't useful
        # against a local vLLM.
        if turn_usage and (turn_usage["max_prompt"] or turn_usage["completion_total"]):
            max_p = turn_usage["max_prompt"]
            chat_hi = max(int(cl.user_session.get("chat_max_prompt_tokens") or 0), max_p)
            cl.user_session.set("chat_max_prompt_tokens", chat_hi)
            last_msg = bridge.last_text_message()
            if last_msg is not None:
                c = turn_usage["completion_total"]
                n = turn_usage["calls"]
                pct = (chat_hi / _CONTEXT_WINDOW_TOKENS) * 100.0
                footer = (
                    f"\n\n---\n"
                    f"📊 _last turn: max ctx {max_p:,} / {c:,} out"
                    f"{f' · {n} LLM calls' if n > 1 else ''} · "
                    f"chat high-water: {chat_hi:,} ({pct:.0f}% of {_CONTEXT_WINDOW_TOKENS // 1000}k)"
                    f" · 📅 {format_datetime()}_"
                )
                # mutate in-memory content; bridge.finalize() will call update()
                # which sends the new step_dict (incl. content) over the socket.
                last_msg.content = (last_msg.content or "") + footer

        # Finalize: persist all streamed cl.Messages to the data layer. Without
        # this, stream_token tokens reach the UI but no chat.db row is written.
        await bridge.finalize()
    except asyncio.CancelledError:
        # Stop button (or a disconnect) cancels the on_message coroutine.
        # Propagate it to the agent: set the run's cancel flag (its loop raises
        # RunCancelled at the next step) and SIGKILL any subprocess it's
        # running. Without this the worker keeps grinding headless.
        cancellation.cancel(run_id)
        raise

    # Persist response into agent-side history for the next turn. We feed back
    # the full thread (assistant text + function calls + function results) so
    # the agent has continuity. Reasoning_content is intentionally NOT carried
    # forward — it inflates the prompt and the model re-derives it on demand.
    for msg in final_response:
        if msg.role in ("assistant", "function") and (
            (msg.content and (isinstance(msg.content, str) or isinstance(msg.content, list)))
            or getattr(msg, "function_call", None)
        ):
            history.append(msg)
    cl.user_session.set(HISTORY_KEY, history)
    # PERSISTED COMPACTION. Compaction used to run on a throwaway deepcopy
    # inside the agent, so it repeated every turn and regenerated a
    # different digest each time (the agent's memory silently changed
    # between turns, and the KV prefix never matched). Committing it here
    # makes it once-per-overflow and keeps the next turn's prefix stable.
    _persist_compaction(thread_id, history)
    # Persist the agent-side history sidecar so a reload/restart restores this
    # exact context (incl. tool results) instead of the text-only rebuild.
    from sandbox_agent.chat_history import save_history
    save_history(thread_id, history)

    if log_user_msg is not None:
        # Daily markdown log — only for user-initiated turns (synthetic
        # system-event turns aren't logged as "user said X").
        try:
            log_turn(log_user_msg, final_response)
        except Exception:  # noqa: BLE001 — never let logging kill the chat
            logger.exception("log_turn failed (chat already delivered)")


def _image_element_to_data_url(el) -> Optional[str]:
    """Read a Chainlit image element off disk (where Chainlit persists it) and
    return a base64 data URL. We use data URLs (not paths) because vLLM is on
    a different host — paths/URLs that only resolve in the agent container
    aren't reachable from the LLM box. Returns None if we can't recover the
    bytes."""
    path = getattr(el, "path", None)
    mime = (getattr(el, "mime", None)
            or (mimetypes.guess_type(path)[0] if path else None)
            or "image/png")
    data: Optional[bytes] = None
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            logger.debug("on_message: failed to read image at %s", path, exc_info=True)
    if data is None:
        raw = getattr(el, "content", None)
        if isinstance(raw, (bytes, bytearray)):
            data = bytes(raw)
        elif isinstance(raw, str):
            data = raw.encode()
    if data is None:
        url = getattr(el, "url", None)
        if url:
            return url   # let vLLM fetch it if it can
        return None
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _prepend_ts_to_agent_msg(msg: Message, sent_at: str) -> None:
    """Mutate an agent-side user Message in place to carry a `[sent_at] ` prefix
    so the agent sees per-message time. Handles plain-text and multimodal
    (ContentItem list) content. The displayed cl.Message is untouched — this
    only modifies the qwen-agent Message that's appended to HISTORY_KEY."""
    if not sent_at:
        return
    prefix = f"[{sent_at}] "
    if isinstance(msg.content, str):
        msg.content = prefix + msg.content
        return
    if isinstance(msg.content, list):
        for item in msg.content:
            if getattr(item, "text", None) is not None:
                item.text = prefix + item.text
                return
        # No text part — insert one at the front so the prefix is the first
        # thing the model reads on this turn.
        msg.content.insert(0, ContentItem(text=prefix.rstrip()))


def _build_user_message(message: cl.Message) -> Message:
    """Build a `qwen_agent.llm.schema.Message` from a Chainlit incoming message.
    Plain text → `content=str`. With any image attachment → `content=List
    [ContentItem]` (text part + one image part per uploaded image). qwen-agent's
    oai.py local-mod (`_multimodal_to_oai_dict`) preserves the parts on the
    wire so vision-capable vLLM models actually see them."""
    text = message.content or ""
    elements = getattr(message, "elements", None) or []

    def _is_image(el) -> bool:
        mime = (getattr(el, "mime", None) or "").lower()
        if mime.startswith("image/"):
            return True
        # cl.Image instances may not always carry mime; fall back to class name.
        return type(el).__name__.lower() == "image"

    def _is_file(el) -> bool:
        """Check if element is a non-image file (PDF, doc, etc.)."""
        name = type(el).__name__.lower()
        if name in ("pdf", "file"):
            return True
        mime = (getattr(el, "mime", None) or "").lower()
        if mime in ("application/pdf",) or not mime.startswith("image/"):
            return True
        return False

    image_elems = [el for el in elements if _is_image(el)]
    file_elems = [el for el in elements if _is_file(el)]
    has_media = image_elems or file_elems

    if not has_media:
        return Message(role="user", content=text)

    parts: List[ContentItem] = []
    if text:
        parts.append(ContentItem(text=text))
    for el in image_elems:
        data_url = _image_element_to_data_url(el)
        if data_url:
            parts.append(ContentItem(image=data_url))
    for el in file_elems:
        path = getattr(el, "path", None)
        if path and os.path.exists(path):
            parts.append(ContentItem(file=path))
    if not any(getattr(p, "image", None) for p in parts) and not any(getattr(p, "file", None) for p in parts):
        # All media reads failed — fall back to plain text so we don't send
        # an empty multimodal message.
        return Message(role="user", content=text)
    return Message(role="user", content=parts)


@cl.on_message
async def on_message(message: cl.Message):
    history: List[Message] = cl.user_session.get(HISTORY_KEY) or []
    # One-line diagnostic for the image-upload path — surfaces what Chainlit
    # actually hands us so we can tell if `message.elements` is populated or
    # if the file_refs payload never reached on_message. Remove once stable.
    _els = getattr(message, "elements", None) or []
    if _els:
        _summary = ", ".join(
            f"{type(e).__name__}(mime={getattr(e,'mime',None)!r},"
            f"path={'Y' if getattr(e,'path',None) and os.path.exists(getattr(e,'path','') or '') else 'N'},"
            f"bytes={len(getattr(e,'content',b'') or b'')})"
            for e in _els
        )
        logger.info("on_message: %d element(s) attached → %s", len(_els), _summary)
    else:
        logger.info("on_message: 0 elements attached (text-only)")
    user_msg = _build_user_message(message)
    if isinstance(user_msg.content, list):
        n_img = sum(1 for it in user_msg.content if getattr(it, "image", None))
        logger.info("on_message: built multimodal Message with %d image part(s)", n_img)

    # Hidden per-message timestamp (host_mlx-style separate field, not inline
    # in displayed text): persist `sent_at` on cl.Message.metadata so it lands
    # in chat.db's metadata JSONB column (survives reloads), and prepend it to
    # the AGENT-SIDE Message.content only. message.content stays untouched →
    # the chat bubble shows whatever the user typed, with no timestamp visible.
    sent_at = format_datetime()
    md = dict(getattr(message, "metadata", None) or {})
    md["sent_at"] = sent_at
    message.metadata = md
    try:
        await message.update()
    except Exception:  # noqa: BLE001
        logger.debug("on_message: persisting sent_at metadata failed", exc_info=True)
    _prepend_ts_to_agent_msg(user_msg, sent_at)

    history.append(user_msg)
    cl.user_session.set(HISTORY_KEY, history)

    agent = _get_agent()
    async with _get_turn_lock():
        await _execute_agent_turn(agent, history, log_user_msg=user_msg)
