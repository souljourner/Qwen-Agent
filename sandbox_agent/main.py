"""Main entry point for the sandboxed Qwen Agent.

Lane-based concurrency (OpenClaw pattern):
- Main lane: interactive UI (Gradio WebUI or REPL) using primary vLLM model
- Heartbeat lane: periodic checks using background Ollama model
- Cron lane: scheduled task execution using background Ollama model
"""

import concurrent.futures
import json
import logging
import os
import sys
import functools
import time
from threading import BoundedSemaphore, Lock, Thread
from typing import Iterator, List, Optional

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message

from sandbox_agent.config import (
    BACKGROUND_LLM_CFG,
    DATA_DIR,
    HEARTBEAT_INTERVAL_SECONDS,
    PRIMARY_LLM_CFG,
    PRIMARY_MODEL_CONCURRENCY,
    SECONDARY_MODEL_CONCURRENCY,
    SPILLABLE_CONTEXT_TOKENS,
    TOOL_LIST,
    load_system_message,
    session_metadata,
)
from sandbox_agent.activity_log import clear_state, get_recent_events, log_event, set_state
from sandbox_agent.model_tracker import (
    model_start, model_done, set_agent_status, clear_agent_status, set_current_tool, set_current_preview,
)
from sandbox_agent.chat_logger import log_background_task, log_turn
from sandbox_agent.daily_digest import add_digest_entry, cleanup_old_digests
from sandbox_agent.heartbeat.heartbeat_runner import HeartbeatRunner
from sandbox_agent.scheduler.task_queue import TaskQueue
from sandbox_agent.token_budget import compute_request_timeout, estimate_messages_tokens, trim_to_budget
from sandbox_agent import model_health

# Import tools to trigger @register_tool decorators
import sandbox_agent.tools.api_tools  # noqa: F401
import sandbox_agent.tools.self_edit_tools  # noqa: F401
import sandbox_agent.tools.code_interpreter  # noqa: F401
import sandbox_agent.tools.exec_tool  # noqa: F401
import sandbox_agent.chat_logger  # noqa: F401 (registers list_chat_logs, read_chat_log)
import sandbox_agent.tools.project_tools  # noqa: F401
import sandbox_agent.tools.notification_tools  # noqa: F401
import sandbox_agent.tools.display_tools  # noqa: F401
import sandbox_agent.pipeline.pipeline_tools  # noqa: F401
import sandbox_agent.scheduler.scheduler_tools  # noqa: F401
import sandbox_agent.tools.browser_tools  # noqa: F401
import sandbox_agent.tools.skill_tools  # noqa: F401
import sandbox_agent.tools.session_search_tools  # noqa: F401

logger = logging.getLogger(__name__)

# Concurrency gate for the primary vLLM model. The 27b-linux primary supports
# `PRIMARY_MODEL_CONCURRENCY` simultaneous requests, so we use a BoundedSemaphore
# rather than a Mutex: up to N callers (any mix of user chats and background
# tasks) can hold a slot at once. When all slots are taken, user chat falls
# through to the 397B backup (LockingAgent.run uses acquire(blocking=False));
# background tasks block until a slot frees up (run_on_best_available uses
# acquire(blocking=True)). BoundedSemaphore is preferred over plain Semaphore
# so an over-release (release without a matching acquire) raises immediately.
_primary_model_lock = BoundedSemaphore(PRIMARY_MODEL_CONCURRENCY)
_secondary_model_lock = BoundedSemaphore(SECONDARY_MODEL_CONCURRENCY)


def _acquire_turn_slot(blocking: bool, prefer: str = "primary",
                       poll_interval: float = 0.25, only: str = None):
    """Acquire one model-turn slot across the two tiers.

    Tries the `prefer` tier first, the other second (non-blocking each);
    when `blocking`, polls until either tier frees. `only` restricts to a
    single tier (pinned turns: big contexts must stay on the primary).
    Returns (tier_name, release) or
    None. `release` is idempotent — a double call must never trip
    BoundedSemaphore's over-release check mid-stream."""
    def _try_once():
        order = [("primary", _primary_model_lock), ("secondary", _secondary_model_lock)]
        if prefer == "secondary":
            order.reverse()
        if only:
            # Pinned turns bypass health entirely: pinning is a correctness
            # constraint (the history only fits there), health is a hint.
            order = [(t, s) for t, s in order if t == only]
        elif len(order) > 1:
            # Health-aware ORDERING only — an unhealthy tier stays in the list
            # and is still used when the healthy one has no free slot. If every
            # tier is unhealthy, healthy_subset returns all of them so we never
            # refuse to route.
            from sandbox_agent.model_health import healthy_subset
            ok = set(healthy_subset([t for t, _ in order]))
            order.sort(key=lambda item: item[0] not in ok)
        for tier, sem in order:
            if sem.acquire(blocking=False):
                released = [False]

                def _release(sem=sem, released=released):
                    if not released[0]:
                        released[0] = True
                        sem.release()
                return (tier, _release)
        return None

    grant = _try_once()
    while grant is None and blocking:
        time.sleep(poll_interval)
        grant = _try_once()
    return grant


# Lock for background work — ensures heartbeat and cron tasks run one at a time, never colliding.
_background_work_lock = Lock()

# Stuck-task detection uses progress signals (activity-log growth + model_tracker
# state) rather than a fixed wall-clock ceiling. A legitimate long run emits tool
# calls and model-start/done transitions continuously; a wedged one (e.g., openai
# retry hanging on a dead stream) emits nothing. If no progress signal fires
# for STUCK_NO_PROGRESS_SECONDS, the cron loop abandons the worker, marks the
# task failed so it retries with backoff, and keeps polling. Python can't kill
# the abandoned thread, but at least the queue no longer freezes behind it.
STUCK_CHECK_INTERVAL_SECONDS = 30
STUCK_NO_PROGRESS_SECONDS = int(os.environ.get("STUCK_NO_PROGRESS_SECONDS", 10 * 60))
# Hard ceiling on the "model busy" grace: a wedged task that keeps a model slot
# marked busy would otherwise never be abandoned and stalls the queue forever.
STUCK_HARD_ABANDON_SECONDS = int(os.environ.get("STUCK_HARD_ABANDON_SECONDS", 2 * 60 * 60))


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
        set_current_tool(tool_name)
        log_event("tool_call", tool_name=tool_name, tool_args=args_preview)
        result = original_call_tool(tool_name, tool_args, **kwargs)
        # Skills auto-inject: first use of a trigger tool (browser_*) in a
        # conversation prepends the matching guide to the result. Dedup via
        # the [skill:NAME] marker in history; never touches the system prompt.
        from sandbox_agent.tools.skill_tools import maybe_inject_skill
        result = maybe_inject_skill(tool_name, result, kwargs.get("messages") or [])
        result_preview = str(result)[:200]
        logger.info(f"Tool result: {tool_name} -> {result_preview}")
        log_event("tool_result", tool_name=tool_name, tool_result=result_preview)
        set_state(current_tool=None)
        set_current_tool(None)
        return result

    agent._call_tool = _logged_call_tool

    # Mid-loop compaction: FnCallAgent._run re-compacts before EVERY LLM call
    # (system + task message pinned verbatim). Complements the entry-time
    # compaction below, which alone let tool results overflow the context.
    from sandbox_agent.compaction import compact_midrun
    _cfg_ctx = (llm_cfg or {}).get("context_window_tokens")
    agent._precall_compact = functools.partial(compact_midrun, context_tokens=_cfg_ctx)

    # Wrap _run to compact context before the FnCallAgent tool-call loop starts
    original_run = agent._run

    def _compacting_run(messages, **kwargs):
        from sandbox_agent import cancellation
        from sandbox_agent.compaction import maybe_compact
        # allow_llm=False: this copy is THROWAWAY (the agent deep-copies
        # per run), so summarizing here would duplicate the end-of-turn
        # persisted compaction and discard the result. Deterministic
        # tiers only; _persist_compaction does the one real summarize.
        messages = maybe_compact(messages, context_tokens=_cfg_ctx,
                                 allow_llm=False)
        # cancellation.guard raises RunCancelled at each yield point if this
        # run (see cancellation.begin_run) has been cancelled — between tool
        # calls and between streamed chunks. A tool call wedged inside a
        # subprocess is unstuck separately (cancel() SIGKILLs the child),
        # which then produces the next yield this guard checks.
        yield from cancellation.guard(original_run(messages, **kwargs))

    agent._run = _compacting_run
    return agent


def _stream_tap(response, label: str) -> None:
    """Push the tail of the latest streamed assistant text to the dashboard.

    Throttled inside set_current_preview — safe to call on every yield."""
    if not response:
        return
    for msg in reversed(response):
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            reasoning = msg.get("reasoning_content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            reasoning = getattr(msg, "reasoning_content", None)
        if role != "assistant":
            continue
        text = content if (content and str(content).strip()) else reasoning
        if not text:
            return
        tail = str(text).strip()[-300:]
        set_current_preview(f"{label} · {tail}" if label else tail)
        return


def _response_has_assistant_content(response) -> bool:
    """Qwen-Agent preserves input type: if conversation history is plain dicts,
    yielded items are dicts; if Message objects, they're Messages. Handle both
    so a dict-history UI path doesn't raise AttributeError mid-stream.

    A response is considered to have content if ANY assistant message has:
      - non-empty `content` (a text answer), OR
      - a `function_call` (the model issued a tool call — empty text is normal there), OR
      - non-empty `reasoning_content` (the model thought, even if no final answer)
    The previous version only checked `content`, which misclassified
    tool-call-only and reasoning-only responses as silent failures."""
    if not response:
        return False
    for msg in response:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            fc = msg.get("function_call") or msg.get("tool_calls")
            rc = msg.get("reasoning_content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            fc = getattr(msg, "function_call", None) or getattr(msg, "tool_calls", None)
            rc = getattr(msg, "reasoning_content", None)
        if role != "assistant":
            continue
        if content and str(content).strip():
            return True
        if fc:
            return True
        if rc and str(rc).strip():
            return True
    return False


def _dump_response_for_debug(response, context: str) -> str:
    """Return a compact summary of `response` for debug logging when we
    misdetect it as empty. Lists each message's role + which fields are
    populated + first 200 chars of any text content."""
    if not response:
        return f"[{context}] response is empty/None"
    lines = [f"[{context}] response has {len(response)} messages:"]
    for i, msg in enumerate(response):
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            fc = msg.get("function_call") or msg.get("tool_calls")
            rc = msg.get("reasoning_content")
            name = msg.get("name")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            fc = getattr(msg, "function_call", None) or getattr(msg, "tool_calls", None)
            rc = getattr(msg, "reasoning_content", None)
            name = getattr(msg, "name", None)
        content_str = str(content) if content else ""
        rc_str = str(rc) if rc else ""
        lines.append(
            f"  [{i}] role={role!r} name={name!r} "
            f"content_len={len(content_str)} reasoning_len={len(rc_str)} "
            f"function_call={'set' if fc else 'none'}"
        )
        if content_str:
            lines.append(f"      content: {content_str[:200]!r}")
        if rc_str:
            lines.append(f"      reasoning: {rc_str[:200]!r}")
        if fc:
            lines.append(f"      function_call: {str(fc)[:200]!r}")
    return "\n".join(lines)


class LockingAgent(Assistant):
    """Wraps two model-bound Assistants with tiered turn routing.

    Chat turns prefer the primary tier (qwen3.6-27b on the Mac, 2 slots) and spill to
    the secondary (qwen3.6-27b-linux, 10 slots). When every slot is busy
    the turn WAITS with a visible notice — no more ungated pile-on. Slots are
    acquired per TURN via `_acquire_turn_slot` and held for the whole stream.
    """

    def __init__(self, inner: Assistant, backup: Assistant, lock=None):
        # `lock` accepted-and-ignored for signature compat; slots come from
        # the module-level tier semaphores via _acquire_turn_slot.
        self._inner = inner      # PRIMARY tier agent (qwen3.6-27b, Mac)
        self._backup = backup    # SECONDARY tier agent (qwen3.6-27b-linux)

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

        # Routing priority:
        # 1. History too big for the secondary (> SPILLABLE_CONTEXT_TOKENS) →
        #    PRIMARY only. With both tiers budgeted equally this cannot fire;
        #    it re-activates if PRIMARY_CONTEXT_TOKENS is ever raised.
        # 2. Otherwise: prefer primary, spill to secondary.
        # All-slots-busy in any mode → WAIT with a visible notice.
        # NOTE: there is deliberately no vision pin. Both tiers run
        # qwen3.6-27b (Mac and Linux) and both are multimodal, so pinning
        # image turns to one tier would only waste the other's capacity.
        msg_objs = msg_list if messages else []
        if msg_objs and estimate_messages_tokens(msg_objs) > SPILLABLE_CONTEXT_TOKENS:
            pin_only, route_tag = "primary", "[primary pinned]"
        else:
            pin_only, route_tag = None, None
        self._pin_only = pin_only  # retry path must honor the same restriction

        grant = _acquire_turn_slot(blocking=False, only=pin_only)
        waited_note = ""
        if grant is None:
            wait_start = time.monotonic()
            yield [Message(role="assistant",
                           content="⚠️ All model slots busy — waiting for a free slot…")]
            grant = _acquire_turn_slot(blocking=True, only=pin_only)
            waited_note = f" [waited {time.monotonic() - wait_start:.0f}s]"
        tier, release_slot = grant
        if tier == "primary":
            agent = self._inner
            model_name = PRIMARY_LLM_CFG["model"]
            tag = route_tag or "[primary]"
            log_event("chat_start", detail=f"{tag}{waited_note} {str(user_msg.content)[:180]}" if user_msg else f"{tag}{waited_note}")
        else:
            agent = self._backup
            model_name = BACKGROUND_LLM_CFG["model"]
            tag = route_tag or "[secondary spill]"
            log_event("chat_start", detail=f"{tag}{waited_note} {str(user_msg.content)[:160]}" if user_msg else f"{tag}{waited_note}")
            if not route_tag:
                logger.info(f"Chat routed to {model_name} (primary slots all in use)")

        user_preview = str(user_msg.content)[:100] if user_msg else "chat"
        set_state(status="chatting", model_in_use=model_name)
        set_agent_status(status="chatting", current_task=f"User chat: {user_preview}")
        model_start(model_name, f"User chat: {user_preview}")
        try:
            response = []
            for response in agent.run(*args, **kwargs):
                _stream_tap(response, f"chat: {user_preview}")
                yield response
        finally:
            set_current_preview(None)
            release_slot()
            model_done(model_name)

        # Detect empty completion — model returned 200 but produced no content.
        # With reasoning-parser enabled on both models this can happen when
        # thinking burns the token budget; both models can hit it for the same
        # prompt, so retrying without telling the user would silently mask the
        # issue and replace a 27B answer with a 397B answer of unknown quality.
        # Surface the retry visibly so the user/agent sees what happened.
        has_content = _response_has_assistant_content(response)

        if not has_content:
            logger.warning(f"Empty completion from {model_name} for: {user_preview}")
            # Dump the actual response so we can see what made the model
            # appear "empty" — content-only? function_call missed? reasoning
            # only? Helps separate real model failures from detection bugs.
            logger.warning(_dump_response_for_debug(response, f"{model_name} empty"))
            log_event("silent_failure", detail=f"{model_name} returned empty response", model=model_name)

            # The original slot was already released (finally above ran when
            # the stream ended) — acquire fresh, preferring the OTHER tier.
            # Pinned turns keep their restriction: a big-context retry must
            # not land on a tier that cannot fit it.
            # If the preferred tier is full, the retry may legitimately land
            # on the same model again; caps are never exceeded.
            pin_only = getattr(self, "_pin_only", None)
            other = pin_only or ("secondary" if agent is self._inner else "primary")
            retry_grant = (_acquire_turn_slot(blocking=False, prefer=other, only=pin_only)
                           or _acquire_turn_slot(blocking=True, prefer=other, only=pin_only))
            retry_tier, retry_release = retry_grant
            retry_agent = self._inner if retry_tier == "primary" else self._backup
            retry_model = PRIMARY_LLM_CFG["model"] if retry_tier == "primary" else BACKGROUND_LLM_CFG["model"]
            logger.info(f"Retrying on {retry_model}")

            # Visible interim notice so the user sees the fallback rather than
            # silently getting a different model's answer.
            notice = Message(
                role="assistant",
                content=f"⚠️ {model_name} returned empty content — retrying on {retry_model}.",
            )
            yield [notice]

            model_start(retry_model, f"Retry: {user_preview}")
            try:
                response = []
                for response in retry_agent.run(*args, **kwargs):
                    yield response
            except Exception as e:
                # Retry blew up (timeout, HTTP error, etc.). Don't let it
                # propagate up — that would lose conversation state in
                # web_ui.py's history extension. Yield a visible error and
                # continue cleanly.
                logger.exception(f"Retry on {retry_model} raised")
                log_event("silent_failure_retry", detail=f"{retry_model}: {type(e).__name__}: {str(e)[:200]}", model=retry_model)
                err = Message(
                    role="assistant",
                    content=f"⚠️ Both models failed. {model_name}: empty completion. {retry_model}: {type(e).__name__}: {str(e)[:200]}",
                )
                yield [err]
                response = [err]
            finally:
                retry_release()
                model_done(retry_model)

            # If retry came back but also empty, surface that visibly too.
            if not _response_has_assistant_content(response):
                logger.error(f"Both models returned empty content for: {user_preview}")
                log_event("silent_failure_both", detail="Both models returned empty response")
                err = Message(
                    role="assistant",
                    content=(
                        f"⚠️ Both {model_name} and {retry_model} returned empty content. "
                        "This usually means thinking-mode consumed the token budget before producing "
                        "an answer. Try rephrasing or breaking the request into smaller steps."
                    ),
                )
                yield [err]
                response = [err]

        log_event("chat_complete")
        clear_state()
        clear_agent_status()

        if user_msg and response:
            try:
                log_turn(user_msg, response)
            except Exception:
                logger.debug("log_turn failed", exc_info=True)

    def run_nonstream(self, *args, **kwargs):
        messages = args[0] if args else kwargs.get("messages", [])
        pin_only = None
        if messages:
            msg_list = [m if isinstance(m, Message) else Message(**m) for m in messages]
            if estimate_messages_tokens(msg_list) > SPILLABLE_CONTEXT_TOKENS:
                pin_only = "primary"
        grant = (_acquire_turn_slot(blocking=False, only=pin_only)
                 or _acquire_turn_slot(blocking=True, only=pin_only))
        tier, release_slot = grant
        agent = self._inner if tier == "primary" else self._backup
        try:
            return agent.run_nonstream(*args, **kwargs)
        finally:
            release_slot()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _summarize_task_result(task_name: str, result_text: str, tool_calls: List[dict], project: str = "") -> str:
    """Use Ollama to write a concise summary of a completed task for the daily digest."""
    try:
        import requests as _req
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
            f"{BACKGROUND_LLM_CFG['model_server']}/chat/completions",
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
            },
            timeout=60,
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"].get("content", "")
        return summary if summary else result_text[:500]
    except Exception as e:
        logger.warning(f"Failed to summarize task result: {e}")
        return result_text[:500] if result_text else "Completed"


def _has_assistant_content(response) -> bool:
    return any(
        msg.role == "assistant" and msg.content and str(msg.content).strip()
        for msg in response
    ) if response else False


def _run_on_tier(tier: str, system_message: str, messages: List[Message],
                 timeout: int, task_label: str) -> List[Message]:
    """One attempt on one tier. Caller owns the slot; this owns the status."""
    tier_cfg = BACKGROUND_LLM_CFG if tier == "secondary" else PRIMARY_LLM_CFG
    logger.info(f"Background task using {tier} tier ({tier_cfg['model']}, timeout={timeout}s)")
    set_state(model_in_use=tier_cfg["model"])
    set_agent_status(status="background", current_task=task_label)
    log_event("model_select", detail=f"{tier} (background)", model=tier_cfg["model"])
    model_start(tier_cfg["model"], task_label)
    try:
        agent = create_agent(system_message, llm_cfg=tier_cfg)
        agent.llm.generate_cfg["request_timeout"] = timeout
        # Compact for THIS tier's window (see _compacting_run in create_agent;
        # this outer pass also makes the timeout computation accurate).
        from sandbox_agent.compaction import maybe_compact
        msgs = maybe_compact(messages, context_tokens=tier_cfg.get("context_window_tokens"))
        response: List[Message] = []
        for response in agent.run(messages=msgs):
            _stream_tap(response, task_label)
        return response
    finally:
        set_current_preview(None)
        model_done(tier_cfg["model"])
        clear_agent_status()


def run_on_best_available(system_message: str, messages: List[Message], task_label: str = "background task") -> List[Message]:
    """Run a background agent session on a tier slot, failing over across tiers.

    Background work PREFERS the secondary tier (qwen3.6-27b-linux, 10 slots):
    a long cron task must not eat one of chat's 2 primary slots. It spills
    INTO the primary tier only when all secondary slots are busy, and blocks
    when all 12 slots are held.

    FAILOVER: if the attempt raises (model down, connection refused, timeout)
    or returns nothing, it retries once on the OTHER tier. Slot acquisition is
    a local semaphore with no health awareness, so it will happily hand out a
    slot for a model that is down; without this, a single dead tier failed
    every cron run, heartbeat, pipeline stage and evaluation while the other
    tier sat idle. Pipeline stages have limited attempts, so an outage could
    burn a strategy's attempts and reject it for infrastructure reasons.

    The chat path (LockingAgent) and llm_call (llm_client._resolve_chain)
    both already failed over; this function was the only one that did not,
    and it is the one that runs all unattended work.

    If the preferred tier is full the retry may legitimately land on the same
    model again — slot caps are never exceeded.
    """
    timeout = compute_request_timeout(messages)

    tier, release_slot = _acquire_turn_slot(blocking=True, prefer="secondary")
    first_failure = None
    try:
        response = _run_on_tier(tier, system_message, messages, timeout, task_label)
        if _has_assistant_content(response):
            model_health.record_success(tier)
            return response
        model_health.record_failure(tier)
        first_failure = "empty response"
        log_event("silent_failure", detail=f"Empty response for: {task_label}",
                  model=(BACKGROUND_LLM_CFG if tier == "secondary" else PRIMARY_LLM_CFG)["model"])
    except Exception as e:  # noqa: BLE001 — any failure is worth a cross-tier retry
        model_health.record_failure(tier)
        first_failure = f"{type(e).__name__}: {e}"
        logger.warning("Background task %s failed on %s tier (%s) — retrying on the other tier",
                       task_label, tier, first_failure)
    finally:
        release_slot()

    other = "primary" if tier == "secondary" else "secondary"
    logger.warning("Retrying background task %s on the %s tier after: %s",
                   task_label, other, first_failure)
    retry_tier, retry_release = _acquire_turn_slot(blocking=True, prefer=other)
    try:
        try:
            response = _run_on_tier(retry_tier, system_message, messages, timeout,
                                    f"retry: {task_label}")
        except Exception:
            model_health.record_failure(retry_tier)
            raise
        if _has_assistant_content(response):
            model_health.record_success(retry_tier)
        else:
            model_health.record_failure(retry_tier)
            logger.error("Background task %s produced nothing on both %s and %s tiers",
                         task_label, tier, retry_tier)
            log_event("silent_failure_retry",
                      detail=f"Retry also empty for: {task_label} ({tier} -> {retry_tier})")
        return response
    finally:
        retry_release()


def _run_cron_task(task, system_message: str, task_queue: TaskQueue, events_before: int) -> None:
    """Execute one cron task. Runs inside a worker thread so the cron loop can
    enforce a wall-clock timeout on it; wrapped in a cancellation run so
    `cancel_task` (or the stuck-detector) can interrupt it mid-flight."""
    from sandbox_agent import cancellation, chat_origin
    # Capture the generation before any work: if the stuck-detector abandons
    # this worker it bumps the task's generation and our failure update below
    # becomes a no-op (see task_queue.update_task expected_generation).
    generation = getattr(task, "run_generation", 0)
    # Carry the originating chat (if any) forward so sub-tasks scheduled
    # during this cron run inherit the same origin and route back to the
    # original chat too. Cleared on finally so it doesn't leak to the next
    # task on this pooled worker thread.
    chat_origin.set_current_origin(getattr(task, "origin", None))
    try:
        try:
            with cancellation.begin_run(task.id):
                _execute_cron_task(task, system_message, task_queue, events_before)
        except cancellation.RunCancelled:
            # `cancel_task` already removed the task from the queue (or the
            # stuck-detector marked it failed); the worker just unwinds.
            logger.info(f"Cron: task [{task.id}] run cancelled mid-flight")
            log_event("cron_cancelled", task_id=task.id, task_name=task.name)
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
            task_queue.update_task(task.id, status="failed", last_error=str(e)[:500],
                                   expected_generation=generation)
            try:
                from sandbox_agent import task_notify
                task_notify.notify_task_done(
                    task.id, task.name, f"FAILED: {str(e)[:400]}",
                    source="cron", ok=False, origin=getattr(task, "origin", None),
                )
            except Exception:  # noqa: BLE001
                pass
            clear_state()
    finally:
        chat_origin.set_current_origin(None)


def _is_empty_result(text: str) -> bool:
    """True when a task run produced no usable output (empty/whitespace) —
    such runs are marked failed so the backoff/retry path re-attempts them
    instead of recording a silent no-op as success. Deliberately NO length
    floor: short results like 'DONE' are legitimate."""
    return not (text or "").strip()


def _execute_cron_task(task, system_message: str, task_queue: TaskQueue, events_before: int) -> None:
    # Captured at worker start: if the stuck-detector abandons this worker it
    # bumps the task's generation, and our status updates below become no-ops
    # (update_task ignores stale generations) — prevents a zombie worker from
    # double-completing a re-queued task.
    generation = task.run_generation

    # Pipeline tasks get special handling
    if task.name.startswith("pipeline:"):
        from sandbox_agent.pipeline.stage_runner import run_pipeline_stage
        result_text = run_pipeline_stage(task, system_message)
        task_queue.update_task(task.id, status="completed", result=result_text[:1000],
                               expected_generation=generation)
        log_background_task(task.name, task.id, result_text[:1000])
        log_event("cron_complete", task_id=task.id, task_name=task.name)
        add_digest_entry(
            project=task.project,
            task_name=task.name,
            summary=result_text[:500],
            source="pipeline",
        )
        try:
            from sandbox_agent import task_notify
            task_notify.notify_task_done(
                task.id, task.name, result_text,
                source="pipeline", origin=getattr(task, "origin", None),
            )
        except Exception:  # noqa: BLE001
            pass
        logger.info(f"Cron: pipeline task [{task.id}] completed")
        clear_state()
        return

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
    response = run_on_best_available(system_message, messages, task_label=f"Cron: {task.name}")
    result_text = ""
    for msg in response:
        if msg.role == "assistant" and isinstance(msg.content, str):
            result_text += msg.content
    logger.info(f"Cron: task [{task.id}] result ({len(result_text)} chars): {result_text[:200]}...")
    if _is_empty_result(result_text):
        # No usable output — count it as a failure so the backoff path retries,
        # instead of recording a silent no-op as success (and, for recurring
        # tasks, skipping straight to the next window).
        logger.warning(f"Cron: task [{task.id}] produced empty output — marking failed for retry")
        task_queue.update_task(task.id, status="failed",
                               last_error="Task run produced empty output",
                               expected_generation=generation)
        log_event("cron_empty_result", task_id=task.id, task_name=task.name)
        clear_state()
        return
    task_queue.update_task(task.id, status="completed", result=result_text[:1000],
                           expected_generation=generation)
    log_background_task(task.name, task.id, result_text[:1000])
    log_event("cron_complete", task_id=task.id, task_name=task.name)
    all_events = get_recent_events(500)
    task_tool_calls = [e for e in all_events[events_before:] if e.get("type") == "tool_call"]
    digest_summary = _summarize_task_result(task.name, result_text, task_tool_calls, task.project or "")
    add_digest_entry(
        project=task.project,
        task_name=task.name,
        summary=digest_summary,
        source="cron",
    )
    try:
        from sandbox_agent import task_notify
        task_notify.notify_task_done(
            task.id, task.name, digest_summary or result_text,
            source="cron", origin=getattr(task, "origin", None),
        )
    except Exception:  # noqa: BLE001
        pass
    logger.info(f"Cron: task [{task.id}] completed")
    clear_state()


def _is_progressing(events_baseline: int, last_progress_at: "datetime") -> "tuple[bool, datetime]":
    """Progress signal: did the task emit anything since we last looked?

    Signals considered "progress":
      - activity.jsonl grew (tool_call / tool_result / model_select / ...)
      - model_tracker shows any model 'busy since T' with T newer than our
        last-progress timestamp (i.e., a model_start fired)
      - code_interpreter / exec streamed a new line to the dashboard preview
        (so a long ML training that just print()s epoch counters keeps the
        stuck-detector at bay without firing activity events)

    Returns (progressed, new_last_progress_at). When nothing moved, the second
    element is the same timestamp that was passed in."""
    from datetime import datetime as _dt

    from sandbox_agent.model_tracker import get_last_preview_at, read_status_from_file

    progressed = False
    new_mark = last_progress_at

    current_events = len(get_recent_events(500))
    if current_events > events_baseline:
        progressed = True
        new_mark = _dt.now()

    status = read_status_from_file()
    for info in (status.get("models") or {}).values():
        since = info.get("since")
        if not since:
            continue
        try:
            t = _dt.fromisoformat(since)
        except ValueError:
            continue
        if t > new_mark:
            progressed = True
            new_mark = t

    preview_at = get_last_preview_at()
    if preview_at is not None and preview_at > new_mark:
        progressed = True
        new_mark = preview_at

    return progressed, new_mark


def _model_is_actually_idle() -> bool:
    """True iff no model in model_tracker is currently marked 'busy'. When a
    task is supposedly running but no model claims to be working, the worker
    is almost certainly wedged in our own code (e.g., openai retry loop) —
    call this before declaring stuck so we don't kill a legitimate inference."""
    from sandbox_agent.model_tracker import read_status_from_file
    status = read_status_from_file()
    models = (status.get("models") or {}).values()
    if not models:
        return True
    return all(info.get("status") != "busy" for info in models)


def cron_loop(system_message: str, task_queue: TaskQueue, poll_interval: int = 60) -> None:
    """Background loop that checks for due scheduled tasks and runs them.

    Each task runs on a single-worker executor so the cron thread itself never
    blocks. While the worker runs, the loop polls model_tracker + activity
    log for progress signals. When none fires for STUCK_NO_PROGRESS_SECONDS
    AND the task's model is idle (or no model is busy at all), the task is
    abandoned, marked failed (which re-queues it via the existing backoff
    path), and the queue keeps moving."""
    logger.info(
        f"Cron loop started (poll interval: {poll_interval}s, "
        f"stuck threshold: {STUCK_NO_PROGRESS_SECONDS}s no-progress)"
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cron-task")
    while True:
        try:
            due_tasks = task_queue.get_due_tasks()
            for task in due_tasks:
                logger.info(f"Cron: waiting for background work lock for [{task.id}] {task.name}")
                with _background_work_lock:
                    logger.info(f"Cron: executing task [{task.id}] {task.name}")
                    set_state(status="cron_task", current_task=f"[{task.id}] {task.name}")
                    set_agent_status(status="cron_task", current_task=task.name)
                    log_event("cron_start", task_id=task.id, task_name=task.name)
                    task_queue.update_task(task.id, status="running")
                    events_baseline = len(get_recent_events(500))

                    from datetime import datetime as _dt
                    last_progress_at = _dt.now()
                    future = executor.submit(_run_cron_task, task, system_message, task_queue, events_baseline)

                    abandoned = False
                    while True:
                        try:
                            future.result(timeout=STUCK_CHECK_INTERVAL_SECONDS)
                            break  # task finished (success or internal exception)
                        except concurrent.futures.TimeoutError:
                            pass

                        progressed, last_progress_at = _is_progressing(events_baseline, last_progress_at)
                        if progressed:
                            events_baseline = len(get_recent_events(500))
                            continue

                        idle_seconds = (_dt.now() - last_progress_at).total_seconds()
                        if idle_seconds < STUCK_NO_PROGRESS_SECONDS:
                            continue

                        # Only declare stuck if the model is also idle — a
                        # long legitimate generation keeps a model busy with
                        # no intermediate activity events, and we don't want
                        # to kill it just because it's quiet. BUT: a wedged
                        # task can hold a model slot "busy" forever, which
                        # would stall the queue indefinitely — so past the
                        # hard ceiling we abandon regardless of model state.
                        if not _model_is_actually_idle() and idle_seconds < STUCK_HARD_ABANDON_SECONDS:
                            # Busy model but no activity — could still be a
                            # streaming generation. Extend the grace window
                            # by not abandoning yet; keep polling.
                            continue

                        logger.error(
                            f"Cron: task [{task.id}] {task.name} appears stuck — "
                            f"no progress for {int(idle_seconds)}s and all models idle. "
                            f"Abandoning worker and marking failed."
                        )
                        log_event("cron_stuck", task_id=task.id, task_name=task.name)
                        # Signal the worker to unwind and SIGKILL any subprocess
                        # it's wedged in, so the abandoned thread actually exits
                        # instead of zombie-ing (possibly still holding a model
                        # slot). The status stays "failed" — the worker's
                        # RunCancelled handler doesn't touch it — so the task is
                        # re-queued via the existing backoff path.
                        try:
                            from sandbox_agent import cancellation
                            cancellation.cancel(task.id)
                        except Exception:  # noqa: BLE001
                            logger.debug("stuck-detector: cancellation.cancel raised", exc_info=True)
                        # Invalidate the abandoned worker FIRST so that if it
                        # is a zombie that later finishes, its completed/failed
                        # update is ignored (stale generation) and can't
                        # double-run or flip the re-queued task's state.
                        task_queue.bump_generation(task.id)
                        task_queue.update_task(
                            task.id, status="failed",
                            last_error=(
                                f"Stuck: no activity for {int(idle_seconds)}s; "
                                f"worker abandoned"
                            ),
                        )
                        executor.shutdown(wait=False)
                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cron-task")
                        abandoned = True
                        break

                    if abandoned:
                        clear_state()
        except Exception:
            logger.exception("Cron loop error")
        time.sleep(poll_interval)


def _start_background_lanes(system_message: str, task_queue: TaskQueue) -> None:
    """Start heartbeat and cron background threads."""
    def bg_runner(messages: List[Message]) -> List[Message]:
        return run_on_best_available(system_message, messages, task_label="Heartbeat check")

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


_BOOTSTRAPPED = False
_BOOTSTRAP_TASK_QUEUE: Optional[TaskQueue] = None


def bootstrap_background(status_server_port: int = 7861) -> TaskQueue:
    """One-time bootstrap of all background machinery for non-Gradio entrypoints
    (Chainlit, REPL with-bg, tests). Idempotent — repeated calls are no-ops.

    Sets up, in order:
      1. DATA_DIR + git repo (idempotent)
      2. Pipeline lock + stuck-stage cleanup
      3. System message (loaded into config cache, available via load_system_message)
      4. Shared TaskQueue + orphan task recovery + scheduler-tools wiring
      5. Pipeline orphan-stage rescheduling
      6. LLM bridge (HTTP server on localhost; sets ci._llm_init_code)
      7. Status server (port `status_server_port`, separate process via multiprocessing)
      8. Background lanes (heartbeat thread + cron thread)

    Returns the shared TaskQueue (some callers want a handle to it).
    Pass `status_server_port=0` to skip starting the status server (useful in
    dev where you might already have one running on the standard port).
    """
    global _BOOTSTRAPPED, _BOOTSTRAP_TASK_QUEUE
    if _BOOTSTRAPPED:
        assert _BOOTSTRAP_TASK_QUEUE is not None
        return _BOOTSTRAP_TASK_QUEUE

    os.makedirs(DATA_DIR, exist_ok=True)
    # Don't bother with git repo init here — main.main() does that for the CLI
    # path, and Chainlit deployments inherit the bind-mounted host repo.

    from sandbox_agent.pipeline.orchestrator import (
        clear_lock_on_startup,
        reschedule_orphaned_stages_on_startup,
    )
    clear_lock_on_startup()

    from sandbox_agent.migrations import migrate_pre_skills, quarantine_unparseable_pipeline_state
    migrate_pre_skills()  # archive a stale pre-skills DATA_DIR SOUL before assembling the prompt
    quarantine_unparseable_pipeline_state()  # stop per-boot warnings from agent-corrupted state files
    system_message = load_system_message()
    logger.info("System message loaded")

    task_queue = TaskQueue()
    orphaned = task_queue.reset_orphaned_running_tasks(reason="process startup")
    if orphaned:
        logger.warning(f"Recovered {len(orphaned)} orphaned task(s): {orphaned}")
    task_queue.annotate_missed_windows()  # stamp cron tasks that missed fires while down

    from sandbox_agent.scheduler.scheduler_tools import set_task_queue
    set_task_queue(task_queue)

    rescheduled = reschedule_orphaned_stages_on_startup()
    if rescheduled:
        logger.warning(f"Rescheduled {len(rescheduled)} orphaned pipeline stage(s): {rescheduled}")

    from sandbox_agent.tools.llm_bridge import start_bridge, get_kernel_init_code
    import sandbox_agent.tools.code_interpreter as ci
    bridge_port = start_bridge()
    ci._llm_init_code = get_kernel_init_code(bridge_port)
    logger.info(
        f"LLM bridge started on port {bridge_port} — chain: "
        f"{PRIMARY_LLM_CFG['model']} → {BACKGROUND_LLM_CFG['model']} "
        f"@ {PRIMARY_LLM_CFG['model_server']}"
    )

    if status_server_port > 0:
        try:
            from sandbox_agent.status_server import start_status_server
            start_status_server(status_server_port)
            logger.info(f"Status server started on port {status_server_port}")
        except Exception:
            logger.exception("Status server failed to start; continuing without it")

    _start_background_lanes(system_message, task_queue)

    _BOOTSTRAPPED = True
    _BOOTSTRAP_TASK_QUEUE = task_queue
    return task_queue


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

    # Clear stale pipeline lock, reset stuck pipeline stages, clear stale
    # lock_holder fields. Dedicated task-queue reset happens a few lines below
    # once the shared TaskQueue is constructed.
    from sandbox_agent.pipeline.orchestrator import clear_lock_on_startup
    clear_lock_on_startup()

    # Load system messages
    # Base system message (static, used for background sessions — no metadata for KV cache stability)
    from sandbox_agent.migrations import migrate_pre_skills, quarantine_unparseable_pipeline_state
    migrate_pre_skills()  # archive a stale pre-skills DATA_DIR SOUL before assembling the prompt
    quarantine_unparseable_pipeline_state()  # stop per-boot warnings from agent-corrupted state files
    system_message = load_system_message()
    # Main agent gets one-time metadata (date, time, location) appended
    main_system_message = system_message + session_metadata()
    logger.info("System message loaded")

    # Shared task queue
    task_queue = TaskQueue()

    # Recover tasks that were 'running' when the previous process died. Without
    # this, they'd stay 'running' forever and get_due_tasks would skip them.
    orphaned = task_queue.reset_orphaned_running_tasks(reason="process startup")
    if orphaned:
        logger.warning(f"Recovered {len(orphaned)} orphaned task(s): {orphaned}")
    task_queue.annotate_missed_windows()  # stamp cron tasks that missed fires while down

    # Inject task queue into scheduler tools
    from sandbox_agent.scheduler.scheduler_tools import set_task_queue
    set_task_queue(task_queue)

    # Re-enqueue pipeline stages whose task row is missing from the queue (e.g.
    # advance_pipeline ran but its follow-up _schedule_stage call was lost
    # across a crash/rebuild). Must run AFTER set_task_queue.
    from sandbox_agent.pipeline.orchestrator import reschedule_orphaned_stages_on_startup
    rescheduled_stages = reschedule_orphaned_stages_on_startup()
    if rescheduled_stages:
        logger.warning(f"Rescheduled {len(rescheduled_stages)} orphaned pipeline stage(s): {rescheduled_stages}")

    # Start LLM bridge so code_interpreter can make LLM calls.
    # The bridge always falls through primary (qwen3.6-27b-linux) → backup
    # (qwen3.5 397B); no per-call model pinning.
    from sandbox_agent.tools.llm_bridge import start_bridge, get_kernel_init_code
    import sandbox_agent.tools.code_interpreter as ci
    bridge_port = start_bridge()
    ci._llm_init_code = get_kernel_init_code(bridge_port)
    logger.info(
        f"LLM bridge started on port {bridge_port} — chain: "
        f"{PRIMARY_LLM_CFG['model']} → {BACKGROUND_LLM_CFG['model']} "
        f"@ {PRIMARY_LLM_CFG['model_server']}"
    )

    # Main agent on primary vLLM model.
    # Note: tried using *_CHAT_LLM_CFG (thinking off) to avoid empty-content
    # streaming responses, but qwen-agent's FnCallAgent loop never terminates
    # when thinking is disabled — the model produces short responses that the
    # tool-call parser treats as incomplete, looping at ~5s/iter until the
    # 50-iter cap. Reverted to thinking-on; the visible-retry patch handles
    # the occasional empty-content case.
    # Main agent uses system message WITH metadata (date/time/location);
    # background sessions use the base system_message WITHOUT metadata (static for KV cache)
    inner_agent = create_agent(main_system_message, llm_cfg=PRIMARY_LLM_CFG)
    logger.info(f"Main agent created (model: {PRIMARY_LLM_CFG['model']})")

    # Start background lanes
    _start_background_lanes(system_message, task_queue)

    # Backup agent (397B) for when every primary-model slot is occupied
    backup_agent = create_agent(main_system_message, llm_cfg=BACKGROUND_LLM_CFG)
    logger.info(
        f"Backup agent created (model: {BACKGROUND_LLM_CFG['model']}, "
        f"primary concurrency={PRIMARY_MODEL_CONCURRENCY})"
    )

    if args.mode == "gradio":
        _run_gradio(inner_agent, backup_agent, args.host, args.port)
    else:
        _run_repl(inner_agent, backup_agent)


def _run_gradio(inner_agent: Assistant, backup_agent: Assistant, host: str, port: int) -> None:
    """Launch the Gradio WebUI."""
    from qwen_agent.gui import WebUI

    # Wrap with smart routing: vLLM when free, Ollama when vLLM is busy
    agent = LockingAgent(inner_agent, backup_agent)

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

            # Compact context if approaching token budget (OpenClaw-style
            # three-tier). Default (secondary) budget: compaction runs before
            # tier selection here, and a REPL history kept ≤170k stays
            # spillable to either tier.
            from sandbox_agent.compaction import maybe_compact
            messages = maybe_compact(messages)

            printer = StreamPrinter()
            grant = (_acquire_turn_slot(blocking=False)
                     or _acquire_turn_slot(blocking=True))
            tier, release_slot = grant
            agent = inner_agent if tier == "primary" else backup_agent
            if tier == "secondary":
                print(f"  (primary slots full — using {BACKGROUND_LLM_CFG['model']})")
            try:
                response: List[Message] = []
                for response in agent.run(messages=messages):
                    printer.update(response)
            finally:
                release_slot()
            printer.finish()

            log_turn(messages[-1], response)
            messages.extend(response)
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
