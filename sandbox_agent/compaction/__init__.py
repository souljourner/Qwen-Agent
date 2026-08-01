"""OpenClaw-style context compaction for the sandbox agent.

Single entry point: maybe_compact(messages) -> List[Message]

Three-tier strategy:
  1. Truncate oversized tool results (cheapest)
  2. LLM-based history summarization with adaptive chunking
  3. Both — summarize then truncate, with trim_to_budget as final fallback
"""

import logging
from typing import List, Optional

from qwen_agent.llm.schema import Message

from sandbox_agent.config import COMPACTION_ENABLED

logger = logging.getLogger(__name__)

# UI heads-up hooks for slow (summarization-tier) compactions, keyed by worker
# thread ident — same isolation pattern as code_interpreter's progress hooks.
# A summarize compaction cold-prefills the whole history through the
# summarizer model and can run for minutes; the chat must not hang silently.
import math
import threading
import time as _time

_notice_hooks: dict = {}
_SECONDS_PER_CHUNK = 35  # summarizer prefill+generation per ~68k-token chunk


def register_compaction_notice_hook(fn) -> None:
    _notice_hooks[threading.get_ident()] = fn


def unregister_compaction_notice_hook() -> None:
    _notice_hooks.pop(threading.get_ident(), None)


def _notify(payload: dict) -> None:
    hook = _notice_hooks.get(threading.get_ident())
    if hook is None:
        return
    try:
        hook(payload)
    except Exception:  # noqa: BLE001 — a UI hook must never break compaction
        logger.debug("compaction notice hook failed", exc_info=True)


def maybe_compact(messages: List[Message], pinned_head: int = 0,
                  context_tokens: Optional[int] = None) -> List[Message]:
    """Compact messages if they exceed the token budget.

    `pinned_head` leading messages pass through VERBATIM (but still count
    toward the budget) — mid-run callers pin the system prompt + task message
    so summarization can never eat the agent's instructions.

    Returns the original list unchanged if compaction is disabled or
    unnecessary. All failures are non-fatal — falls back to trimming on error.
    No minimum message count: a fresh background session with one giant
    message plus one giant tool result must still be compactable (a pipeline
    stage once hit the vLLM context ceiling exactly this way).
    """
    if not COMPACTION_ENABLED or not messages:
        return messages
    # Per-tier budget: laguna turns pass their 900k window; default is the
    # secondary/global 200k. All tier decisions and fallbacks use this.
    from sandbox_agent.config import MAX_CONTEXT_TOKENS as _DEFAULT_CTX
    ctx = context_tokens or _DEFAULT_CTX

    from sandbox_agent.compaction.estimator import select_tier
    from sandbox_agent.compaction.compactor import truncate_tool_results, summarize_history
    from sandbox_agent.compaction.checkpoint import save_checkpoint
    from sandbox_agent.model_tracker import set_agent_status, clear_agent_status
    from sandbox_agent.token_budget import trim_to_budget, estimate_messages_tokens

    try:
        tier, overflow = select_tier(messages, context_tokens=ctx)

        if tier == "fits":
            return messages

        pinned_head = max(0, min(pinned_head, len(messages)))
        head = list(messages[:pinned_head])
        tail = list(messages[pinned_head:])
        if not tail:
            return messages  # nothing compactable outside the pinned head

        msg_count = len(messages)
        est_tokens = estimate_messages_tokens(messages)
        logger.info(f"Compaction triggered: tier={tier}, overflow={overflow} tokens, "
                     f"messages={msg_count}, est_tokens={est_tokens}, pinned_head={pinned_head}")

        # Save checkpoint before any modification
        save_checkpoint(messages, "pre-compaction")

        set_agent_status(status="compacting", current_tool="context_compaction")

        if tier == "truncate_tools":
            tail_result = truncate_tool_results(tail)
            new_tier, _ = select_tier(head + tail_result, context_tokens=ctx)
            if new_tier == "fits":
                logger.info(f"Tier 1 (tool truncation) sufficient: {msg_count} -> "
                            f"{len(head) + len(tail_result)} messages")
                return head + tail_result
            # Not enough — escalate to tier 2
            logger.info("Tier 1 insufficient, escalating to summarization")
            tail_result = _summarize_with_notice(tail_result, est_tokens)

        elif tier == "compact":
            tail_result = _summarize_with_notice(tail, est_tokens)

        elif tier == "compact_and_truncate":
            tail_result = _summarize_with_notice(tail, est_tokens)
            tail_result = truncate_tool_results(tail_result)

        else:
            tail_result = tail

        result = head + tail_result

        # Final safety net — if still over budget, trim the tail (never the head)
        final_tier, _ = select_tier(result, context_tokens=ctx)
        if final_tier != "fits":
            logger.warning(f"Compaction incomplete (tier={final_tier}), applying trim fallback")
            from sandbox_agent.config import COMPACTION_RESERVE_TOKENS
            head_tokens = estimate_messages_tokens(head) if head else 0
            tail_budget = max(10_000, ctx - COMPACTION_RESERVE_TOKENS - head_tokens)
            result = head + trim_to_budget(tail_result, max_tokens=tail_budget)

        new_tokens = estimate_messages_tokens(result)
        logger.info(f"Compaction complete: {msg_count} msgs ({est_tokens} tok) -> "
                     f"{len(result)} msgs ({new_tokens} tok)")
        return result

    except Exception:
        logger.exception("Compaction failed, falling back to trim_to_budget")
        # Trim to THIS TIER's budget — the 200k default would amputate 600k
        # tokens off a pinned 800k laguna history.
        from sandbox_agent.config import COMPACTION_RESERVE_TOKENS
        from sandbox_agent.token_budget import trim_to_budget
        return trim_to_budget(messages, max_tokens=max(10_000, ctx - COMPACTION_RESERVE_TOKENS))
    finally:
        try:
            clear_agent_status()
        except Exception:
            pass


def _summarize_with_notice(msgs: List[Message], est_tokens: int) -> List[Message]:
    """Wrap summarize_history with start/done UI notices + duration estimate."""
    from sandbox_agent.compaction.compactor import summarize_history
    from sandbox_agent.config import COMPACTION_RESERVE_TOKENS, SUMMARIZER_CONTEXT_TOKENS, COMPACTION_BASE_CHUNK_RATIO
    chunk_tokens = max(1, int((SUMMARIZER_CONTEXT_TOKENS - COMPACTION_RESERVE_TOKENS)
                              * COMPACTION_BASE_CHUNK_RATIO))
    est_chunks = max(1, math.ceil(est_tokens / chunk_tokens))
    est_minutes = max(1, round(est_chunks * _SECONDS_PER_CHUNK / 60))
    _notify({"phase": "start", "est_tokens": est_tokens,
             "est_chunks": est_chunks, "est_minutes": est_minutes})
    started = _time.monotonic()
    try:
        return summarize_history(msgs)
    finally:
        _notify({"phase": "done", "elapsed_s": int(_time.monotonic() - started)})


def compact_midrun(messages: List[Message],
                   context_tokens: Optional[int] = None) -> List[Message]:
    """Compaction for INSIDE the fncall tool-call loop (runs before every LLM
    call). Pins the leading system message(s) + first user message — for a
    background/pipeline session that first user message IS the task; the
    bloat lives in the accumulated tool results after it."""
    head_end = 0
    for m in messages:
        if m.role == "system":
            head_end += 1
        else:
            break
    if head_end < len(messages) and messages[head_end].role == "user":
        head_end += 1
    return maybe_compact(messages, pinned_head=head_end, context_tokens=context_tokens)
