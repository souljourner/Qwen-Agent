"""OpenClaw-style context compaction for the sandbox agent.

Single entry point: maybe_compact(messages) -> List[Message]

Three-tier strategy:
  1. Truncate oversized tool results (cheapest)
  2. LLM-based history summarization with adaptive chunking
  3. Both — summarize then truncate, with trim_to_budget as final fallback
"""

import logging
from typing import List

from qwen_agent.llm.schema import Message

from sandbox_agent.config import COMPACTION_ENABLED

logger = logging.getLogger(__name__)


def maybe_compact(messages: List[Message], pinned_head: int = 0) -> List[Message]:
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

    from sandbox_agent.compaction.estimator import select_tier
    from sandbox_agent.compaction.compactor import truncate_tool_results, summarize_history
    from sandbox_agent.compaction.checkpoint import save_checkpoint
    from sandbox_agent.model_tracker import set_agent_status, clear_agent_status
    from sandbox_agent.token_budget import trim_to_budget, estimate_messages_tokens

    try:
        tier, overflow = select_tier(messages)

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
            new_tier, _ = select_tier(head + tail_result)
            if new_tier == "fits":
                logger.info(f"Tier 1 (tool truncation) sufficient: {msg_count} -> "
                            f"{len(head) + len(tail_result)} messages")
                return head + tail_result
            # Not enough — escalate to tier 2
            logger.info("Tier 1 insufficient, escalating to summarization")
            tail_result = summarize_history(tail_result)

        elif tier == "compact":
            tail_result = summarize_history(tail)

        elif tier == "compact_and_truncate":
            tail_result = summarize_history(tail)
            tail_result = truncate_tool_results(tail_result)

        else:
            tail_result = tail

        result = head + tail_result

        # Final safety net — if still over budget, trim the tail (never the head)
        final_tier, _ = select_tier(result)
        if final_tier != "fits":
            logger.warning(f"Compaction incomplete (tier={final_tier}), applying trim fallback")
            from sandbox_agent.config import COMPACTION_RESERVE_TOKENS, MAX_CONTEXT_TOKENS
            head_tokens = estimate_messages_tokens(head) if head else 0
            tail_budget = max(10_000, MAX_CONTEXT_TOKENS - COMPACTION_RESERVE_TOKENS - head_tokens)
            result = head + trim_to_budget(tail_result, max_tokens=tail_budget)

        new_tokens = estimate_messages_tokens(result)
        logger.info(f"Compaction complete: {msg_count} msgs ({est_tokens} tok) -> "
                     f"{len(result)} msgs ({new_tokens} tok)")
        return result

    except Exception:
        logger.exception("Compaction failed, falling back to trim_to_budget")
        from sandbox_agent.token_budget import trim_to_budget
        return trim_to_budget(messages)
    finally:
        try:
            clear_agent_status()
        except Exception:
            pass


def compact_midrun(messages: List[Message]) -> List[Message]:
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
    return maybe_compact(messages, pinned_head=head_end)
