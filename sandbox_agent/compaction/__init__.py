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


def maybe_compact(messages: List[Message]) -> List[Message]:
    """Compact messages if they exceed the token budget.

    Returns the original list unchanged if compaction is disabled or unnecessary.
    All failures are non-fatal — falls back to simple trim_to_budget on error.
    """
    if not COMPACTION_ENABLED or len(messages) < 4:
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

        msg_count = len(messages)
        est_tokens = estimate_messages_tokens(messages)
        logger.info(f"Compaction triggered: tier={tier}, overflow={overflow} tokens, "
                     f"messages={msg_count}, est_tokens={est_tokens}")

        # Save checkpoint before any modification
        save_checkpoint(messages, "pre-compaction")

        set_agent_status(status="compacting", current_tool="context_compaction")

        if tier == "truncate_tools":
            result = truncate_tool_results(messages)
            # Verify it worked
            new_tier, _ = select_tier(result)
            if new_tier == "fits":
                logger.info(f"Tier 1 (tool truncation) sufficient: {msg_count} -> {len(result)} messages")
                return result
            # Not enough — escalate to tier 2
            logger.info("Tier 1 insufficient, escalating to summarization")
            result = summarize_history(result)

        elif tier == "compact":
            result = summarize_history(messages)

        elif tier == "compact_and_truncate":
            result = summarize_history(messages)
            result = truncate_tool_results(result)

        else:
            result = messages

        # Final safety net — if still over budget, use simple trimming
        final_tier, _ = select_tier(result)
        if final_tier != "fits":
            logger.warning(f"Compaction incomplete (tier={final_tier}), applying trim_to_budget fallback")
            result = trim_to_budget(result)

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
