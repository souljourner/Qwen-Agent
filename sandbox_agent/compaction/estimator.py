"""Token estimation and compaction tier selection.

Estimates whether the current message list fits within the token budget
and selects the cheapest compaction strategy if it doesn't.
"""

import logging
from typing import List, Literal, Tuple

from qwen_agent.llm.schema import Message

from sandbox_agent.config import (
    CHARS_PER_TOKEN,
    COMPACTION_SAFETY_MARGIN,
    COMPACTION_TOOL_RESULT_HARD_CAP,
    COMPACTION_TOOL_RESULT_MIN_KEEP,
    ESTIMATOR_OVERHEAD_TOKENS,
    MAX_CONTEXT_TOKENS,
    COMPACTION_RESERVE_TOKENS,
)
from sandbox_agent.token_budget import estimate_messages_tokens, estimate_message_tokens

logger = logging.getLogger(__name__)

Tier = Literal["fits", "truncate_tools", "compact", "compact_and_truncate"]


def select_tier(messages: List[Message], context_tokens: int = None) -> Tuple[Tier, int]:
    """Decide which compaction tier to use.

    Returns (tier, overflow_tokens) where overflow_tokens is how many tokens
    over budget we are (0 if fits).
    """
    # ESTIMATOR_OVERHEAD_TOKENS covers what the char-count can't see: the
    # ~40 tool JSON schemas sent with every request plus system formatting.
    # Without it a request can measure "fits" here yet overflow on the wire.
    estimated = int((estimate_messages_tokens(messages) + ESTIMATOR_OVERHEAD_TOKENS)
                    * COMPACTION_SAFETY_MARGIN)
    budget = (context_tokens or MAX_CONTEXT_TOKENS) - COMPACTION_RESERVE_TOKENS
    overflow = max(0, estimated - budget)

    if overflow == 0:
        return "fits", 0

    # Check if truncating tool results alone would be enough
    reducible_chars = _estimate_tool_result_reduction(messages)
    overflow_chars = int(overflow * CHARS_PER_TOKEN)
    # Need 50% headroom beyond the overflow to be confident truncation alone works
    truncate_threshold = int(overflow_chars * 1.5)

    if reducible_chars <= 0:
        return "compact", overflow

    if reducible_chars >= truncate_threshold:
        return "truncate_tools", overflow

    return "compact_and_truncate", overflow


def _estimate_tool_result_reduction(messages: List[Message]) -> int:
    """Estimate how many chars we could save by truncating tool results."""
    reducible = 0
    for msg in messages:
        if msg.role != "function":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        char_len = len(content)
        if char_len > COMPACTION_TOOL_RESULT_HARD_CAP:
            reducible += char_len - COMPACTION_TOOL_RESULT_MIN_KEEP
        elif char_len > COMPACTION_TOOL_RESULT_MIN_KEEP * 2:
            # Could reduce by up to half
            reducible += char_len // 2
    return reducible
