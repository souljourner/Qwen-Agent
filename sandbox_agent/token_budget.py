"""Token budget management — keeps conversations under the context limit.

Uses a cheap heuristic (chars / 4) instead of real tokenization.
Drops oldest non-system turns to stay under budget while preserving
the system message prefix (for KV cache hits).
"""

import logging
from typing import List

from qwen_agent.llm.schema import Message

from sandbox_agent.config import CHARS_PER_TOKEN, MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length. ~4 chars per token for English."""
    return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(msg: Message) -> int:
    """Estimate tokens for a single message (content + overhead)."""
    overhead = 10  # role, name, formatting tokens
    if isinstance(msg.content, str):
        return estimate_tokens(msg.content) + overhead
    elif isinstance(msg.content, list):
        total = 0
        for item in msg.content:
            if hasattr(item, "text") and item.text:
                total += estimate_tokens(item.text)
        return total + overhead
    return overhead


def estimate_messages_tokens(messages: List[Message]) -> int:
    """Estimate total tokens across all messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def trim_to_budget(messages: List[Message], max_tokens: int = MAX_CONTEXT_TOKENS) -> List[Message]:
    """Trim messages to fit within token budget.

    Strategy: preserve system message (index 0) and the most recent turns.
    Drop the oldest non-system messages first. This maintains KV cache
    for the system prefix and keeps recent context intact.

    Args:
        messages: Full conversation history.
        max_tokens: Token budget (default from config).

    Returns:
        Trimmed message list. May be the same list if already under budget.
    """
    total = estimate_messages_tokens(messages)
    if total <= max_tokens:
        return messages

    logger.info(f"Token budget exceeded: ~{total} tokens > {max_tokens} limit. Trimming old turns.")

    # Separate system message from the rest
    system_msgs = []
    rest = []
    for msg in messages:
        if msg.role == "system":
            system_msgs.append(msg)
        else:
            rest.append(msg)

    system_tokens = estimate_messages_tokens(system_msgs)
    budget_for_rest = max_tokens - system_tokens

    # Drop oldest messages from rest until we fit
    dropped_count = 0
    while rest and estimate_messages_tokens(rest) > budget_for_rest:
        dropped = rest.pop(0)
        dropped_count += 1
        logger.debug(f"Dropped {dropped.role} message (~{estimate_message_tokens(dropped)} tokens)")

    # Insert a note so the model knows context was trimmed
    if dropped_count > 0:
        note = Message(
            role="system",
            content=f"[Note: {dropped_count} earlier messages were trimmed to stay within the {max_tokens}-token context budget. "
                    f"Older conversation history is no longer available.]",
        )
        rest.insert(0, note)

    trimmed = system_msgs + rest
    new_total = estimate_messages_tokens(trimmed)
    logger.info(f"Trimmed to ~{new_total} tokens ({len(messages)} -> {len(trimmed)} messages, {dropped_count} dropped)")
    return trimmed


MIN_REQUEST_TIMEOUT = 600    # 10 minutes minimum
MAX_REQUEST_TIMEOUT = 1800   # 30 minutes at full 200k context
# Large-context turns (laguna pinned): a ~900k cold prefill measures ~27min
# at laguna's ~550 tok/s — the 1800s cap would kill it mid-prefill.
MAX_REQUEST_TIMEOUT_LARGE = 3600


def compute_request_timeout(messages: List[Message]) -> int:
    """Compute a request timeout based on message payload size.

    Linear scale: 10 min at ~0 tokens, 30 min at 200k tokens.
    Larger payloads need more time for prefill + generation.
    """
    tokens = estimate_messages_tokens(messages)
    if tokens <= MAX_CONTEXT_TOKENS:
        # Linear interpolation: 0 tokens → MIN, MAX_CONTEXT_TOKENS → MAX
        fraction = tokens / MAX_CONTEXT_TOKENS
        return int(MIN_REQUEST_TIMEOUT + fraction * (MAX_REQUEST_TIMEOUT - MIN_REQUEST_TIMEOUT))
    # Beyond the standard window (laguna-pinned turns): scale on toward
    # MAX_REQUEST_TIMEOUT_LARGE at the primary tier's full budget.
    from sandbox_agent.config import PRIMARY_CONTEXT_TOKENS
    span = max(PRIMARY_CONTEXT_TOKENS - MAX_CONTEXT_TOKENS, 1)
    fraction = min((tokens - MAX_CONTEXT_TOKENS) / span, 1.0)
    return int(MAX_REQUEST_TIMEOUT + fraction * (MAX_REQUEST_TIMEOUT_LARGE - MAX_REQUEST_TIMEOUT))


def truncate_output(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + (
        "\n... (OUTPUT TRUNCATED — too much printed to stdout. "
        "Store data in variables or files instead of printing. "
        "Only print the final summary.)"
    )
