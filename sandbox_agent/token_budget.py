"""Token budget management — keeps conversations under the context limit.

Cheap heuristic (chars / CHARS_PER_TOKEN) — deliberately NOT real
tokenization: the bundled tiktoken tokenizer takes 2.2s on 100k chars of
repetitive content and raises "Max stack size exceeded for backtracking"
outright above ~1M chars, neither of which is acceptable in the hot path
(this runs several times per turn over the whole history).

CHARS_PER_TOKEN is calibrated against reality instead: a real 419-message
thread measured 718,405 chars -> 207,492 true tokens = 3.46 chars/token.

History: this undercounted a live 281,152-token chat as 183,684 (1.53x),
letting it blow past every budget until the model confabulated. Two causes,
both fixed: the 4.0 divisor was too generous, and function_call arguments
(whole file contents — 51,179 tokens across 131 calls in that thread) were
never counted at all.
"""

import logging
from typing import List

from qwen_agent.llm.schema import Message

from sandbox_agent.config import CHARS_PER_TOKEN, MAX_CONTEXT_TOKENS

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimated tokens for a string (calibrated chars-per-token heuristic)."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Message) -> int:
    """Tokens for one message: content + function_call + reasoning + framing.

    function_call arguments MUST be counted — tools like project_write_file
    pass entire file contents there, and treating them as ~10 tokens of
    overhead is what silently blew the context budget (2026-08-05)."""
    overhead = 10  # role, name, chat-template framing
    total = 0

    content = msg.content
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
            if text:
                total += estimate_tokens(str(text))

    fc = getattr(msg, "function_call", None)
    if fc is not None:
        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
        args = getattr(fc, "arguments", None) or (fc.get("arguments") if isinstance(fc, dict) else None)
        if name:
            total += estimate_tokens(str(name))
        if args:
            total += estimate_tokens(str(args))

    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        total += estimate_tokens(str(reasoning))

    return total + overhead


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

    # Separate the pinned head (system messages + the compaction digest) from
    # the rest. The digest MUST be pinned: it is role="user" at position 0, so
    # an oldest-first drop would delete the entire summarized past FIRST and
    # keep recent chatter — the agent then "silently forgets and hallucinates
    # the earlier messages". This is the same failure budget.py documents for
    # qwen_agent's own truncation, and it was live in our own trim too.
    from sandbox_agent.compaction.digest import is_digest
    system_msgs = []
    digest_msgs = []
    rest = []
    for msg in messages:
        if msg.role == "system":
            system_msgs.append(msg)
        elif is_digest(msg):
            digest_msgs.append(msg)
        else:
            rest.append(msg)

    pinned = system_msgs + digest_msgs
    system_tokens = estimate_messages_tokens(pinned)
    budget_for_rest = max_tokens - system_tokens

    # Drop oldest messages from rest until we fit
    dropped_count = 0
    while rest and estimate_messages_tokens(rest) > budget_for_rest:
        dropped = rest.pop(0)
        dropped_count += 1
        logger.debug(f"Dropped {dropped.role} message (~{estimate_message_tokens(dropped)} tokens)")

    # Insert a note so the model knows context was trimmed
    if dropped_count > 0:
        # role="user", NOT "system": a second system message makes
        # qwen_agent's _truncate_input_messages_roughly reject the entire
        # request (the 2cb762e bug — reachable through this fallback, and
        # under persisted compaction it would be written to disk).
        note = Message(
            role="user",
            content=f"[Note: {dropped_count} earlier messages were trimmed to stay within the {max_tokens}-token context budget. "
                    f"This is an automated notice, not a message from the user. "
                    f"Older conversation history is no longer available.]",
        )
        rest.insert(0, note)

    trimmed = pinned + rest
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
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + (
        "\n... (OUTPUT TRUNCATED — too much printed to stdout. "
        "Store data in variables or files instead of printing. "
        "Only print the final summary.)"
    )
