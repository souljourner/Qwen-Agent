"""The retention ladder — what gets compacted, in priority order.

Runs cheapest/least-lossy first and STOPS the moment we are under target,
so on most turns the expensive, lossy tier never runs at all:

    L0  pointers      free, lossless-in-practice (recoverable bulk on disk)
    L1  tool results  aggregate budget, newest-first
    L2  narration     older assistant deliberation condensed
    L3  summarize     LAST resort — folds oldest turns into the digest

Never touched at any level: the system prompt, and the budget-filled
verbatim tail (which is where your most recent exchanges live).

The whole thing is PURE: it never mutates its input and never touches the
filesystem. The caller commits — so a failure part-way through can never
leave a half-destroyed history.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from qwen_agent.llm.schema import Message

from sandbox_agent.config import (
    CHARS_PER_TOKEN,
    COMPACTION_POINTERS_ENABLED,
    COMPACTION_TOOL_RESULT_TOTAL_SHARE,
)
from sandbox_agent.token_budget import estimate_messages_tokens

logger = logging.getLogger(__name__)


@dataclass
class CompactionResult:
    messages: List[Message]
    changed: bool = False
    archived: List[Message] = field(default_factory=list)
    levels: List[str] = field(default_factory=list)
    before_tokens: int = 0
    after_tokens: int = 0
    error: Optional[str] = None


def _tail_indices(messages: List[Message], target_tokens: int) -> set:
    """Indices of the budget-filled verbatim tail — never compacted."""
    from sandbox_agent.compaction.compactor import segment_messages
    seg = segment_messages(messages, target_tokens=target_tokens)
    if not seg.recent:
        return set()
    tail_ids = {id(m) for m in seg.recent}
    return {i for i, m in enumerate(messages) if id(m) in tail_ids}


def compact_for_persistence(history: List[Message], *, target_tokens: int,
                            allow_llm: bool = True) -> CompactionResult:
    """Run the ladder until `history` fits `target_tokens`.

    `allow_llm=False` runs only the deterministic tiers (L0-L2) — used by
    the cheap start-of-turn guard where a multi-minute pause is unacceptable.
    """
    before = estimate_messages_tokens(history)
    result = CompactionResult(messages=list(history), before_tokens=before,
                              after_tokens=before)
    if before <= target_tokens:
        return result

    work = list(history)
    protect = _tail_indices(work, target_tokens)

    # --- L0: pointers (free, lossless in practice) ---
    if COMPACTION_POINTERS_ENABLED:
        try:
            from sandbox_agent.compaction.pointers import pointerize
            candidate, saved = pointerize(work, protect_indices=protect)
            if saved:
                work = candidate
                result.levels.append("pointers")
        except Exception:  # noqa: BLE001 — a level may fail without aborting
            logger.exception("L0 pointers failed; continuing")
    if estimate_messages_tokens(work) <= target_tokens:
        return _finish(result, work, history)

    # --- L1: aggregate tool-result budget ---
    try:
        from sandbox_agent.compaction.compactor import truncate_tool_results
        budget_chars = int(target_tokens * COMPACTION_TOOL_RESULT_TOTAL_SHARE * CHARS_PER_TOKEN)
        candidate = truncate_tool_results(work, total_budget_chars=budget_chars)
        if estimate_messages_tokens(candidate) < estimate_messages_tokens(work):
            work = candidate
            result.levels.append("tool_results")
    except Exception:  # noqa: BLE001
        logger.exception("L1 tool-result truncation failed; continuing")
    if estimate_messages_tokens(work) <= target_tokens:
        return _finish(result, work, history)

    # --- L2: condense older assistant narration ---
    try:
        candidate = _condense_narration(work, protect)
        if estimate_messages_tokens(candidate) < estimate_messages_tokens(work):
            work = candidate
            result.levels.append("narration")
    except Exception:  # noqa: BLE001
        logger.exception("L2 narration condensing failed; continuing")
    if estimate_messages_tokens(work) <= target_tokens:
        return _finish(result, work, history)

    # --- L3: summarize into the digest (lossy, LLM, all-or-nothing) ---
    if not allow_llm:
        logger.info("Over target but LLM tiers disabled — leaving to end-of-turn compaction")
        return _finish(result, work, history)
    try:
        from sandbox_agent.compaction.compactor import summarize_history
        before_l3 = work
        candidate = summarize_history(work, target_tokens=target_tokens)
        if candidate is not before_l3:
            kept = {id(m) for m in candidate}
            result.archived = [m for m in before_l3 if id(m) not in kept]
            work = candidate
            result.levels.append("summarize")
        else:
            result.error = "summarization aborted (history left intact)"
    except Exception as e:  # noqa: BLE001
        logger.exception("L3 summarization failed")
        result.error = f"summarize failed: {type(e).__name__}"
    return _finish(result, work, history)


def _finish(result: CompactionResult, work: List[Message],
            original: List[Message]) -> CompactionResult:
    result.messages = work
    result.after_tokens = estimate_messages_tokens(work)
    result.changed = len(work) != len(original) or any(
        a is not b for a, b in zip(work, original))
    if result.changed:
        logger.info("Compaction ladder %s: %d -> %d tokens",
                    "+".join(result.levels) or "none",
                    result.before_tokens, result.after_tokens)
    return result


def _condense_narration(messages: List[Message], protect: set) -> List[Message]:
    """Trim long assistant prose outside the tail, keeping head and tail of
    each message (conclusions usually live at the end)."""
    from sandbox_agent.compaction.compactor import _head_tail_truncate
    limit = 2_000
    out = []
    for i, msg in enumerate(messages):
        content = msg.content
        if (i in protect or msg.role != "assistant" or msg.function_call
                or not isinstance(content, str) or len(content) <= limit):
            out.append(msg)
            continue
        out.append(Message(role=msg.role,
                           content=_head_tail_truncate(content, limit, 400),
                           name=msg.name, extra=msg.extra))
    return out
