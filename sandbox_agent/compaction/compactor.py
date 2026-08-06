"""Core compaction engine — three-tier context management.

Tier 1: Truncate oversized tool results (cheapest)
Tier 2: LLM-based history summarization with adaptive chunking
Tier 3: Both — summarize then truncate remaining
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from qwen_agent.llm.schema import Message

from sandbox_agent.config import (
    CHARS_PER_TOKEN,
    COMPACTION_BASE_CHUNK_RATIO,
    COMPACTION_FAILURE_CHARS,
    COMPACTION_MAX_FAILURES,
    COMPACTION_MAX_IDENTIFIERS,
    COMPACTION_MIN_CHUNK_RATIO,
    COMPACTION_RECENT_TURNS_PRESERVE,
    COMPACTION_SAFETY_MARGIN,
    COMPACTION_TOOL_RESULT_HARD_CAP,
    COMPACTION_TOOL_RESULT_MAX_SHARE,
    COMPACTION_MIN_TAIL_TURNS,
    COMPACTION_PROTECT_LAST_TOOL_RESULTS,
    COMPACTION_TOOL_RESULT_MIN_KEEP,
    MAX_CONTEXT_TOKENS,
    COMPACTION_RESERVE_TOKENS,
)
from sandbox_agent.token_budget import estimate_message_tokens, estimate_messages_tokens

logger = logging.getLogger(__name__)

# Sentinel emitted by _head_tail_truncate ("[... truncated N chars ...]").
# Re-truncating already-truncated content must be a no-op, or the markers
# nest on every compaction pass.
_ALREADY_TRUNCATED = "... truncated "


@dataclass
class MessageSegments:
    """Messages split into logical segments for compaction.

    `digest` is a previously-persisted compaction digest peeled off the
    front so it can never be re-summarized into a digest-of-a-digest, and
    so it never opens a spurious "turn" (it is role="user").
    """
    system: List[Message] = field(default_factory=list)
    digest: Optional[Message] = None
    history: List[Message] = field(default_factory=list)   # to be summarized
    recent: List[Message] = field(default_factory=list)    # kept verbatim


# --- Tier 1: Tool result truncation ---


def truncate_tool_results(messages: List[Message], *,
                          total_budget_chars: Optional[int] = None,
                          protect_last_n: int = None) -> List[Message]:
    """Shrink tool results to fit an AGGREGATE character budget.

    Previously each result was capped individually at 40k chars, so sixty
    40k-char results (2.4M chars ~ 685k tokens) were all "within cap" and
    this tier did nothing useful. Now the budget is shared: newest results
    keep their space, older ones fall back to COMPACTION_TOOL_RESULT_MIN_KEEP.
    The newest `protect_last_n` results are never touched — the agent is
    usually still working with them."""
    if protect_last_n is None:
        protect_last_n = COMPACTION_PROTECT_LAST_TOOL_RESULTS
    if total_budget_chars is None:
        total_budget_chars = int(MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN
                                 * COMPACTION_TOOL_RESULT_MAX_SHARE)

    fn_idx = [i for i, m in enumerate(messages) if m.role == "function"]
    if not fn_idx:
        return messages

    protected = set(fn_idx[-protect_last_n:]) if protect_last_n else set()
    remaining = total_budget_chars
    # Reserve nothing for protected results — they are kept whole regardless.
    keep_len = {}
    for i in reversed(fn_idx):  # newest first
        content = messages[i].content
        n = len(content) if isinstance(content, str) else 0
        if i in protected:
            # Exempt from the shared budget (the agent is likely still using
            # it) but NOT from the per-result hard cap — otherwise a single
            # huge recent result makes this tier a no-op.
            keep_len[i] = min(n, COMPACTION_TOOL_RESULT_HARD_CAP)
            continue
        if remaining <= 0:
            keep_len[i] = COMPACTION_TOOL_RESULT_MIN_KEEP
            continue
        allow = min(n, remaining, COMPACTION_TOOL_RESULT_HARD_CAP)
        allow = max(allow, COMPACTION_TOOL_RESULT_MIN_KEEP)
        keep_len[i] = allow
        remaining -= allow

    out = []
    for i, msg in enumerate(messages):
        content = msg.content
        if msg.role != "function" or not isinstance(content, str):
            out.append(msg)
            continue
        limit = keep_len.get(i, len(content))
        if len(content) <= limit or _ALREADY_TRUNCATED in content:
            out.append(msg)
            continue
        out.append(Message(
            role=msg.role,
            content=_head_tail_truncate(content, limit, COMPACTION_TOOL_RESULT_MIN_KEEP),
            name=msg.name,
            extra=msg.extra,
        ))
    return out


def _head_tail_truncate(text: str, budget: int, min_keep: int) -> str:
    """Split text into head + tail, preserving important endings."""
    if len(text) <= budget:
        return text
    budget = max(budget, min_keep)

    # Check for important tail patterns (errors, JSON closings, summaries)
    has_important_tail = bool(re.search(
        r"(Error|Exception|Traceback|FAIL|}\s*$|]\s*$|Summary|Result)",
        text[-2000:],
        re.IGNORECASE,
    ))

    if has_important_tail:
        head_budget = min(int(budget * 0.7), budget - 500)
        tail_budget = budget - head_budget
    else:
        head_budget = budget
        tail_budget = 0

    head = text[:head_budget]
    omitted = len(text) - head_budget - tail_budget
    marker = f"\n\n[... truncated {omitted:,} chars ...]\n\n"

    if tail_budget > 0:
        tail = text[-tail_budget:]
        return head + marker + tail
    return head + marker


# --- Tier 2: History summarization ---


def summarize_history(messages: List[Message], *,
                      target_tokens: Optional[int] = None) -> List[Message]:
    """Summarize older conversation history, keeping recent turns verbatim.

    Returns a new message list with: system msgs + summary msg + recent turns.
    Falls back to original messages if summarization fails.
    """
    from sandbox_agent.compaction.summarizer import summarize_chunk, merge_summaries

    segments = segment_messages(messages, target_tokens=target_tokens)

    if not segments.history:
        logger.info("Compaction: no history to summarize (all messages are recent)")
        return messages

    # Extract identifiers and tool failures from history before summarizing
    history_text = _messages_to_text(segments.history)
    identifiers = extract_identifiers(history_text)
    failures = extract_tool_failures(segments.history)

    # Build chunks with adaptive sizing
    chunks = build_chunks(segments.history)

    if not chunks:
        return messages

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_text = _messages_to_text(chunk)
        if failures and i == len(chunks) - 1:
            # Append tool failures to the last chunk for context
            failures_text = "\n\n## Recent Tool Failures:\n" + "\n".join(f"- {f}" for f in failures)
            chunk_text += failures_text

        summary = summarize_chunk(chunk_text)
        if not summary:
            # ALL-OR-NOTHING. Previously a failed chunk was skipped and the
            # remaining summaries still became the digest — silently DELETING
            # the material those failed chunks covered. Aborting leaves the
            # history intact; the caller retries next turn.
            logger.warning("Compaction: chunk %d/%d failed to summarize — aborting "
                           "compaction (history left intact)", i + 1, len(chunks))
            return messages
        summaries.append(summary)

    # Merge if multiple chunks
    final_summary = merge_summaries(summaries, identifiers)
    if not final_summary:
        logger.warning("Compaction: summary merge failed, using first chunk summary")
        final_summary = summaries[0]

    # Quality audit (non-blocking)
    latest_user_ask = _find_latest_user_ask(messages)
    passed, issues = quality_audit(final_summary, identifiers, latest_user_ask)
    if not passed:
        logger.warning(f"Compaction quality audit issues: {issues}")

    # Reassemble
    return reassemble(segments.system, final_summary, segments.recent,
                      len(segments.history), existing_digest=segments.digest,
                      durable=_durable_from_failures(failures))


def segment_messages(messages: List[Message], *,
                     target_tokens: Optional[int] = None,
                     min_tail_turns: int = None) -> MessageSegments:
    """Split into system / digest / history-to-summarize / verbatim tail.

    The tail is BUDGET-FILLED: walk backwards keeping whole turns until the
    budget is spent, instead of the old hardcoded 2 turns. That fixed count
    is what turned a 281k history into 13.4k when the budget allowed 170k
    (2026-08-06 incident) — it discarded ~12x more than necessary.

    The newest turn is always kept even if it alone busts the budget; the
    ladder's cheaper tiers shrink it instead.
    """
    from sandbox_agent.compaction.digest import is_digest

    if min_tail_turns is None:
        min_tail_turns = COMPACTION_MIN_TAIL_TURNS
    segments = MessageSegments()

    non_system = []
    for msg in messages:
        if msg.role == "system":
            segments.system.append(msg)
        else:
            non_system.append(msg)
    if not non_system:
        return segments

    # Peel a leading digest so it never opens a turn nor gets re-summarized.
    if is_digest(non_system[0]):
        segments.digest = non_system[0]
        non_system = non_system[1:]
    if not non_system:
        return segments

    turns = _split_into_turns(non_system)

    if not target_tokens:  # legacy callers: previous fixed-count behavior
        preserve = min(COMPACTION_RECENT_TURNS_PRESERVE, len(turns))
        if preserve and len(turns) > preserve:
            segments.history = [m for t in turns[:-preserve] for m in t]
            segments.recent = [m for t in turns[-preserve:] for m in t]
        else:
            segments.recent = non_system
        return segments

    budget = target_tokens - estimate_messages_tokens(segments.system)
    if segments.digest is not None:
        budget -= estimate_message_tokens(segments.digest)
    budget = max(budget, 1_000)

    kept: List[List[Message]] = []
    used = 0
    for turn in reversed(turns):
        cost = estimate_messages_tokens(turn)
        # `kept and` is load-bearing: the newest turn is kept unconditionally
        if kept and used + cost > budget:
            break
        kept.insert(0, turn)
        used += cost
    while len(kept) < min(min_tail_turns, len(turns)):
        kept.insert(0, turns[len(turns) - len(kept) - 1])

    older = turns[:len(turns) - len(kept)]
    segments.history = [m for t in older for m in t]
    segments.recent = [m for t in kept for m in t]

    # A tail must start with a user message or qwen_agent rejects the
    # request; a rebuilt history can begin mid tool-exchange.
    while segments.recent and segments.recent[0].role != "user":
        segments.history.append(segments.recent.pop(0))
    return segments


def _split_into_turns(messages: List[Message]) -> List[List[Message]]:
    """Split messages into turns. Each user message starts a new turn.

    Tool-use/result pairs (assistant with function_call + following function msg)
    are always kept together within the same turn.
    """
    turns = []
    current_turn = []

    for msg in messages:
        if msg.role == "user" and current_turn:
            turns.append(current_turn)
            current_turn = []
        current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    return turns


def build_chunks(history: List[Message]) -> List[List[Message]]:
    """Split history into chunks for independent summarization.

    Uses adaptive chunk sizing: reduces chunk size when average message is large.
    Respects tool-use/result pairs at chunk boundaries.
    """
    if not history:
        return []

    # Chunks are consumed by the SUMMARIZER model (qwen3.6, ~262k window) —
    # size them against ITS window, never the (possibly 900k) history budget.
    from sandbox_agent.config import SUMMARIZER_CONTEXT_TOKENS
    budget_tokens = SUMMARIZER_CONTEXT_TOKENS - COMPACTION_RESERVE_TOKENS
    total_tokens = estimate_messages_tokens(history)

    # Adaptive chunk ratio
    chunk_ratio = _compute_adaptive_chunk_ratio(history, budget_tokens)
    max_chunk_tokens = int(budget_tokens * chunk_ratio)

    if total_tokens <= max_chunk_tokens:
        return [history]

    # Split into turns first to respect boundaries
    turns = _split_into_turns(history)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for turn in turns:
        turn_tokens = estimate_messages_tokens(turn)
        if current_chunk and (current_tokens + turn_tokens) > max_chunk_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.extend(turn)
        current_tokens += turn_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _compute_adaptive_chunk_ratio(messages: List[Message], budget_tokens: int) -> float:
    """Reduce chunk size when individual messages are large."""
    if not messages:
        return COMPACTION_BASE_CHUNK_RATIO

    total_tokens = estimate_messages_tokens(messages)
    avg_tokens = total_tokens / len(messages)
    avg_with_safety = avg_tokens * COMPACTION_SAFETY_MARGIN
    avg_ratio = avg_with_safety / budget_tokens

    if avg_ratio > 0.1:  # avg message > 10% of context
        reduction = min(avg_ratio * 2, COMPACTION_BASE_CHUNK_RATIO - COMPACTION_MIN_CHUNK_RATIO)
        return max(COMPACTION_MIN_CHUNK_RATIO, COMPACTION_BASE_CHUNK_RATIO - reduction)

    return COMPACTION_BASE_CHUNK_RATIO


def _durable_from_failures(failures: List[str]) -> List[str]:
    """Tool failures are a user-priority retention category — record them as
    DURABLE bullets so a later summarization can never paraphrase them away."""
    return [f"- ERROR: {f}" for f in (failures or [])[:8]]


def reassemble(system_msgs: List[Message], summary: str, recent: List[Message],
               summarized_count: int, *, existing_digest: Optional[Message] = None,
               durable: Optional[List[str]] = None) -> List[Message]:
    """Build [system] + [digest] + [verbatim tail].

    MERGES into `existing_digest` rather than stacking a second one — a
    persisted history already starts with a digest, and stacking would give
    the model two conflicting "[Context compacted…]" preambles and
    eventually digests-of-digests.
    """
    from sandbox_agent.compaction.digest import merge_into, parse_sections

    proposed_durable, working = parse_sections(summary)
    if not proposed_durable and not working:
        working = summary  # model ignored the section format — keep it all
    proposed_durable = list(durable or []) + proposed_durable

    digest_msg = merge_into(existing_digest, proposed_durable, working, summarized_count)
    return list(system_msgs) + [digest_msg] + list(recent)


def extract_identifiers(text: str) -> Set[str]:
    """Extract opaque identifiers from text (UUIDs, URLs, paths, IPs, IDs)."""
    patterns = [
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",  # UUIDs
        r"https?://[^\s\"'<>\])+]+",  # URLs
        r"(?:^|\s)(/[\w./-]+\.\w+)",  # Unix file paths
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d{1,5})?",  # IPs with optional port
    ]

    found = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            value = match.group(0).strip()
            # Clean boundaries
            value = value.strip("\"'`()[]{}<>;:,. ")
            if len(value) >= 4:
                found.add(value)

    # Limit to max identifiers
    if len(found) > COMPACTION_MAX_IDENTIFIERS:
        # Prefer longer identifiers (more specific)
        found = set(sorted(found, key=len, reverse=True)[:COMPACTION_MAX_IDENTIFIERS])

    return found


def extract_tool_failures(messages: List[Message]) -> List[str]:
    """Extract recent tool failures for preservation in summary."""
    failures = []
    error_patterns = re.compile(
        r"(Error|Exception|Traceback|Failed|HTTP [45]\d\d|status[: ]+[45]\d\d)",
        re.IGNORECASE,
    )

    for msg in messages:
        if msg.role != "function":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if error_patterns.search(content):
            tool_name = msg.name or "tool"
            summary = content.strip().replace("\n", " ")[:COMPACTION_FAILURE_CHARS]
            failures.append(f"{tool_name}: {summary}")

    # Keep last N failures
    return failures[-COMPACTION_MAX_FAILURES:]


def quality_audit(summary: str, identifiers: Set[str], latest_user_ask: str) -> Tuple[bool, List[str]]:
    """Check summary quality. Returns (passed, list_of_issues)."""
    issues = []

    required_sections = [
        "## Decisions",
        "## Open TODOs",
        "## Constraints",
        "## Pending User Asks",
        "## Exact Identifiers",
    ]
    summary_lower = summary.lower()
    for section in required_sections:
        if section.lower().lstrip("# ") not in summary_lower:
            issues.append(f"missing_section:{section}")

    # Check identifier preservation (allow 80% match)
    if identifiers:
        present = sum(1 for ident in identifiers if ident in summary)
        if present < len(identifiers) * 0.8:
            missing = [i for i in identifiers if i not in summary][:3]
            issues.append(f"missing_identifiers:{','.join(missing)}")

    # Check latest user ask is reflected
    if latest_user_ask:
        ask_words = set(latest_user_ask.lower().split())
        ask_words -= {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "it", "i"}
        # Strip punctuation from summary words for fuzzy matching
        summary_words = set(re.sub(r"[^\w\s]", " ", summary_lower).split())
        overlap = ask_words & summary_words
        min_required = 1 if len(ask_words) <= 5 else 2
        if len(overlap) < min_required:
            issues.append("latest_user_ask_not_reflected")

    return len(issues) == 0, issues


def _messages_to_text(messages: List[Message], *, char_budget: Optional[int] = None) -> str:
    """Render messages to plain text for summarization.

    FULL FIDELITY by design. This used to truncate every message inline
    (tool results 8k chars, user/assistant 4k, function_call args 500) while
    build_chunks MEASURED the same messages at full size — a ~50x gap that
    handed the summarizer ~2% of the conversation and produced the
    2026-08-06 "compacted 281k into 13.4k and knows nothing" incident.
    Chunks are already sized to the summarizer's window by build_chunks, so
    rendering everything is correct by construction; `char_budget` is only a
    belt-and-braces guard against a mis-sized chunk."""
    parts = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if msg.role == "function":
            name = msg.name or "tool"
            parts.append(f"[Tool Result: {name}]\n{content}")
        elif msg.role == "assistant":
            if msg.function_call:
                fc = msg.function_call
                parts.append(f"[Assistant calls tool: {fc.name}({fc.arguments})]\n{content}")
            else:
                parts.append(f"[Assistant]\n{content}")
        elif msg.role == "user":
            parts.append(f"[User]\n{content}")
    text = "\n\n".join(parts)
    if char_budget and len(text) > char_budget:
        logger.warning("Chunk render %d chars exceeds budget %d — head/tail guard applied",
                       len(text), char_budget)
        text = _head_tail_truncate(text, char_budget, COMPACTION_TOOL_RESULT_MIN_KEEP)
    return text


def _find_latest_user_ask(messages: List[Message]) -> str:
    """Find the most recent user message content."""
    for msg in reversed(messages):
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return content[:500]
    return ""
