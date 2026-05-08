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
    COMPACTION_TOOL_RESULT_MIN_KEEP,
    MAX_CONTEXT_TOKENS,
    COMPACTION_RESERVE_TOKENS,
)
from sandbox_agent.token_budget import estimate_message_tokens, estimate_messages_tokens

logger = logging.getLogger(__name__)


@dataclass
class MessageSegments:
    """Messages split into logical segments for compaction."""
    system: List[Message] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)
    recent: List[Message] = field(default_factory=list)


# --- Tier 1: Tool result truncation ---


def truncate_tool_results(messages: List[Message]) -> List[Message]:
    """Truncate oversized tool (function) results.

    Uses head/tail split to preserve important beginnings and error endings.
    Never truncates below COMPACTION_TOOL_RESULT_MIN_KEEP chars.
    """
    max_chars_per_result = int(MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN * COMPACTION_TOOL_RESULT_MAX_SHARE)
    hard_cap = COMPACTION_TOOL_RESULT_HARD_CAP
    limit = min(max_chars_per_result, hard_cap)
    min_keep = COMPACTION_TOOL_RESULT_MIN_KEEP

    result = []
    for msg in messages:
        if msg.role != "function":
            result.append(msg)
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if len(content) <= limit:
            result.append(msg)
            continue

        truncated = _head_tail_truncate(content, limit, min_keep)
        new_msg = Message(role=msg.role, content=truncated, name=msg.name, extra=msg.extra)
        result.append(new_msg)

    return result


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


def summarize_history(messages: List[Message]) -> List[Message]:
    """Summarize older conversation history, keeping recent turns verbatim.

    Returns a new message list with: system msgs + summary msg + recent turns.
    Falls back to original messages if summarization fails.
    """
    from sandbox_agent.compaction.summarizer import summarize_chunk, merge_summaries

    segments = segment_messages(messages)

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
        if summary:
            summaries.append(summary)

    if not summaries:
        logger.warning("Compaction: all chunk summarizations failed, falling back to original messages")
        return messages

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
    return reassemble(segments.system, final_summary, segments.recent, len(segments.history))


def segment_messages(messages: List[Message]) -> MessageSegments:
    """Split messages into system, history (older), and recent (preserved) segments.

    Respects tool-use/result pairing: an assistant message with function_call is
    always grouped with its following function-result message.
    """
    segments = MessageSegments()

    # Separate system messages
    non_system = []
    for msg in messages:
        if msg.role == "system":
            segments.system.append(msg)
        else:
            non_system.append(msg)

    if not non_system:
        return segments

    # Find turn boundaries (each user message starts a new turn)
    turns = _split_into_turns(non_system)

    # Preserve last N turns
    preserve_count = min(COMPACTION_RECENT_TURNS_PRESERVE, len(turns))
    if preserve_count > 0 and len(turns) > preserve_count:
        history_turns = turns[:-preserve_count]
        recent_turns = turns[-preserve_count:]
    elif preserve_count > 0:
        # All turns are "recent" — nothing to summarize
        segments.recent = non_system
        return segments
    else:
        history_turns = turns
        recent_turns = []

    for turn in history_turns:
        segments.history.extend(turn)
    for turn in recent_turns:
        segments.recent.extend(turn)

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

    budget_tokens = MAX_CONTEXT_TOKENS - COMPACTION_RESERVE_TOKENS
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


def reassemble(
    system_msgs: List[Message],
    summary: str,
    recent: List[Message],
    summarized_count: int,
) -> List[Message]:
    """Build new message list: system + summary + recent turns."""
    result = list(system_msgs)

    summary_msg = Message(
        role="system",
        content=(
            f"[Context compacted: {summarized_count} earlier messages were summarized "
            f"to save context space. Recent messages follow verbatim.]\n\n{summary}"
        ),
    )
    result.append(summary_msg)
    result.extend(recent)
    return result


# --- Helpers ---


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


def _messages_to_text(messages: List[Message]) -> str:
    """Render messages to plain text for summarization."""
    parts = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.role == "function":
            name = msg.name or "tool"
            parts.append(f"[Tool Result: {name}]\n{content[:8000]}")
        elif msg.role == "assistant":
            if msg.function_call:
                fc = msg.function_call
                parts.append(f"[Assistant calls tool: {fc.name}({fc.arguments[:500]})]\n{content[:4000]}")
            else:
                parts.append(f"[Assistant]\n{content[:4000]}")
        elif msg.role == "user":
            parts.append(f"[User]\n{content[:4000]}")
    return "\n\n".join(parts)


def _find_latest_user_ask(messages: List[Message]) -> str:
    """Find the most recent user message content."""
    for msg in reversed(messages):
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return content[:500]
    return ""
