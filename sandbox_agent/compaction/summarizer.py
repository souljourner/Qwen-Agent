"""LLM summarizer — calls the configured compaction model for context compaction.

Defaults to qwen3.5 on vLLM (same server as the primary chat model); override via
COMPACTION_MODEL / COMPACTION_URL environment variables. Uses the OpenAI-compatible
/v1/chat/completions API. All failures are non-fatal — returns empty string on error.
"""

import logging

import requests

from sandbox_agent.compaction import breaker

from sandbox_agent.config import (COMPACTION_ATTEMPTS, COMPACTION_CHUNK_MAX_TOKENS,
                                   COMPACTION_MODEL, COMPACTION_TIMEOUT, COMPACTION_URL)

logger = logging.getLogger(__name__)

CHUNK_SYSTEM_PROMPT = """\
You are a conversation compactor. Summarize this AI agent conversation segment \
preserving all facts needed to continue working.

Include these sections:

## Decisions Made
What was decided and why. Include specific choices.

## Open TODOs
Unfinished tasks. Be specific about what remains.

## Constraints & Requirements
Rules, limits, or requirements established by the user or discovered during work.

## Pending User Asks
Most recent unaddressed user request.

## Tool Results Summary
Key data and findings from tool calls. Include specific numbers and outcomes, \
not just "a search was performed."

## Exact Identifiers
ALL file paths, URLs, UUIDs, IPs, project names, task IDs that appeared. \
Use exact values — never paraphrase an identifier.

## Errors Encountered
Tool failures with specific error messages.

Rules:
- Be factual and specific.
- Preserve exact numbers, dates, filenames, and identifiers verbatim.
- Summarize KEY data points, not that data was returned.
- Do not include conversational pleasantries or meta-commentary."""

MERGE_SYSTEM_PROMPT = """\
Merge these conversation summaries into one cohesive summary. \
Remove redundancy while preserving ALL:
- Decisions and their rationale
- Open TODOs (deduplicate but keep all unique items)
- Constraints and requirements
- The most recent user ask
- All unique identifiers (paths, URLs, IDs, IPs)
- All errors encountered

{identifiers_section}

Output the merged summary using the same section format as the inputs."""


def summarize_chunk(conversation_text: str) -> str:
    """Summarize a chunk of conversation history."""
    return _call_ollama(CHUNK_SYSTEM_PROMPT, conversation_text)


def merge_summaries(summaries: list, identifiers: set) -> str:
    """Merge multiple chunk summaries into one."""
    if len(summaries) == 1:
        return summaries[0]

    ids_section = ""
    if identifiers:
        ids_list = "\n".join(f"- {ident}" for ident in sorted(identifiers))
        ids_section = f"These identifiers MUST appear in the merged summary:\n{ids_list}"

    system = MERGE_SYSTEM_PROMPT.format(identifiers_section=ids_section)
    user = "\n\n---\n\n".join(
        f"### Summary {i+1}\n\n{s}" for i, s in enumerate(summaries)
    )
    return _call_ollama(system, user)


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call the compaction model. Returns "" on failure (never raises).

    Retries once by default: compaction is all-or-nothing, so a single
    transient failure (notably a cold model load on the lazily-spawning
    proxy) would otherwise abandon the whole compaction and repeat it next
    turn — minutes of work thrown away each time.

    Guarded by a circuit breaker so that a summarizer which is genuinely
    down makes compaction fail FAST rather than stalling every turn for the
    full timeout budget. See compaction/breaker.py.
    """
    if breaker.is_open():
        logger.warning("Compaction summarizer breaker OPEN — skipping call "
                       "(history left intact). Status: %s", breaker.status())
        return ""
    for attempt in range(max(1, COMPACTION_ATTEMPTS)):
        result, timed_out = _attempt_call(system_prompt, user_prompt, attempt)
        if result:
            breaker.record_success()
            return result
        if timed_out:
            # A timeout already consumed the whole budget; retrying it just
            # doubles the stall the user is sitting through. Only fast
            # failures (connection refused, 5xx, cold-start) are worth a
            # second try.
            break
    breaker.record_failure()
    return ""


def _attempt_call(system_prompt: str, user_prompt: str, attempt: int):
    """Returns (text, timed_out). Empty text means this attempt failed."""
    try:
        from sandbox_agent.model_tracker import model_start, model_done
        model_start(COMPACTION_MODEL, "context compaction")

        resp = requests.post(
            f"{COMPACTION_URL}/v1/chat/completions",
            json={
                "model": COMPACTION_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Bound the digest so a rambling summary can't blow the budget
                # we just spent effort computing. NOTE: temperature is
                # deliberately NOT sent — host per-model defaults apply
                # (standing instruction; see commit e7f458c). Do not re-add.
                "max_tokens": COMPACTION_CHUNK_MAX_TOKENS,
            },
            timeout=COMPACTION_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"].get("content", "")
        model_done(COMPACTION_MODEL)
        return result, False
    except Exception as e:
        logger.warning("Compaction summarizer attempt %d/%d failed: %s",
                       attempt + 1, COMPACTION_ATTEMPTS, e)
        try:
            from sandbox_agent.model_tracker import model_done
            model_done(COMPACTION_MODEL)
        except Exception:
            pass
        return "", isinstance(e, requests.exceptions.Timeout)
