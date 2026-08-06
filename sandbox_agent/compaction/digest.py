"""The compaction digest: a single message holding everything summarized.

Two physical sections, with very different trust models:

    ### DURABLE
    Code-managed. User corrections, decisions + rationale, open commitments,
    errors worth not repeating. Appended by CODE and NEVER re-sent through
    the model, so it cannot be paraphrased, softened, or dropped by a later
    summarization pass. This is what "your words are the last thing to go"
    actually means mechanically.

    ### WORKING STATE
    Model-managed. Narrative of what was happening. May be re-condensed and
    aged out freely.

The digest is role="user", not "system": two system messages make
qwen_agent reject the entire request (fixed in 2cb762e). It carries an
`extra` marker so it is never mistaken for conversation nor summarized
into a digest-of-a-digest — with a content sentinel fallback, because
existing threads already contain digests written before the marker
existed.
"""

import logging
from typing import List, Optional, Tuple

from qwen_agent.llm.schema import Message

from sandbox_agent.config import (
    COMPACTION_DIGEST_DURABLE_MAX_TOKENS,
    COMPACTION_DIGEST_MAX_TOKENS,
)
from sandbox_agent.token_budget import estimate_tokens

logger = logging.getLogger(__name__)

DIGEST_SENTINEL = "[Context compacted"
DURABLE_HEADER = "### DURABLE (verbatim, never re-summarized)"
WORKING_HEADER = "### WORKING STATE"


def is_digest(msg) -> bool:
    """True for a compaction digest. Marker first, sentinel as fallback for
    digests written before the marker existed (live threads have these)."""
    if msg is None:
        return False
    extra = getattr(msg, "extra", None) or (msg.get("extra") if isinstance(msg, dict) else None)
    if isinstance(extra, dict) and isinstance(extra.get("compaction"), dict):
        return extra["compaction"].get("kind") == "digest"
    content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith(DIGEST_SENTINEL)


def parse_sections(content: str) -> Tuple[List[str], str]:
    """Split a digest body into (durable_lines, working_text).

    A parse failure must never lose DURABLE content — callers carry the
    prior durable block forward when this returns nothing.
    """
    if not isinstance(content, str):
        return [], ""
    durable: List[str] = []
    working: List[str] = []
    bucket = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("### DURABLE"):
            bucket = "d"
            continue
        if stripped.startswith("### WORKING"):
            bucket = "w"
            continue
        if bucket == "d":
            if stripped.startswith("- "):
                durable.append(stripped)
        elif bucket == "w":
            working.append(line)
    return durable, "\n".join(working).strip()


def splice_durable(existing: List[str], proposed: List[str]) -> List[str]:
    """Append genuinely-new durable bullets, preserving order, deduped by
    normalized exact match. Existing entries are never rewritten."""
    seen = {" ".join(line.lower().split()) for line in existing}
    out = list(existing)
    for line in proposed:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("- "):
            line = f"- {line}"
        key = " ".join(line.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out


def _cap_durable(durable: List[str]) -> List[str]:
    """Enforce the durable cap by dropping OLDEST bullets. This is the only
    lossy path for user-priority content, so it is logged loudly."""
    while durable and estimate_tokens("\n".join(durable)) > COMPACTION_DIGEST_DURABLE_MAX_TOKENS:
        dropped = durable.pop(0)
        logger.warning("DURABLE digest cap exceeded — dropped oldest entry: %s", dropped[:120])
    return durable


def render_digest(durable: List[str], working: str, summarized_count: int) -> str:
    durable = _cap_durable(list(durable))
    body = [
        f"{DIGEST_SENTINEL}: {summarized_count} earlier messages were summarized "
        f"to save context space. This is an automated digest, not a message from "
        f"the user. Recent messages follow verbatim.]",
        "",
        DURABLE_HEADER,
    ]
    body.extend(durable or ["- (none recorded yet)"])
    body += ["", WORKING_HEADER, working or "(nothing recorded)"]
    text = "\n".join(body)
    if estimate_tokens(text) > COMPACTION_DIGEST_MAX_TOKENS:
        # Trim WORKING STATE only — DURABLE is never sacrificed to the cap.
        allowance = max(500, COMPACTION_DIGEST_MAX_TOKENS - estimate_tokens("\n".join(durable)))
        working = working[: allowance * 4] + "\n… (working state condensed)"
        body[-1] = working
        text = "\n".join(body)
    return text


def make_digest(durable: List[str], working: str, summarized_count: int,
                generations: int = 1) -> Message:
    return Message(
        role="user",
        content=render_digest(durable, working, summarized_count),
        extra={"compaction": {"v": 1, "kind": "digest",
                              "summarized": summarized_count,
                              "generations": generations}},
    )


def merge_into(existing: Optional[Message], new_durable: List[str],
               new_working: str, newly_summarized: int) -> Message:
    """Accrete new material into an existing digest (or create the first).

    DURABLE accumulates by code. WORKING STATE is replaced by the model's
    latest rendering — it is the only part allowed to drift.
    """
    prior_durable, prior_working = ([], "")
    generations, prior_count = 0, 0
    if existing is not None:
        prior_durable, prior_working = parse_sections(str(existing.content or ""))
        extra = (getattr(existing, "extra", None) or {}).get("compaction", {})
        generations = int(extra.get("generations", 0) or 0)
        prior_count = int(extra.get("summarized", 0) or 0)
        if not prior_durable and not prior_working:
            # Unparseable (or pre-marker) digest: keep it whole as working
            # state rather than losing it.
            prior_working = str(existing.content or "")

    durable = splice_durable(prior_durable, new_durable)
    working = (new_working or prior_working or "").strip()
    return make_digest(durable, working, prior_count + newly_summarized, generations + 1)
