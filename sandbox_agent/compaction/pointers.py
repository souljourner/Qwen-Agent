"""L0 of the retention ladder: replace recoverable bulk with a pointer.

A `project_write_file` call carries the entire file body in its arguments,
but that body is ON DISK — the agent can re-read it. Replacing it with
"[elided 8,240 chars — write to notes.md; sha256:a3f19c2b]" is lossless in
practice and free (no LLM).

SAFETY IS AN ALLOW-LIST, not a heuristic. Deltas can never be pointer-ized:
a `mode="edit"` write or an `apply_patch` describes a CHANGE, and the file
on disk holds only the post-state — "re-read the file" cannot recover what
changed. Code sent to exec/code_interpreter isn't on disk at all and IS the
reasoning. Sent emails are irreversible side effects the agent must
remember exactly.

A stale pointer is strictly safer than a stale inline copy: a pointer is a
past-tense claim ("at this step we wrote N chars to P") that stays true
after P changes, whereas an inline copy silently becomes a lie.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Set, Tuple

from qwen_agent.llm.schema import Message

from sandbox_agent.config import POINTER_MIN_CHARS

logger = logging.getLogger(__name__)

# tool name -> (argument holding recoverable bulk, argument naming the target)
POINTERABLE: Dict[str, Tuple[str, Optional[str]]] = {
    "project_write_file": ("content", "path"),
    "update_soul": ("content", None),
    "update_heartbeat": ("content", None),
    "add_memory": ("content", None),
}

# Never pointer-ize these, whatever their size.
FORBIDDEN: Set[str] = {
    "project_apply_patch",   # a delta — disk holds only the post-state
    "code_interpreter",      # the code IS the reasoning, not on disk
    "exec",                  # ditto
    "send_email",            # irreversible side effect
}


def _is_delta_write(args: dict) -> bool:
    """A write in edit mode is a delta — unrecoverable from the file."""
    if args.get("mode") not in (None, "", "write", "append"):
        return True
    return bool(args.get("old_text") or args.get("new_text"))


def _render(n_chars: int, target: Optional[str], digest: str) -> str:
    where = f" to {target}" if target else ""
    return (f"[elided {n_chars:,} chars{where}; sha256:{digest}; "
            f"content is on disk — re-read it if you need it]")


def pointerize(messages: List[Message], *,
               protect_indices: Optional[Set[int]] = None) -> Tuple[List[Message], int]:
    """Replace recoverable bulk in tool-call arguments with pointers.

    Returns (messages, chars_saved). `protect_indices` (the verbatim tail)
    is never touched — that is where "I just wrote this and may edit it
    next" lives, so no additional per-path rule is needed. Everything
    eligible OUTSIDE the tail is old enough that re-reading the file is the
    right move. JSON shape is preserved exactly: only one string value
    changes, so the model still sees a well-formed call it could have made.
    """
    protect = protect_indices or set()
    out, saved = [], 0
    for i, msg in enumerate(messages):
        fc = getattr(msg, "function_call", None)
        if i in protect or not fc or fc.name in FORBIDDEN or fc.name not in POINTERABLE:
            out.append(msg)
            continue
        content_key, target_key = POINTERABLE[fc.name]
        try:
            args = json.loads(fc.arguments or "{}")
        except (ValueError, TypeError):
            out.append(msg)  # unparseable — never guess
            continue
        if not isinstance(args, dict) or _is_delta_write(args):
            out.append(msg)
            continue
        body = args.get(content_key)
        if not isinstance(body, str) or len(body) < POINTER_MIN_CHARS:
            out.append(msg)
            continue
        target = str(args.get(target_key) or "") if target_key else fc.name
        digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:8]
        args[content_key] = _render(len(body), target or None, digest)
        new_args = json.dumps(args)
        saved += max(0, len(fc.arguments or "") - len(new_args))
        out.append(Message(
            role=msg.role,
            content=msg.content,
            name=msg.name,
            function_call=type(fc)(name=fc.name, arguments=new_args),
            extra=msg.extra,
        ))

    if saved:
        logger.info("Pointer-ization saved %d chars", saved)
    return out, saved
