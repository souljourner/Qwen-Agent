"""Content sanitization for web tool outputs to defend against prompt injection."""

import re
import unicodedata

from sandbox_agent.config import CHARS_PER_TOKEN, MAX_TOOL_OUTPUT_TOKENS

DEFAULT_MAX_LENGTH = MAX_TOOL_OUTPUT_TOKENS * CHARS_PER_TOKEN  # Convert token budget to chars

# Chat template special tokens that could be used for prompt injection
SPECIAL_TOKEN_PATTERNS = [
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    r"<\|end\|>",
    r"<\|assistant\|>",
    r"<\|user\|>",
    r"<\|system\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"### (?:SYSTEM|USER|ASSISTANT|HUMAN|AI):",
    r"<\|(?:pad|unk|mask)\|>",
]

_SPECIAL_TOKEN_RE = re.compile("|".join(SPECIAL_TOKEN_PATTERNS), re.IGNORECASE)

# Unicode categories for zero-width and formatting characters
_INVISIBLE_CATEGORIES = {"Cf", "Cc", "Cs", "Co"}
# Specific characters to always strip
_DANGEROUS_CODEPOINTS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
    "\u2060",  # word joiner
    "\u2061",  # function application
    "\u2062",  # invisible times
    "\u2063",  # invisible separator
    "\u2064",  # invisible plus
    "\ufeff",  # zero-width no-break space (BOM)
    "\ufff9",  # interlinear annotation anchor
    "\ufffa",  # interlinear annotation separator
    "\ufffb",  # interlinear annotation terminator
}


def _strip_special_tokens(text: str) -> str:
    """Remove chat template special tokens that could enable role-switching attacks."""
    return _SPECIAL_TOKEN_RE.sub("", text)


def _strip_invisible_chars(text: str) -> str:
    """Remove zero-width, RTL-override, and other invisible formatting characters."""
    result = []
    for ch in text:
        if ch in _DANGEROUS_CODEPOINTS:
            continue
        cat = unicodedata.category(ch)
        # Keep normal whitespace (spaces, tabs, newlines) which are Zs/Zl/Zp or \n\t
        if cat in _INVISIBLE_CATEGORIES and ch not in ("\n", "\r", "\t"):
            continue
        result.append(ch)
    return "".join(result)


def sanitize_web_content(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Sanitize web content to defend against prompt injection.

    1. Strip chat-template special tokens
    2. Remove invisible/formatting Unicode characters
    3. Truncate to max length
    4. Wrap in delimiters so the model treats it as tool output

    Args:
        text: Raw text from web tool
        max_length: Maximum character length (default 8000)

    Returns:
        Sanitized text wrapped in [TOOL_OUTPUT] delimiters
    """
    if not text:
        return "[TOOL_OUTPUT]\n(empty)\n[/TOOL_OUTPUT]"

    text = _strip_special_tokens(text)
    text = _strip_invisible_chars(text)

    # Strip our own delimiters from content to prevent delimiter injection
    text = text.replace("[TOOL_OUTPUT]", "").replace("[/TOOL_OUTPUT]", "")

    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"

    return f"[TOOL_OUTPUT]\n{text}\n[/TOOL_OUTPUT]"
