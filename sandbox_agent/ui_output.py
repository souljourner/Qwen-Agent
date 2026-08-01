"""UI payload bounds for Chainlit step outputs.

Steps re-send their ENTIRE output on every update; huge outputs starve the
SPA's render loop (2026-08-01 UI audit). Full text always stays in the
agent's context/history — these caps bound only the DOM/socket payload.
"""

_STEP_OUTPUT_HEAD_CHARS = 15_000
_STEP_OUTPUT_TAIL_CHARS = 2_000
_STEP_OUTPUT_CAP_CHARS = 20_000


def _cap_step_output(text: str, keep: str = "both") -> str:
    """Bound a cl.Step output payload for the UI. keep="both" → head+tail
    (final results); keep="tail" → latest lines only (live stdout — that's
    what the user is watching)."""
    if not isinstance(text, str) or len(text) <= _STEP_OUTPUT_CAP_CHARS:
        return text
    omitted = len(text) - _STEP_OUTPUT_HEAD_CHARS - _STEP_OUTPUT_TAIL_CHARS
    marker = (f"\n\n… [{omitted:,} chars omitted from this view — "
              f"the agent received the full output] …\n\n")
    if keep == "tail":
        cut = len(text) - _STEP_OUTPUT_CAP_CHARS
        return f"… [{cut:,} earlier chars omitted] …\n" + text[-_STEP_OUTPUT_CAP_CHARS:]
    return text[:_STEP_OUTPUT_HEAD_CHARS] + marker + text[-_STEP_OUTPUT_TAIL_CHARS:]
