"""Render-time cap for MEMORIES.md injection into the system prompt.

The live memories file grows unbounded via add_memory; injecting it wholesale
costs thousands of tokens on every LLM call. render_memories_capped keeps the
newest entries within a budget while preserving the file's section structure.
Deterministic for a given file, so the system prefix stays KV-cache-stable
between memory writes.
"""

import re
from typing import List, Optional, Tuple

_ENTRY_RE = re.compile(r"^- \[(\d{4}-\d{2}-\d{2})\] ")

_OMISSION_NOTE = ("\n_({n} older memories not shown — call read_memories for the full "
                  "file; archives in memories_archive/)_\n")


def _parse(content: str) -> Tuple[str, List[Tuple[Optional[str], List[Tuple[str, str]]]]]:
    """Split into (preamble, [(section_header_line_or_None, [(date, entry_text)])]).

    An entry is a `- [YYYY-MM-DD] ...` line plus its continuation lines (until
    the next entry or `## ` header). Non-entry stray lines inside a section are
    attached to the preceding entry (or treated as section preamble)."""
    lines = content.split("\n")
    preamble_lines: List[str] = []
    sections: List[Tuple[Optional[str], List[Tuple[str, str]], List[str]]] = []
    current: Optional[List] = None  # [header, entries, section_preamble_lines]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            current = [line, [], []]
            sections.append(current)  # type: ignore[arg-type]
            i += 1
            continue
        m = _ENTRY_RE.match(line)
        if m and current is not None:
            entry_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith("## ") and not _ENTRY_RE.match(lines[i]):
                entry_lines.append(lines[i])
                i += 1
            current[1].append((m.group(1), "\n".join(entry_lines).rstrip("\n")))
            continue
        if current is None:
            preamble_lines.append(line)
        else:
            current[2].append(line)
        i += 1

    preamble = "\n".join(preamble_lines).rstrip("\n")
    return preamble, [(s[0], s[1], s[2]) for s in sections]  # type: ignore[return-value]


def render_memories_capped(content: str, max_chars: int = 6000) -> str:
    """Return `content` unchanged if it fits; otherwise re-render with the
    newest dated entries first (dropping the oldest) plus an omission note.
    The `## Archives` section is always kept whole."""
    if len(content) <= max_chars:
        return content

    preamble, sections = _parse(content)
    all_entries = [e for _, entries, _ in sections for e in entries]
    if not all_entries:
        # Unparseable — head-truncate with a note.
        return content[: max(0, max_chars - 60)].rstrip() + "\n_(older content truncated — read_memories for the full file)_\n"

    # Skeleton cost: preamble + headers + section preambles + Archives entries.
    def _sec_is_archives(header: Optional[str]) -> bool:
        return bool(header) and "archives" in header.lower()

    skeleton = len(preamble)
    for header, entries, sec_pre in sections:
        skeleton += len(header or "") + 2
        skeleton += sum(len(l) + 1 for l in sec_pre if l.strip())
        if _sec_is_archives(header):
            skeleton += sum(len(t) + 1 for _, t in entries)

    budget = max_chars - skeleton - len(_OMISSION_NOTE) - 20

    # Newest-first greedy selection over non-Archives entries. Ties broken by
    # later file position winning (more recently appended).
    candidates = []
    pos = 0
    for si, (header, entries, _) in enumerate(sections):
        if _sec_is_archives(header):
            continue
        for ei, (date, text) in enumerate(entries):
            candidates.append((date, pos, si, ei, text))
            pos += 1
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    selected = set()
    used = 0
    for date, p, si, ei, text in candidates:
        cost = len(text) + 1
        if used + cost > budget:
            continue
        selected.add((si, ei))
        used += cost
    omitted = len(candidates) - len(selected)

    # Re-render in original order.
    out: List[str] = []
    if preamble.strip():
        out.append(preamble)
    for si, (header, entries, sec_pre) in enumerate(sections):
        if header:
            out.append("")
            out.append(header)
        keep_all = _sec_is_archives(header)
        for l in sec_pre:
            if l.strip():
                out.append(l)
        for ei, (date, text) in enumerate(entries):
            if keep_all or (si, ei) in selected:
                out.append(text)
    rendered = "\n".join(out).rstrip("\n") + "\n"
    if omitted > 0:
        rendered += _OMISSION_NOTE.format(n=omitted)
    return rendered
