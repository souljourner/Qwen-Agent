"""Tests for render_memories_capped (render-time MEMORIES.md cap) and
load_system_message integration (DATA_DIR SOUL override + memories cap)."""

import pytest

from sandbox_agent.memories import render_memories_capped


SMALL = """# MEMORIES.md

## User Preferences
- [2026-01-01] Prefers terse replies.

## Facts & Knowledge

## Technical Notes
- [2026-01-02] vLLM needs top-level chat_template_kwargs.

## Task Learnings
"""


def _entry(date: str, i: int, pad: int = 300) -> str:
    return f"- [{date}] entry {i} " + ("x" * pad)


def test_under_cap_passthrough_byte_identical():
    assert render_memories_capped(SMALL, max_chars=6000) == SMALL


def test_over_cap_keeps_newest_first():
    old = _entry("2026-01-01", 1)
    mid = _entry("2026-03-01", 2)
    new = _entry("2026-06-01", 3)
    content = f"# M\n\n## Technical Notes\n{old}\n{mid}\n{new}\n"
    out = render_memories_capped(content, max_chars=len(content) - 200)
    assert "entry 3" in out          # newest kept
    assert "entry 1" not in out      # oldest dropped first
    assert "## Technical Notes" in out
    assert "not shown" in out        # omission note appended


def test_structure_and_order_preserved():
    content = (
        "# M\n\n"
        "## User Preferences\n" + _entry("2026-06-01", 1) + "\n\n"
        "## Technical Notes\n" + _entry("2026-06-02", 2) + "\n" + _entry("2026-01-01", 3) + "\n"
    )
    out = render_memories_capped(content, max_chars=len(content) - 200)
    # Section order intact
    assert out.index("## User Preferences") < out.index("## Technical Notes")
    # Dropped entry came from the oldest date
    assert "entry 3" not in out
    assert "entry 1" in out and "entry 2" in out


def test_multiline_entries_kept_whole():
    entry = "- [2026-06-01] first line\n  continuation line one\n  continuation two"
    content = f"# M\n\n## Task Learnings\n{entry}\n" + _entry("2026-01-01", 9, pad=600) + "\n"
    out = render_memories_capped(content, max_chars=len(content) - 400)
    assert "continuation two" in out       # multiline entry survives intact
    assert "entry 9" not in out


def test_deterministic():
    content = "# M\n\n## Technical Notes\n" + "\n".join(
        _entry(f"2026-0{d}-01", d) for d in range(1, 7)) + "\n"
    a = render_memories_capped(content, max_chars=900)
    b = render_memories_capped(content, max_chars=900)
    assert a == b


def test_unparseable_falls_back_to_head_truncate():
    blob = "no dated entries here " * 300
    out = render_memories_capped(blob, max_chars=500)
    assert len(out) < len(blob)
    assert out.startswith("no dated entries")
    assert "not shown" in out or "truncated" in out.lower()


# --- load_system_message integration ----------------------------------------

@pytest.fixture
def sys_env(tmp_path, monkeypatch):
    import sandbox_agent.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    return tmp_path


def test_load_system_message_caps_memories(sys_env):
    import sandbox_agent.config as cfg
    big = "# MEMORIES.md\n\n## Technical Notes\n" + "\n".join(
        f"- [2026-01-{i:02d}] note {i} " + "y" * 200 for i in range(1, 60)) + "\n"
    (sys_env / "MEMORIES.md").write_text(big)
    sm = cfg.load_system_message()
    # The injected memories section must be capped, not the whole 12k+ blob.
    assert len(sm) < len(big) + 10_000
    assert "note 59" in sm            # newest included
    assert "note 1 " not in sm        # oldest dropped


def test_load_system_message_honors_data_dir_soul(sys_env):
    import sandbox_agent.config as cfg
    (sys_env / "SOUL.md").write_text("# SOUL override\n\n## Skills\ncustom identity here\n")
    sm = cfg.load_system_message()
    assert "custom identity here" in sm


def test_bundled_soul_stays_slim_and_index_matches_skill_files():
    import os
    import re
    base = os.path.join(os.path.dirname(__import__("sandbox_agent").__file__))
    soul = open(os.path.join(base, "SOUL.md")).read()
    assert len(soul) <= 8000, f"bundled SOUL.md re-bloating: {len(soul)} chars"
    # Every name in the Skills index must have a bundled skill file.
    skills_section = soul.split("## Skills", 1)[1].split("##", 1)[0]
    names = re.findall(r"^- ([a-z0-9-]+):", skills_section, flags=re.M)
    assert len(names) >= 5
    for name in names:
        assert os.path.exists(os.path.join(base, "skills", f"{name}.md")), f"missing skill file {name}"
