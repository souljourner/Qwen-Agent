"""Tests for compact_memories tool + read_memories archive param + heartbeat
over-cap trigger + pre-skills migration."""

import json
import os

import pytest

import sandbox_agent.tools.self_edit_tools as se
from sandbox_agent.tools.self_edit_tools import CompactMemories, ReadMemories


CANONICAL = """# MEMORIES.md

## User Preferences
- [2026-01-01] Prefers terse replies.

## Facts & Knowledge
- [2026-01-02] Fact A stated at great length with many redundant clauses repeated over and over,
  Fact B duplicated. Fact B duplicated again with extra words. Fact B once more for good measure,
  plus a long trailing rationale that a compaction pass would fold into a single concise line
  because none of the extra verbiage changes future behavior in any way whatsoever.

## Technical Notes
- [2026-01-03] Note.

## Task Learnings
- [2026-01-04] Learned a thing.
"""

COMPACTED = """# MEMORIES.md

## User Preferences
- [2026-01-01] Prefers terse replies.

## Facts & Knowledge
- [2026-01-02] Facts A+B.

## Technical Notes
- [2026-01-03] Note.

## Task Learnings
"""


@pytest.fixture
def mem_env(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(se, "autocommit", lambda *a, **k: None, raising=False)
    (tmp_path / "MEMORIES.md").write_text(CANONICAL)
    return tmp_path


def test_compact_archives_old_verbatim(mem_env):
    out = CompactMemories().call({"new_content": COMPACTED})
    assert not out.lower().startswith("error")
    archive_dir = mem_env / "memories_archive"
    files = list(archive_dir.glob("*.md"))
    assert len(files) == 1
    assert CANONICAL in files[0].read_text()      # verbatim snapshot
    new = (mem_env / "MEMORIES.md").read_text()
    assert "Facts A+B" in new
    assert "## Archives" in new                    # reference section inserted


def test_compact_appends_to_monthly_archive(mem_env):
    CompactMemories().call({"new_content": COMPACTED})
    shorter = COMPACTED.replace("- [2026-01-03] Note.\n", "")
    out = CompactMemories().call({"new_content": shorter})
    assert not out.lower().startswith("error")
    files = list((mem_env / "memories_archive").glob("*.md"))
    assert len(files) == 1                         # same month → same file
    assert files[0].read_text().count("# Snapshot") == 2


def test_compact_rejects_longer_content(mem_env):
    out = CompactMemories().call({"new_content": CANONICAL + "\nmore stuff " * 50})
    assert out.lower().startswith("error")


def test_compact_rejects_missing_sections(mem_env):
    out = CompactMemories().call({"new_content": "# MEMORIES.md\n\nno sections at all\n"})
    assert out.lower().startswith("error")


def test_compact_stamps_last_compaction(mem_env):
    CompactMemories().call({"new_content": COMPACTED})
    stamp = mem_env / "memories_archive" / ".last_compaction"
    assert stamp.exists()
    meta = json.loads(stamp.read_text())
    assert meta["old_len"] > meta["new_len"]


def test_read_memories_archive_list_and_fetch(mem_env):
    CompactMemories().call({"new_content": COMPACTED})
    listing = ReadMemories().call({"archive": "list"})
    assert ".md" in listing
    month = list((mem_env / "memories_archive").glob("*.md"))[0].stem
    body = ReadMemories().call({"archive": month})
    assert "Fact B duplicated" in body


def test_read_memories_default_unchanged(mem_env):
    out = ReadMemories().call({})
    assert "Prefers terse replies" in out


# --- heartbeat over-cap trigger ----------------------------------------------

class _StubQueue:
    def get_due_tasks(self):
        return []


@pytest.fixture
def hb(tmp_path, monkeypatch):
    from sandbox_agent.heartbeat.heartbeat_runner import HeartbeatRunner
    calls = []

    def runner(messages):
        calls.append(messages)
        return []

    # Empty checklist so the bundled default's pending items don't force an
    # LLM run — these tests isolate the memory-maintenance trigger.
    (tmp_path / "HEARTBEAT.md").write_text("# Heartbeat Checklist\n")
    hb = HeartbeatRunner(runner=runner, task_queue=_StubQueue(), data_dir=str(tmp_path))
    return hb, calls, tmp_path


def test_heartbeat_triggers_memory_maintenance_over_cap(hb, monkeypatch):
    runner, calls, tmp = hb
    import sandbox_agent.heartbeat.heartbeat_runner as hr
    monkeypatch.setattr(hr, "MEMORIES_COMPACT_TRIGGER_CHARS", 100, raising=False)
    (tmp / "MEMORIES.md").write_text("x" * 500)
    runner.run_once()
    assert calls, "over-cap memories should force a heartbeat LLM run"
    text = str(calls[0][0].content)
    assert "Memory Maintenance" in text
    assert "compact_memories" in text


def test_heartbeat_quiet_under_cap(hb):
    runner, calls, tmp = hb
    (tmp / "MEMORIES.md").write_text("small")
    assert runner.run_once() is None
    assert not calls                               # no LLM call at all


def test_heartbeat_respects_min_interval(hb, monkeypatch):
    runner, calls, tmp = hb
    import sandbox_agent.heartbeat.heartbeat_runner as hr
    monkeypatch.setattr(hr, "MEMORIES_COMPACT_TRIGGER_CHARS", 100, raising=False)
    (tmp / "MEMORIES.md").write_text("x" * 500)
    # Fresh compaction stamp → suppressed
    os.makedirs(tmp / "memories_archive", exist_ok=True)
    import time
    (tmp / "memories_archive" / ".last_compaction").write_text(
        json.dumps({"ts": time.time(), "old_len": 1, "new_len": 1}))
    assert runner.run_once() is None
    assert not calls


# --- migration -----------------------------------------------------------------

def test_migration_archives_stale_soul(tmp_path, monkeypatch):
    import sandbox_agent.migrations as mig
    monkeypatch.setattr(mig, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mig, "autocommit", lambda *a, **k: None, raising=False)
    stale = "# Old SOUL\n\n## Task Learnings\nlots of accumulated edits\n"
    (tmp_path / "SOUL.md").write_text(stale)

    mig.migrate_pre_skills()
    assert not (tmp_path / "SOUL.md").exists()
    archived = tmp_path / "soul_archive" / "SOUL-pre-skills.md"
    assert archived.read_text() == stale
    mem = (tmp_path / "MEMORIES.md").read_text()
    assert "soul_archive/SOUL-pre-skills.md" in mem

    # Idempotent: second run is a no-op
    mig.migrate_pre_skills()
    assert (tmp_path / "MEMORIES.md").read_text() == mem


def test_migration_leaves_skills_aware_soul(tmp_path, monkeypatch):
    import sandbox_agent.migrations as mig
    monkeypatch.setattr(mig, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mig, "autocommit", lambda *a, **k: None, raising=False)
    (tmp_path / "SOUL.md").write_text("# New SOUL\n\n## Skills\n- x\n")
    mig.migrate_pre_skills()
    assert (tmp_path / "SOUL.md").exists()         # untouched


def test_migration_noop_without_soul(tmp_path, monkeypatch):
    import sandbox_agent.migrations as mig
    monkeypatch.setattr(mig, "DATA_DIR", str(tmp_path))
    mig.migrate_pre_skills()                       # must not raise
    assert not (tmp_path / "soul_archive").exists()
