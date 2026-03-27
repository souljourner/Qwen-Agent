"""Tests for self-edit tools (SOUL.md, HEARTBEAT.md)."""

import os
import shutil
import tempfile

import pytest

from sandbox_agent.tools.self_edit_tools import (
    UpdateHeartbeat,
    UpdateSoul,
    _patch_section,
    _read_file,
    _write_file,
)


@pytest.fixture
def tmp_data_dir(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.tools.self_edit_tools.DATA_DIR", d)
    yield d
    shutil.rmtree(d)


class TestReadWrite:

    def test_write_creates_file(self, tmp_data_dir):
        _write_file("TEST.md", "hello")
        assert os.path.exists(os.path.join(tmp_data_dir, "TEST.md"))

    def test_read_after_write(self, tmp_data_dir):
        _write_file("TEST.md", "hello world")
        assert _read_file("TEST.md") == "hello world"

    def test_read_nonexistent_returns_empty(self, tmp_data_dir):
        assert _read_file("NONEXISTENT.md") == ""

    def test_read_falls_back_to_bundled(self):
        tool = UpdateSoul()
        result = tool.call("{}")  # No args = read
        assert "Core Identity" in result


class TestPatchSection:

    def test_replace_existing_section(self):
        content = "# Title\n\n## Foo\nold content\n\n## Bar\nbar content\n"
        result = _patch_section(content, "Foo", "new content")
        assert "new content" in result
        assert "old content" not in result
        assert "bar content" in result

    def test_append_new_section(self):
        content = "# Title\n\n## Foo\nfoo content\n"
        result = _patch_section(content, "NewSection", "new stuff")
        assert "## NewSection" in result
        assert "new stuff" in result
        assert "foo content" in result

    def test_replace_last_section(self):
        content = "# Title\n\n## First\nfirst\n\n## Last\nold last\n"
        result = _patch_section(content, "Last", "new last")
        assert "new last" in result
        assert "old last" not in result
        assert "first" in result


class TestSoulTools:

    def test_read_soul(self, tmp_data_dir):
        _write_file("SOUL.md", "# My Identity\n\n## Core\nI am a test agent.")
        tool = UpdateSoul()
        result = tool.call("{}")  # No args = read
        assert "test agent" in result

    def test_patch_section(self, tmp_data_dir):
        _write_file("SOUL.md", "# Soul\n\n## Boundaries\n- old rule\n\n## Capabilities\n- cap1\n")
        tool = UpdateSoul()
        result = tool.call('{"section": "Boundaries", "content": "- new rule 1\\n- new rule 2"}')
        assert "updated" in result.lower()
        content = _read_file("SOUL.md")
        assert "- new rule 1" in content
        assert "- old rule" not in content
        assert "- cap1" in content  # Other section preserved

    def test_append_to_section(self, tmp_data_dir):
        _write_file("SOUL.md", "# Soul\n\n## Capabilities\n- cap1\n")
        tool = UpdateSoul()
        result = tool.call('{"append_to_section": "Capabilities", "line": "- cap2"}')
        assert "cap2" in result
        content = _read_file("SOUL.md")
        assert "- cap1" in content
        assert "- cap2" in content

    def test_append_to_new_section(self, tmp_data_dir):
        _write_file("SOUL.md", "# Soul\n\n## Existing\ncontent\n")
        tool = UpdateSoul()
        result = tool.call('{"append_to_section": "NewSection", "line": "- new item"}')
        content = _read_file("SOUL.md")
        assert "## NewSection" in content
        assert "- new item" in content

    def test_read_empty_soul(self, tmp_data_dir):
        tool = UpdateSoul()
        result = tool.call("{}")
        assert result  # Falls back to bundled


class TestHeartbeatTools:

    def test_read_heartbeat(self, tmp_data_dir):
        _write_file("HEARTBEAT.md", "# Checklist\n- [ ] Check stuff")
        tool = UpdateHeartbeat()
        result = tool.call("{}")  # No args = read
        assert "Check stuff" in result

    def test_replace_heartbeat(self, tmp_data_dir):
        tool = UpdateHeartbeat()
        result = tool.call('{"content": "# Checklist\\n- [ ] New item\\n- [x] Done item"}')
        assert "replaced" in result.lower()
        content = _read_file("HEARTBEAT.md")
        assert "- [ ] New item" in content
        assert "- [x] Done item" in content

    def test_add_item(self, tmp_data_dir):
        _write_file("HEARTBEAT.md", "# Checklist\n- [ ] Existing\n")
        tool = UpdateHeartbeat()
        result = tool.call('{"add_item": "Check disk space"}')
        assert "Check disk space" in result
        content = _read_file("HEARTBEAT.md")
        assert "- [ ] Existing" in content
        assert "- [ ] Check disk space" in content

    def test_add_then_read(self, tmp_data_dir):
        _write_file("HEARTBEAT.md", "# Checklist\n")
        tool = UpdateHeartbeat()
        tool.call('{"add_item": "Task A"}')
        tool.call('{"add_item": "Task B"}')
        result = tool.call("{}")  # Read
        assert "Task A" in result
        assert "Task B" in result
