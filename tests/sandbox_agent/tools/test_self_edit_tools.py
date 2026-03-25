"""Tests for self-edit tools (SOUL.md, HEARTBEAT.md)."""

import os
import shutil
import tempfile

import pytest

from sandbox_agent.tools.self_edit_tools import (
    ReadHeartbeat,
    ReadSoul,
    UpdateHeartbeat,
    UpdateSoul,
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
        """Without monkeypatching DATA_DIR, SOUL.md should read the bundled default."""
        tool = ReadSoul()
        result = tool.call("{}")
        assert "Core Identity" in result


class TestSoulTools:

    def test_read_soul(self, tmp_data_dir):
        _write_file("SOUL.md", "# My Identity\nI am a test agent.")
        tool = ReadSoul()
        result = tool.call("{}")
        assert "test agent" in result

    def test_update_soul(self, tmp_data_dir):
        tool = UpdateSoul()
        result = tool.call('{"content": "# New Soul\\nI am updated."}')
        assert "updated" in result
        assert _read_file("SOUL.md") == "# New Soul\nI am updated."

    def test_read_empty_soul(self, tmp_data_dir):
        tool = ReadSoul()
        result = tool.call("{}")
        # Should fall back to bundled or show empty message
        assert result  # Not empty


class TestHeartbeatTools:

    def test_read_heartbeat(self, tmp_data_dir):
        _write_file("HEARTBEAT.md", "# Checklist\n- [ ] Check stuff")
        tool = ReadHeartbeat()
        result = tool.call("{}")
        assert "Check stuff" in result

    def test_update_heartbeat(self, tmp_data_dir):
        tool = UpdateHeartbeat()
        result = tool.call('{"content": "# Checklist\\n- [ ] New item\\n- [x] Done item"}')
        assert "updated" in result
        content = _read_file("HEARTBEAT.md")
        assert "- [ ] New item" in content
        assert "- [x] Done item" in content

    def test_update_then_read(self, tmp_data_dir):
        UpdateHeartbeat().call('{"content": "# Updated\\n- [ ] Task A"}')
        result = ReadHeartbeat().call("{}")
        assert "Task A" in result
