"""Tests for read_skill + auto-inject (skills system)."""

import pytest

from qwen_agent.llm.schema import ContentItem, Message

import sandbox_agent.tools.skill_tools as sk
from sandbox_agent.tools.skill_tools import ReadSkill, maybe_inject_skill


@pytest.fixture
def skills_env(tmp_path, monkeypatch):
    """Isolated DATA_DIR; bundled skills dir is the real one."""
    monkeypatch.setattr(sk, "DATA_DIR", str(tmp_path))
    return tmp_path


def test_list_mode_shows_all_bundled(skills_env):
    out = ReadSkill().call({})
    for name in ("token-efficiency", "browser-automation", "file-organization",
                 "long-documents", "pipelines", "memory-maintenance"):
        assert name in out


def test_read_by_name(skills_env):
    out = ReadSkill().call({"name": "browser-automation"})
    assert out.startswith("[skill:browser-automation]")
    assert "2FA" in out
    assert out.rstrip().endswith("[end skill]")


def test_data_dir_override_wins(skills_env):
    (skills_env / "skills").mkdir()
    (skills_env / "skills" / "browser-automation.md").write_text(
        "# Browser (custom)\n> my override\n\ncustom body\n")
    out = ReadSkill().call({"name": "browser-automation"})
    assert "custom body" in out
    assert "2FA" not in out


def test_unknown_name_returns_listing(skills_env):
    out = ReadSkill().call({"name": "does-not-exist"})
    assert "token-efficiency" in out       # fell back to the listing
    assert "does-not-exist" in out         # and says what was asked


def test_traversal_rejected(skills_env):
    out = ReadSkill().call({"name": "../SOUL"})
    assert "[skill:" not in out            # no file content returned


def test_oversize_truncated(skills_env, monkeypatch):
    monkeypatch.setattr(sk, "SKILL_MAX_CHARS", 100)
    out = ReadSkill().call({"name": "token-efficiency"})
    assert "truncated" in out.lower()
    assert len(out) < 600


# --- auto-inject -------------------------------------------------------------

def test_injects_on_first_browser_tool(skills_env):
    result = maybe_inject_skill("browser_navigate", "opened page", [])
    assert result.startswith("[skill:browser-automation]")
    assert "opened page" in result
    assert "2FA" in result


def test_skips_when_marker_in_history_str(skills_env):
    history = [Message(role="function", content="[skill:browser-automation] loaded earlier")]
    result = maybe_inject_skill("browser_click", "clicked", history)
    assert result == "clicked"


def test_skips_when_marker_in_multimodal_history(skills_env):
    history = [Message(role="user", content=[
        ContentItem(text="[skill:browser-automation] earlier"),
        ContentItem(image="data:image/png;base64,xxx"),
    ])]
    result = maybe_inject_skill("browser_type", "typed", history)
    assert result == "typed"


def test_multimodal_result_gets_note_on_text_item(skills_env):
    result = [ContentItem(image="data:image/png;base64,xxx"), ContentItem(text="screenshot taken")]
    out = maybe_inject_skill("browser_screenshot", result, [])
    texts = [ci.text for ci in out if getattr(ci, "text", None)]
    assert any(t.startswith("[skill:browser-automation]") for t in texts)


def test_non_trigger_tools_untouched(skills_env):
    assert maybe_inject_skill("web_search", "results", []) == "results"
    assert maybe_inject_skill("exec", "done", []) == "done"


def test_registered_and_in_tool_list():
    from qwen_agent.tools.base import TOOL_REGISTRY
    import sandbox_agent.config as cfg
    assert "read_skill" in TOOL_REGISTRY
    assert "read_skill" in cfg.TOOL_LIST
