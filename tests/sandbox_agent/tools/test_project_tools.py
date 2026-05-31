"""Tests for the rename_project tool."""

import json

import pytest

from sandbox_agent.tools import project_tools as pt
from sandbox_agent.tools.project_tools import RenameProject, UpdateProject


@pytest.fixture
def projects(tmp_path, monkeypatch):
    """Isolate PROJECTS_DIR/DATA_DIR under tmp and no-op autocommit (no real git)."""
    pdir = tmp_path / "projects"
    pdir.mkdir()
    monkeypatch.setattr(pt, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pt, "PROJECTS_DIR", str(pdir))
    monkeypatch.setattr(pt, "autocommit", lambda *a, **k: None)
    return pdir


def _make_project(pdir, slug, name=None, body="hello"):
    d = pdir / slug
    d.mkdir()
    meta = {
        "name": name or slug,
        "description": "desc",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    (d / ".project.json").write_text(json.dumps(meta))
    (d / "notes.md").write_text(body)
    return d


def test_renames_dir_and_preserves_files(projects):
    _make_project(projects, "old-name", body="keep me")
    out = RenameProject().call({"project": "old-name", "new_name": "New Name"})
    assert "old-name" in out and "new-name" in out
    assert not (projects / "old-name").exists()
    assert (projects / "new-name" / "notes.md").read_text() == "keep me"


def test_slugifies_new_name(projects):
    _make_project(projects, "old-name")
    RenameProject().call({"project": "old-name", "new_name": "My Cool Project"})
    assert (projects / "my-cool-project").is_dir()


def test_updates_metadata_name_and_timestamp(projects):
    _make_project(projects, "old-name")
    RenameProject().call({"project": "old-name", "new_name": "Fresh Title"})
    meta = json.loads((projects / "fresh-title" / ".project.json").read_text())
    assert meta["name"] == "Fresh Title"
    assert meta["updated_at"] != "2026-01-01T00:00:00"


def test_source_not_found(projects):
    out = RenameProject().call({"project": "ghost", "new_name": "whatever"})
    assert "not found" in out.lower()


def test_destination_exists_blocks(projects):
    _make_project(projects, "alpha", body="A")
    _make_project(projects, "beta", body="B")
    out = RenameProject().call({"project": "alpha", "new_name": "Beta"})
    assert "already exists" in out.lower()
    assert (projects / "alpha" / "notes.md").read_text() == "A"  # source untouched
    assert (projects / "beta" / "notes.md").read_text() == "B"   # dest untouched


def test_same_name_is_noop(projects):
    _make_project(projects, "same")
    out = RenameProject().call({"project": "same", "new_name": "Same"})
    assert "already named" in out.lower()
    assert (projects / "same").exists()


def test_invalid_new_name_rejected(projects):
    _make_project(projects, "valid")
    out = RenameProject().call({"project": "valid", "new_name": "!!!"})
    assert "invalid" in out.lower()
    assert (projects / "valid").exists()  # untouched


def test_autocommit_invoked(projects, monkeypatch):
    calls = []
    monkeypatch.setattr(pt, "autocommit", lambda fn, msg: calls.append((fn, msg)))
    _make_project(projects, "to-commit")
    RenameProject().call({"project": "to-commit", "new_name": "committed"})
    assert len(calls) == 1
    assert calls[0][0].startswith("projects")


def test_registered():
    from qwen_agent.tools.base import TOOL_REGISTRY
    assert "rename_project" in TOOL_REGISTRY


# --- update_project -------------------------------------------------------

def test_update_changes_description(projects):
    _make_project(projects, "alpha")
    out = UpdateProject().call({"project": "alpha", "description": "new shiny goal"})
    meta = json.loads((projects / "alpha" / ".project.json").read_text())
    assert meta["description"] == "new shiny goal"
    assert "alpha" in out


def test_update_bumps_timestamp(projects):
    _make_project(projects, "alpha")
    UpdateProject().call({"project": "alpha", "description": "x"})
    meta = json.loads((projects / "alpha" / ".project.json").read_text())
    assert meta["updated_at"] != "2026-01-01T00:00:00"


def test_update_preserves_other_meta(projects):
    _make_project(projects, "alpha", name="Alpha Display")
    UpdateProject().call({"project": "alpha", "description": "changed"})
    meta = json.loads((projects / "alpha" / ".project.json").read_text())
    assert meta["name"] == "Alpha Display"            # untouched
    assert meta["created_at"] == "2026-01-01T00:00:00"  # untouched


def test_update_does_not_touch_readme(projects):
    d = _make_project(projects, "alpha")
    (d / "README.md").write_text("# Hand-edited README\n\ncustom body")
    UpdateProject().call({"project": "alpha", "description": "new desc"})
    assert (d / "README.md").read_text() == "# Hand-edited README\n\ncustom body"


def test_update_project_not_found(projects):
    out = UpdateProject().call({"project": "ghost", "description": "x"})
    assert "not found" in out.lower()


def test_update_autocommit_invoked(projects, monkeypatch):
    calls = []
    monkeypatch.setattr(pt, "autocommit", lambda fn, msg: calls.append((fn, msg)))
    _make_project(projects, "alpha")
    UpdateProject().call({"project": "alpha", "description": "x"})
    assert len(calls) == 1


def test_update_registered():
    from qwen_agent.tools.base import TOOL_REGISTRY
    assert "update_project" in TOOL_REGISTRY
