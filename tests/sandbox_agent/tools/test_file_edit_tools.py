"""Tests for project_write_file edit/append modes and project_apply_patch."""

import json
import os
from textwrap import dedent

import pytest


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """Create a temp project directory and patch DATA_DIR."""
    import sandbox_agent.tools.project_tools as pt
    import sandbox_agent.config as cfg

    data_dir = str(tmp_path)
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(pt, "DATA_DIR", data_dir)
    monkeypatch.setattr(pt, "PROJECTS_DIR", os.path.join(data_dir, "projects"))

    # Create a project
    pdir = os.path.join(data_dir, "projects", "test-proj")
    os.makedirs(pdir)
    with open(os.path.join(pdir, ".project.json"), "w") as f:
        json.dump({"name": "test-proj", "description": "test"}, f)

    return pdir


class TestWriteFileAppendMode:
    def test_append_creates_file(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectWriteFile
        tool = ProjectWriteFile()
        result = tool.call(json.dumps({
            "project": "test-proj",
            "path": "doc.md",
            "content": "# Title\n",
            "mode": "append",
        }))
        assert "Appended" in result
        with open(os.path.join(project_dir, "doc.md")) as f:
            assert f.read() == "# Title\n"

    def test_append_adds_to_existing(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectWriteFile
        tool = ProjectWriteFile()

        # Write initial content
        tool.call(json.dumps({
            "project": "test-proj", "path": "doc.md",
            "content": "# Section 1\nContent.\n",
        }))
        # Append
        tool.call(json.dumps({
            "project": "test-proj", "path": "doc.md",
            "content": "\n# Section 2\nMore content.\n",
            "mode": "append",
        }))
        with open(os.path.join(project_dir, "doc.md")) as f:
            content = f.read()
        assert "Section 1" in content
        assert "Section 2" in content


class TestWriteFileEditMode:
    def test_edit_replaces_text(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectWriteFile
        tool = ProjectWriteFile()

        # Create file
        with open(os.path.join(project_dir, "doc.md"), "w") as f:
            f.write("Hello world\nFoo bar\nGoodbye")

        result = tool.call(json.dumps({
            "project": "test-proj", "path": "doc.md",
            "mode": "edit",
            "old_text": "Foo bar",
            "new_text": "Baz qux",
        }))
        assert "Edited" in result
        with open(os.path.join(project_dir, "doc.md")) as f:
            assert "Baz qux" in f.read()

    def test_edit_missing_old_text(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectWriteFile
        tool = ProjectWriteFile()

        with open(os.path.join(project_dir, "doc.md"), "w") as f:
            f.write("Hello world")

        result = tool.call(json.dumps({
            "project": "test-proj", "path": "doc.md",
            "mode": "edit",
            "old_text": "NONEXISTENT",
            "new_text": "replacement",
        }))
        assert "not found" in result

    def test_edit_missing_file(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectWriteFile
        tool = ProjectWriteFile()

        result = tool.call(json.dumps({
            "project": "test-proj", "path": "nonexistent.md",
            "mode": "edit",
            "old_text": "x",
            "new_text": "y",
        }))
        assert "does not exist" in result

    def test_edit_requires_old_text(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectWriteFile
        tool = ProjectWriteFile()

        result = tool.call(json.dumps({
            "project": "test-proj", "path": "doc.md",
            "mode": "edit",
            "new_text": "y",
        }))
        assert "old_text" in result


class TestApplyPatch:
    def test_add_file(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectApplyPatch
        tool = ProjectApplyPatch()

        patch = dedent("""\
            *** Begin Patch
            *** Add File: new-file.md
            +# New Document
            +This is a new file.
            *** End Patch""")
        result = tool.call(json.dumps({"project": "test-proj", "patch": patch}))
        assert "A new-file.md" in result
        with open(os.path.join(project_dir, "new-file.md")) as f:
            content = f.read()
        assert "# New Document" in content
        assert "This is a new file." in content

    def test_delete_file(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectApplyPatch
        tool = ProjectApplyPatch()

        # Create file to delete
        with open(os.path.join(project_dir, "to-delete.md"), "w") as f:
            f.write("delete me")

        patch = dedent("""\
            *** Begin Patch
            *** Delete File: to-delete.md
            *** End Patch""")
        result = tool.call(json.dumps({"project": "test-proj", "patch": patch}))
        assert "D to-delete.md" in result
        assert not os.path.exists(os.path.join(project_dir, "to-delete.md"))

    def test_update_file(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectApplyPatch
        tool = ProjectApplyPatch()

        with open(os.path.join(project_dir, "doc.md"), "w") as f:
            f.write("# Title\n\nOld content here.\n\nKeep this.\n")

        patch = dedent("""\
            *** Begin Patch
            *** Update File: doc.md
            @@ Old content here.
            -Old content here.
            +New content here.
            +Added another line.
            *** End Patch""")
        result = tool.call(json.dumps({"project": "test-proj", "patch": patch}))
        assert "M doc.md" in result
        with open(os.path.join(project_dir, "doc.md")) as f:
            content = f.read()
        assert "New content here." in content
        assert "Added another line." in content
        assert "Old content" not in content
        assert "Keep this." in content

    def test_multi_file_patch(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectApplyPatch
        tool = ProjectApplyPatch()

        with open(os.path.join(project_dir, "existing.md"), "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        patch = dedent("""\
            *** Begin Patch
            *** Add File: research/new.md
            +# Research
            +Findings here.
            *** Update File: existing.md
            @@ Line 2
            -Line 2
            +Line 2 (updated)
            *** End Patch""")
        result = tool.call(json.dumps({"project": "test-proj", "patch": patch}))
        assert "A research/new.md" in result
        assert "M existing.md" in result

    def test_invalid_patch_format(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectApplyPatch
        tool = ProjectApplyPatch()

        result = tool.call(json.dumps({"project": "test-proj", "patch": "not a patch"}))
        assert "Error" in result

    def test_update_fuzzy_matching(self, project_dir):
        from sandbox_agent.tools.project_tools import ProjectApplyPatch
        tool = ProjectApplyPatch()

        # File has trailing spaces
        with open(os.path.join(project_dir, "doc.md"), "w") as f:
            f.write("# Title  \n\nContent here  \n")

        # Patch doesn't have trailing spaces
        patch = dedent("""\
            *** Begin Patch
            *** Update File: doc.md
            @@ Content here
            -Content here
            +Updated content
            *** End Patch""")
        result = tool.call(json.dumps({"project": "test-proj", "patch": patch}))
        assert "M doc.md" in result
