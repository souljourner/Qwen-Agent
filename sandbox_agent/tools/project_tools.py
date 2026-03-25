"""Project workspace tools — persistent file management for long-running projects."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR
from sandbox_agent.tools.git_autocommit import autocommit

PROJECTS_DIR = os.path.join(DATA_DIR, "projects")


def _ensure_projects_dir():
    os.makedirs(PROJECTS_DIR, exist_ok=True)


def _project_dir(project_name: str) -> str:
    """Get the directory for a project. Validates name is safe."""
    safe_name = "".join(c for c in project_name if c.isalnum() or c in "-_ ").strip()
    safe_name = safe_name.replace(" ", "-").lower()
    if not safe_name:
        raise ValueError(f"Invalid project name: {project_name}")
    return os.path.join(PROJECTS_DIR, safe_name)


def _project_meta_path(project_dir: str) -> str:
    return os.path.join(project_dir, ".project.json")


def _load_meta(project_dir: str) -> dict:
    meta_path = _project_meta_path(project_dir)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _save_meta(project_dir: str, meta: dict):
    with open(_project_meta_path(project_dir), "w") as f:
        json.dump(meta, f, indent=2, default=str)


@register_tool("create_project")
class CreateProject(BaseTool):
    """Create a new project workspace."""

    name = "create_project"
    description = (
        "Create a new project workspace for long-running work. "
        "Each project gets its own folder for files, research, drafts, and reports. "
        "Use this when starting a multi-session effort (e.g., business plan, research project)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Project name (e.g., 'new-business-plan', 'market-research').",
            },
            "description": {
                "type": "string",
                "description": "Brief description of the project's goal.",
            },
        },
        "required": ["name", "description"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["name"])

        if os.path.exists(pdir):
            return f"Project already exists: {params['name']}"

        os.makedirs(pdir, exist_ok=True)
        meta = {
            "name": params["name"],
            "description": params["description"],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        _save_meta(pdir, meta)

        # Create a README for the project
        readme_path = os.path.join(pdir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"# {params['name']}\n\n{params['description']}\n\nCreated: {meta['created_at']}\n")

        return f"Project '{params['name']}' created at {pdir}"


@register_tool("list_projects")
class ListProjects(BaseTool):
    """List all projects."""

    name = "list_projects"
    description = "List all project workspaces with their descriptions and file counts."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        _ensure_projects_dir()
        projects = []
        for name in sorted(os.listdir(PROJECTS_DIR)):
            pdir = os.path.join(PROJECTS_DIR, name)
            if not os.path.isdir(pdir):
                continue
            meta = _load_meta(pdir)
            files = [f for f in os.listdir(pdir) if not f.startswith(".")]
            projects.append({
                "name": name,
                "description": meta.get("description", ""),
                "created": meta.get("created_at", ""),
                "files": len(files),
            })

        if not projects:
            return "No projects yet. Use create_project to start one."

        lines = ["Projects:\n"]
        for p in projects:
            lines.append(f"- **{p['name']}**: {p['description']} ({p['files']} files, created {p['created'][:10]})")
        return "\n".join(lines)


@register_tool("project_write_file")
class ProjectWriteFile(BaseTool):
    """Write or overwrite a file in a project workspace."""

    name = "project_write_file"
    description = (
        "Write a file to a project workspace. Use for saving research, drafts, plans, "
        "data, reports, or any project artifact. Supports subdirectories (e.g., 'research/competitors.md'). "
        "Auto-committed to git."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name.",
            },
            "path": {
                "type": "string",
                "description": "File path within the project (e.g., 'plan.md', 'research/market-analysis.md').",
            },
            "content": {
                "type": "string",
                "description": "File content to write.",
            },
        },
        "required": ["project", "path", "content"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["project"])

        if not os.path.exists(pdir):
            return f"Project '{params['project']}' not found. Create it first with create_project."

        # Validate path — no escaping project dir
        file_path = os.path.normpath(params["path"])
        if file_path.startswith("..") or file_path.startswith("/"):
            return "Invalid path: must be relative within the project."

        full_path = os.path.join(pdir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w") as f:
            f.write(params["content"])

        # Update project metadata
        meta = _load_meta(pdir)
        meta["updated_at"] = datetime.now().isoformat()
        _save_meta(pdir, meta)

        # Git commit
        try:
            rel_path = os.path.relpath(full_path, DATA_DIR)
            autocommit(rel_path, f"Update {params['path']} in {params['project']}")
        except Exception:
            pass

        return f"Written: {params['path']} ({len(params['content'])} chars)"


@register_tool("project_read_file")
class ProjectReadFile(BaseTool):
    """Read a file from a project workspace."""

    name = "project_read_file"
    description = (
        "Read a file from a project workspace. Use to recall research, drafts, or data from previous sessions. "
        "For large files, consider using code_interpreter to read and process them instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name.",
            },
            "path": {
                "type": "string",
                "description": "File path within the project.",
            },
        },
        "required": ["project", "path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["project"])

        file_path = os.path.normpath(params["path"])
        if file_path.startswith("..") or file_path.startswith("/"):
            return "Invalid path: must be relative within the project."

        full_path = os.path.join(pdir, file_path)
        if not os.path.exists(full_path):
            return f"File not found: {params['path']} in project '{params['project']}'"

        content = open(full_path).read()

        # Cap to avoid blowing up context — suggest code_interpreter for large files
        max_chars = 16000
        if len(content) > max_chars:
            content = content[:max_chars] + (
                f"\n\n... (truncated at {max_chars} chars — use code_interpreter to read the full file at {full_path})"
            )

        return content


@register_tool("project_list_files")
class ProjectListFiles(BaseTool):
    """List files in a project workspace."""

    name = "project_list_files"
    description = "List all files in a project workspace, including subdirectories."
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name.",
            },
        },
        "required": ["project"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["project"])

        if not os.path.exists(pdir):
            return f"Project '{params['project']}' not found."

        files = []
        for root, dirs, filenames in os.walk(pdir):
            # Skip hidden files/dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, pdir)
                size = os.path.getsize(full)
                files.append(f"- {rel} ({size:,} bytes)")

        if not files:
            return f"Project '{params['project']}' has no files yet."

        meta = _load_meta(pdir)
        header = f"Project: {params['project']}"
        if meta.get("description"):
            header += f" — {meta['description']}"
        return header + "\n\n" + "\n".join(files)


@register_tool("project_delete_file")
class ProjectDeleteFile(BaseTool):
    """Delete a file from a project workspace."""

    name = "project_delete_file"
    description = "Delete a file from a project workspace. Auto-committed to git."
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name.",
            },
            "path": {
                "type": "string",
                "description": "File path within the project to delete.",
            },
        },
        "required": ["project", "path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["project"])

        file_path = os.path.normpath(params["path"])
        if file_path.startswith("..") or file_path.startswith("/"):
            return "Invalid path: must be relative within the project."

        full_path = os.path.join(pdir, file_path)
        if not os.path.exists(full_path):
            return f"File not found: {params['path']}"

        os.remove(full_path)
        return f"Deleted: {params['path']} from {params['project']}"
