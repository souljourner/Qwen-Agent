"""Project workspace tools — persistent file management for long-running projects."""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR
from sandbox_agent.tools.git_autocommit import autocommit


def _validate_data_path(path: str) -> str:
    """Validate a path is within DATA_DIR. Returns the resolved absolute path or raises."""
    resolved = os.path.realpath(os.path.join(DATA_DIR, path))
    data_real = os.path.realpath(DATA_DIR)
    if not resolved.startswith(data_real + os.sep) and resolved != data_real:
        raise ValueError(f"Path escapes sandbox: {path}")
    return resolved

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


# Files the pipeline orchestrator owns exclusively. Agents used to invent a
# shadow "pipeline/pipeline_state.json" and hand-edit status files, producing
# contradictory pipeline state ("phantom promote"). Writes are denied at the
# tool layer; pipeline/state.json is the single source of truth.
_ORCHESTRATOR_OWNED = ("pipeline/state.json", "pipeline/pipeline_state.json", "status.md",
                       "pipeline/metrics.json")


def _is_orchestrator_owned(rel_path: str) -> bool:
    norm = os.path.normpath(rel_path).replace(os.sep, "/")
    return norm in _ORCHESTRATOR_OWNED


_ORCHESTRATOR_OWNED_MSG = (
    "Error: '{path}' is pipeline-orchestrator-owned state — agents must not write it. "
    "Pipeline status is tracked automatically in pipeline/state.json; write your outputs "
    "to the stage's declared artifact files only."
)


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


@register_tool("delete_project")
class DeleteProject(BaseTool):
    """Delete an entire project and all its files."""

    name = "delete_project"
    description = (
        "Delete a project and all its files permanently. This cannot be undone. "
        "Use list_projects and project_list_files first to verify what will be deleted."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name to delete.",
            },
        },
        "required": ["project"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["project"])

        if not os.path.exists(pdir):
            return f"Project '{params['project']}' not found."

        # Count files before deletion
        file_count = sum(len(files) for _, _, files in os.walk(pdir))
        shutil.rmtree(pdir)
        return f"Deleted project '{params['project']}' ({file_count} files removed)"


@register_tool("rename_project")
class RenameProject(BaseTool):
    """Rename an existing project (folder + metadata); files are preserved."""

    name = "rename_project"
    description = (
        "Rename an existing project. Renames the project's folder and updates its metadata; "
        "all files inside are preserved. The new name is slugified the same way as "
        "create_project (lowercased, spaces become hyphens). Fails if a project with the "
        "new name already exists."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Current project name.",
            },
            "new_name": {
                "type": "string",
                "description": "New project name (e.g., 'q3-launch-plan').",
            },
        },
        "required": ["project", "new_name"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            src = _project_dir(params["project"])
            dst = _project_dir(params["new_name"])
        except ValueError as e:
            return f"Error: {e}"

        if not os.path.exists(src):
            return f"Project '{params['project']}' not found."
        if os.path.abspath(src) == os.path.abspath(dst):
            return f"Project is already named '{os.path.basename(dst)}'."
        if os.path.exists(dst):
            return f"A project named '{os.path.basename(dst)}' already exists."

        shutil.move(src, dst)

        # Keep metadata in sync with the new name.
        meta = _load_meta(dst)
        meta["name"] = params["new_name"]
        meta["updated_at"] = datetime.now().isoformat()
        _save_meta(dst, meta)

        old_slug, new_slug = os.path.basename(src), os.path.basename(dst)
        autocommit("projects/", f"Rename project '{old_slug}' → '{new_slug}'")
        return f"Renamed project '{old_slug}' → '{new_slug}'"


@register_tool("update_project")
class UpdateProject(BaseTool):
    """Update a project's description in its metadata (does not rename or touch files)."""

    name = "update_project"
    description = (
        "Update a project's description (the summary shown by list_projects). Use this to "
        "revise a project's goal after creation. Does not rename the project (use "
        "rename_project for that) and does not modify README.md or any other files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name.",
            },
            "description": {
                "type": "string",
                "description": "New project description.",
            },
        },
        "required": ["project", "description"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            pdir = _project_dir(params["project"])
        except ValueError as e:
            return f"Error: {e}"

        if not os.path.exists(pdir):
            return f"Project '{params['project']}' not found."

        meta = _load_meta(pdir)
        meta["description"] = params["description"]
        meta["updated_at"] = datetime.now().isoformat()
        _save_meta(pdir, meta)

        slug = os.path.basename(pdir)
        autocommit(f"projects/{slug}/.project.json", f"Update description for project '{slug}'")
        return f"Updated description for project '{slug}'."


@register_tool("project_write_file")
class ProjectWriteFile(BaseTool):
    """Write, append, or overwrite a file in a project workspace."""

    name = "project_write_file"
    description = (
        "Write a file to a project workspace. Supports 3 modes:\n"
        "- 'write' (default): create or overwrite the file\n"
        "- 'append': add content to the end of an existing file (creates if missing)\n"
        "- 'edit': find and replace a specific string in the file\n"
        "For large documents, use 'append' to build section by section. "
        "For targeted changes, use 'edit' with old_text/new_text. "
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
                "description": "File content to write (for 'write' and 'append' modes).",
            },
            "mode": {
                "type": "string",
                "enum": ["write", "append", "edit"],
                "description": "Write mode: 'write' (overwrite), 'append' (add to end), 'edit' (find and replace). Default: 'write'.",
            },
            "old_text": {
                "type": "string",
                "description": "Text to find (required for 'edit' mode). Must be an exact match of existing content.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text (required for 'edit' mode).",
            },
        },
        "required": ["project", "path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except Exception as e:  # noqa: BLE001 — surface schema errors as a clean tool result
            return f"Error: invalid arguments — {e}"
        pdir = _project_dir(params["project"])

        if not os.path.exists(pdir):
            return f"Project '{params['project']}' not found. Create it first with create_project."

        # Validate path — no escaping project dir
        file_path = os.path.normpath(params["path"])
        if file_path.startswith("..") or file_path.startswith("/"):
            return "Invalid path: must be relative within the project."

        if _is_orchestrator_owned(file_path):
            return _ORCHESTRATOR_OWNED_MSG.format(path=file_path)

        full_path = os.path.join(pdir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Mode validation: an EXPLICIT empty/blank mode is an error (don't silently
        # default — the caller showed intent to specify one). Omitting the key
        # entirely is fine and falls back to "write".
        if "mode" in params:
            raw_mode = params["mode"]
            if not isinstance(raw_mode, str) or not raw_mode.strip():
                return "Error: 'mode' must not be empty. Use 'write', 'append', or 'edit'."
            mode = raw_mode.strip()
        else:
            mode = "write"
        if mode not in ("write", "append", "edit"):
            return f"Error: invalid mode '{mode}'. Use 'write', 'append', or 'edit'."

        # Content validation: write/append must have non-empty content. (edit
        # uses old_text/new_text, not content, so this check doesn't apply.)
        if mode in ("write", "append"):
            content = params.get("content", "")
            if not content:
                return f"Error: 'content' must not be empty for {mode} mode."

        if mode == "append":
            with open(full_path, "a") as f:
                f.write(content)
            result_msg = f"Appended to {params['path']} ({len(content)} chars)"

        elif mode == "edit":
            old_text = params.get("old_text")
            new_text = params.get("new_text")
            if not old_text:
                return "Error: 'old_text' is required for edit mode."
            if new_text is None:
                return "Error: 'new_text' is required for edit mode."
            if not os.path.exists(full_path):
                return f"Error: file '{params['path']}' does not exist. Use 'write' mode to create it."
            with open(full_path, "r") as f:
                existing = f.read()
            if old_text not in existing:
                # Try trimmed matching as fallback
                old_stripped = old_text.strip()
                found = False
                for line_start in range(len(existing)):
                    candidate = existing[line_start:line_start + len(old_text)]
                    if candidate.strip() == old_stripped:
                        existing = existing[:line_start] + new_text + existing[line_start + len(candidate):]
                        found = True
                        break
                if not found:
                    preview = old_text[:100] + ("..." if len(old_text) > 100 else "")
                    return f"Error: old_text not found in {params['path']}. Preview: '{preview}'"
                result_msg = f"Edited {params['path']} (fuzzy match)"
            else:
                count = existing.count(old_text)
                existing = existing.replace(old_text, new_text, 1)
                if count > 1:
                    result_msg = f"Edited {params['path']} (replaced first of {count} occurrences)"
                else:
                    result_msg = f"Edited {params['path']} (replaced {len(old_text)} chars with {len(new_text)} chars)"
            with open(full_path, "w") as f:
                f.write(existing)

        else:  # write mode (default)
            with open(full_path, "w") as f:
                f.write(content)
            result_msg = f"Written: {params['path']} ({len(content)} chars)"

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

        return result_msg


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

        with open(full_path) as f:
            content = f.read()

        # Cap to avoid blowing up context — suggest code_interpreter for large files
        max_chars = 16000
        if len(content) > max_chars:
            content = content[:max_chars] + (
                f"\n\n... (truncated at {max_chars} chars — use code_interpreter to read the full file at {full_path})"
            )

        return content


@register_tool("project_list_files")
class ProjectListFiles(BaseTool):
    """List files and directories in a project workspace (one level)."""

    name = "project_list_files"
    description = (
        "List files and subdirectories in a project directory (one level, like ls). "
        "Use path to browse into subdirectories."
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
                "description": "Directory path within the project to list (e.g., 'research', 'research/hallucheck'). Omit for project root.",
            },
        },
        "required": ["project"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["project"])

        if not os.path.exists(pdir):
            return f"Project '{params['project']}' not found."

        # Resolve target directory
        subpath = params.get("path", "")
        if subpath:
            subpath = os.path.normpath(subpath)
            if subpath.startswith("..") or subpath.startswith("/"):
                return "Invalid path: must be relative within the project."
            target = os.path.join(pdir, subpath)
        else:
            target = pdir

        if not os.path.isdir(target):
            return f"Directory not found: {subpath or '(root)'}"

        entries = []
        for name in sorted(os.listdir(target)):
            if name.startswith("."):
                continue
            full = os.path.join(target, name)
            if os.path.isdir(full):
                # Count items in subdirectory
                count = len([f for f in os.listdir(full) if not f.startswith(".")])
                entries.append(f"  {name}/ ({count} items)")
            else:
                size = os.path.getsize(full)
                entries.append(f"  {name} ({size:,} bytes)")

        if not entries:
            return f"Empty directory: {subpath or '(root)'}"

        meta = _load_meta(pdir)
        header = f"Project: {params['project']}"
        if meta.get("description"):
            header += f" — {meta['description']}"
        if subpath:
            header += f"\nPath: {subpath}/"
        return header + "\n\n" + "\n".join(entries)


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


@register_tool("project_apply_patch")
class ProjectApplyPatch(BaseTool):
    """Apply a structured patch to one or more files in a project."""

    name = "project_apply_patch"
    description = (
        "Apply a patch to one or more project files in a single call. "
        "Supports adding, deleting, and updating files. "
        "Update uses context-based matching (like unified diff) — provide a few lines of "
        "surrounding context with - (remove) and + (add) markers.\n\n"
        "Format:\n"
        "*** Begin Patch\n"
        "*** Add File: path/to/new.md\n"
        "+line 1 of new file\n"
        "+line 2 of new file\n"
        "*** Update File: path/to/existing.md\n"
        "@@ context line to locate the edit\n"
        "-old line to remove\n"
        "+new line to add\n"
        " unchanged context line\n"
        "*** Delete File: path/to/remove.md\n"
        "*** End Patch\n\n"
        "Best for making targeted edits to multiple files without rewriting them entirely."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name.",
            },
            "patch": {
                "type": "string",
                "description": "Patch content using *** Begin Patch / *** End Patch format.",
            },
        },
        "required": ["project", "patch"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        pdir = _project_dir(params["project"])

        if not os.path.exists(pdir):
            return f"Project '{params['project']}' not found."

        patch_text = params["patch"]
        # Deny patches touching orchestrator-owned state files (same guard as
        # project_write_file — see _ORCHESTRATOR_OWNED).
        for line in patch_text.splitlines():
            m = re.match(r"\*\*\* (?:Add|Update|Delete) File: (.+)$", line.strip())
            if m and _is_orchestrator_owned(m.group(1).strip()):
                return _ORCHESTRATOR_OWNED_MSG.format(path=m.group(1).strip())
        try:
            result = _apply_patch(pdir, patch_text)
        except Exception as e:
            return f"Patch failed: {e}"

        # Update project metadata
        meta = _load_meta(pdir)
        meta["updated_at"] = datetime.now().isoformat()
        _save_meta(pdir, meta)

        # Git commit
        try:
            autocommit("projects/", f"Apply patch in {params['project']}")
        except Exception:
            pass

        return result


def _apply_patch(project_dir: str, patch_text: str) -> str:
    """Parse and apply a patch in OpenClaw-style format.

    Format:
        *** Begin Patch
        *** Add File: path
        +content lines (each prefixed with +)
        *** Update File: path
        @@ optional context line
        -old line
        +new line
         unchanged context line
        *** Delete File: path
        *** End Patch
    """
    lines = patch_text.strip().split("\n")
    if not lines:
        return "Error: empty patch"

    # Strip heredoc wrappers if present
    if lines[0] in ("<<EOF", "<<'EOF'", '<<"EOF"') and lines[-1].rstrip() == "EOF":
        lines = lines[1:-1]

    if lines[0].strip() != "*** Begin Patch":
        return "Error: patch must start with '*** Begin Patch'"
    if lines[-1].strip() != "*** End Patch":
        return "Error: patch must end with '*** End Patch'"

    body = lines[1:-1]
    added, modified, deleted = [], [], []
    i = 0

    while i < len(body):
        line = body[i].strip()

        if not line:
            i += 1
            continue

        if line.startswith("*** Add File: "):
            file_path = line[len("*** Add File: "):]
            content_lines = []
            i += 1
            while i < len(body) and body[i].startswith("+"):
                content_lines.append(body[i][1:])
                i += 1
            full_path = _safe_project_path(project_dir, file_path)
            if full_path is None:
                return f"Error: invalid path '{file_path}'"
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write("\n".join(content_lines) + "\n" if content_lines else "")
            added.append(file_path)

        elif line.startswith("*** Delete File: "):
            file_path = line[len("*** Delete File: "):]
            full_path = _safe_project_path(project_dir, file_path)
            if full_path is None:
                return f"Error: invalid path '{file_path}'"
            if os.path.exists(full_path):
                os.remove(full_path)
                deleted.append(file_path)
            else:
                return f"Error: file not found for deletion: {file_path}"
            i += 1

        elif line.startswith("*** Update File: "):
            file_path = line[len("*** Update File: "):]
            full_path = _safe_project_path(project_dir, file_path)
            if full_path is None:
                return f"Error: invalid path '{file_path}'"
            if not os.path.exists(full_path):
                return f"Error: file not found for update: {file_path}"

            # Parse chunks
            i += 1
            chunks = []
            while i < len(body) and not body[i].startswith("***"):
                if not body[i].strip():
                    i += 1
                    continue
                chunk, consumed = _parse_update_chunk(body, i)
                chunks.append(chunk)
                i += consumed

            if not chunks:
                return f"Error: update for '{file_path}' has no edit chunks"

            # Apply chunks
            try:
                result_text = _apply_update_chunks(full_path, chunks)
                with open(full_path, "w") as f:
                    f.write(result_text)
                modified.append(file_path)
            except Exception as e:
                return f"Error updating {file_path}: {e}"
        else:
            return f"Error: unrecognized patch line: '{line}'"

    # Format summary
    parts = []
    if added:
        parts.extend(f"A {f}" for f in added)
    if modified:
        parts.extend(f"M {f}" for f in modified)
    if deleted:
        parts.extend(f"D {f}" for f in deleted)

    if not parts:
        return "No files were modified."
    return "Patch applied:\n" + "\n".join(parts)


def _safe_project_path(project_dir: str, file_path: str):
    """Validate and resolve a path within the project. Returns None if invalid."""
    normalized = os.path.normpath(file_path)
    if normalized.startswith("..") or normalized.startswith("/"):
        return None
    return os.path.join(project_dir, normalized)


def _parse_update_chunk(lines, start):
    """Parse one update chunk starting at `start`. Returns (chunk_dict, lines_consumed)."""
    i = start
    context = None

    # Optional @@ context marker
    if lines[i].startswith("@@ ") or lines[i].strip() == "@@":
        if lines[i].strip() != "@@":
            context = lines[i][3:]  # text after "@@ "
        i += 1

    old_lines = []
    new_lines = []
    consumed_content = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("***") or (line.startswith("@@ ") and consumed_content > 0):
            break

        marker = line[0] if line else " "
        if marker == "-":
            old_lines.append(line[1:])
        elif marker == "+":
            new_lines.append(line[1:])
        elif marker == " ":
            old_lines.append(line[1:])
            new_lines.append(line[1:])
        else:
            # Treat as context (no prefix = unchanged line)
            old_lines.append(line)
            new_lines.append(line)

        i += 1
        consumed_content += 1

    return {"context": context, "old_lines": old_lines, "new_lines": new_lines}, i - start


def _apply_update_chunks(file_path: str, chunks: list) -> str:
    """Apply update chunks to a file. Returns the new file content."""
    with open(file_path, "r") as f:
        original = f.read()

    original_lines = original.split("\n")
    # Remove trailing empty line from split
    if original_lines and original_lines[-1] == "":
        original_lines.pop()

    replacements = []
    search_start = 0

    for chunk in chunks:
        context = chunk.get("context")
        old_lines = chunk["old_lines"]
        new_lines = chunk["new_lines"]

        # If there's a context marker, seek to it first
        if context:
            ctx_idx = _seek_line(original_lines, context, search_start)
            if ctx_idx is None:
                raise ValueError(f"Context not found: '{context}'")
            search_start = ctx_idx  # Start from context line (it may be the edited line itself)

        if not old_lines:
            # Pure insertion at end of file
            insert_at = len(original_lines)
            replacements.append((insert_at, 0, new_lines))
            continue

        # Find the old_lines sequence in the original
        found = _seek_sequence(original_lines, old_lines, search_start)
        if found is None:
            preview = old_lines[0] if old_lines else "(empty)"
            raise ValueError(f"Could not find lines to replace starting with: '{preview}'")

        replacements.append((found, len(old_lines), new_lines))
        search_start = found + len(old_lines)

    # Apply replacements in reverse order to preserve indices
    replacements.sort(key=lambda r: r[0])
    result = list(original_lines)
    for start_idx, old_len, new_lines in reversed(replacements):
        result[start_idx:start_idx + old_len] = new_lines

    # Ensure trailing newline
    if not result or result[-1] != "":
        result.append("")
    return "\n".join(result)


def _seek_line(lines: list, target: str, start: int) -> int:
    """Find a single line in the file, with fuzzy matching fallback."""
    # Exact match
    for i in range(start, len(lines)):
        if lines[i] == target:
            return i
    # Trimmed match
    target_stripped = target.strip()
    for i in range(start, len(lines)):
        if lines[i].strip() == target_stripped:
            return i
    return None


def _seek_sequence(lines: list, pattern: list, start: int) -> int:
    """Find a sequence of lines, with progressive fuzzy matching."""
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None

    max_start = len(lines) - len(pattern)

    # Pass 1: exact match
    for i in range(start, max_start + 1):
        if all(lines[i + j] == pattern[j] for j in range(len(pattern))):
            return i

    # Pass 2: strip trailing whitespace
    for i in range(start, max_start + 1):
        if all(lines[i + j].rstrip() == pattern[j].rstrip() for j in range(len(pattern))):
            return i

    # Pass 3: strip all whitespace
    for i in range(start, max_start + 1):
        if all(lines[i + j].strip() == pattern[j].strip() for j in range(len(pattern))):
            return i

    return None


@register_tool("move_file")
class MoveFile(BaseTool):
    """Move or rename a file within the data directory."""

    name = "move_file"
    description = (
        "Move or rename a file within the data directory. Both paths are relative to DATA_DIR. "
        "Can move files between projects or reorganize within a project. "
        "Creates destination directories if needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "src": {
                "type": "string",
                "description": "Source path relative to DATA_DIR (e.g., 'projects/flatsixai/old-name.md').",
            },
            "dst": {
                "type": "string",
                "description": "Destination path relative to DATA_DIR (e.g., 'projects/flatsixai/research/new-name.md').",
            },
        },
        "required": ["src", "dst"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            src = _validate_data_path(params["src"])
            dst = _validate_data_path(params["dst"])
        except ValueError as e:
            return f"Error: {e}"

        if not os.path.exists(src):
            return f"Source not found: {params['src']}"

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return f"Moved: {params['src']} → {params['dst']}"


@register_tool("delete_file")
class DeleteFile(BaseTool):
    """Delete a file or empty directory within the data directory."""

    name = "delete_file"
    description = (
        "Delete a file or empty directory within the data directory. "
        "Path is relative to DATA_DIR. Cannot delete non-empty directories."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to DATA_DIR (e.g., 'projects/flatsixai/old-file.md').",
            },
        },
        "required": ["path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            full_path = _validate_data_path(params["path"])
        except ValueError as e:
            return f"Error: {e}"

        if not os.path.exists(full_path):
            return f"Not found: {params['path']}"

        if os.path.isdir(full_path):
            try:
                os.rmdir(full_path)  # Only removes empty directories
                return f"Deleted directory: {params['path']}"
            except OSError:
                return f"Cannot delete non-empty directory: {params['path']}"
        else:
            os.remove(full_path)
            return f"Deleted: {params['path']}"
