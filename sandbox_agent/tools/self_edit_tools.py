"""Tools that allow the agent to read and update its own configuration files (SOUL.md, HEARTBEAT.md, MEMORIES.md)."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR
from sandbox_agent.tools.git_autocommit import autocommit

# Editable files and their fallback locations (bundled defaults)
_EDITABLE_FILES = {
    "SOUL.md": Path(__file__).parent.parent / "SOUL.md",
    "HEARTBEAT.md": Path(__file__).parent.parent / "heartbeat" / "HEARTBEAT.md",
    "MEMORIES.md": Path(__file__).parent.parent / "MEMORIES.md",
}


def _resolve_path(filename: str) -> Path:
    """Resolve to DATA_DIR copy if it exists, otherwise bundled default."""
    data_copy = Path(DATA_DIR) / filename
    if data_copy.exists():
        return data_copy
    bundled = _EDITABLE_FILES.get(filename)
    if bundled and bundled.exists():
        return bundled
    return data_copy  # Will be created in DATA_DIR


def _writable_path(filename: str) -> Path:
    """Always return the DATA_DIR path (writable volume in Docker)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return Path(DATA_DIR) / filename


def _read_file(filename: str) -> str:
    path = _resolve_path(filename)
    if path.exists():
        return path.read_text()
    return ""


def _write_file(filename: str, content: str) -> str:
    path = _writable_path(filename)
    path.write_text(content)
    return str(path)


def _patch_section(content: str, section: str, new_section_content: str) -> str:
    """Replace a ## section's content, preserving everything else."""
    header = f"## {section}"
    if header not in content:
        # Section doesn't exist — append it
        return content.rstrip("\n") + f"\n\n{header}\n{new_section_content}\n"

    # Find the section start and end
    start = content.index(header)
    after_header = start + len(header)

    # Find next ## section (or end of file)
    next_section = content.find("\n## ", after_header)
    if next_section == -1:
        end = len(content)
    else:
        end = next_section

    # Replace section content (keep the header)
    return content[:after_header] + "\n" + new_section_content.strip() + "\n" + content[end:]


@register_tool("update_soul")
class UpdateSoul(BaseTool):
    """Read or patch the SOUL.md identity file."""

    name = "update_soul"
    description = (
        "Read or patch your SOUL.md identity file. Three modes:\n"
        "1) No args → read the full file\n"
        "2) section + content → replace just that section (e.g., section='Boundaries', content='new text')\n"
        "3) append_to_section + line → add a line to a section\n"
        "Changes take effect on new background sessions and after restart, NOT the current conversation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "description": "Section name to replace (e.g., 'Boundaries', 'Capabilities'). The ## prefix is added automatically.",
            },
            "content": {
                "type": "string",
                "description": "New content for the section (replaces the section body, keeps the header).",
            },
            "append_to_section": {
                "type": "string",
                "description": "Section name to append a line to.",
            },
            "line": {
                "type": "string",
                "description": "Line to append to the section (e.g., '- New capability: ...').",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        current = _read_file("SOUL.md") or "(SOUL.md is empty or does not exist)"

        section = params.get("section", "")
        content = params.get("content", "")
        append_section = params.get("append_to_section", "")
        line = params.get("line", "")

        if section and content:
            # Mode 2: replace a section
            new_content = _patch_section(current, section, content)
            _write_file("SOUL.md", new_content)
            autocommit("SOUL.md", f"Update SOUL.md section: {section}")
            return f"Section '{section}' updated. Changes take effect on next session."
        elif append_section and line:
            # Mode 3: append a line to a section
            header = f"## {append_section}"
            if header in current:
                # Find end of section content (next ## or EOF)
                start = current.index(header) + len(header)
                next_section = current.find("\n## ", start)
                if next_section == -1:
                    insert_pos = len(current.rstrip("\n"))
                else:
                    insert_pos = next_section
                new_content = current[:insert_pos].rstrip("\n") + "\n" + line + "\n" + current[insert_pos:]
            else:
                new_content = current.rstrip("\n") + f"\n\n## {append_section}\n{line}\n"
            _write_file("SOUL.md", new_content)
            autocommit("SOUL.md", f"Append to SOUL.md section: {append_section}")
            return f"Appended to '{append_section}': {line[:80]}"
        else:
            # Mode 1: just read
            return current


@register_tool("update_heartbeat")
class UpdateHeartbeat(BaseTool):
    """Read or update the HEARTBEAT.md checklist."""

    name = "update_heartbeat"
    description = (
        "Read or update your HEARTBEAT.md checklist. Three modes:\n"
        "1) No args → read the current checklist\n"
        "2) content → replace the entire file\n"
        "3) add_item → add a new checklist item (- [ ] ...)\n"
        "Use '- [ ]' for pending items and '- [x]' for completed items."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "New content for HEARTBEAT.md (replaces entire file). Leave empty to just read.",
            },
            "add_item": {
                "type": "string",
                "description": "Add a new checklist item (without the '- [ ] ' prefix — it's added automatically).",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        current = _read_file("HEARTBEAT.md") or "(HEARTBEAT.md is empty or does not exist)"

        content = params.get("content", "")
        add_item = params.get("add_item", "")

        if content:
            _write_file("HEARTBEAT.md", content)
            autocommit("HEARTBEAT.md", "Update HEARTBEAT.md (agent self-edit)")
            return f"HEARTBEAT.md replaced. Changes take effect on next heartbeat."
        elif add_item:
            new_content = current.rstrip("\n") + f"\n- [ ] {add_item}\n"
            _write_file("HEARTBEAT.md", new_content)
            autocommit("HEARTBEAT.md", f"Add heartbeat item: {add_item[:50]}")
            return f"Added checklist item: - [ ] {add_item}"
        else:
            return current


@register_tool("read_memories")
class ReadMemories(BaseTool):
    """Read the agent's memory file (for checking latest state mid-session)."""

    name = "read_memories"
    description = (
        "Read the FULL MEMORIES.md (the copy in your system prompt is capped to the "
        "newest entries). Also reads compaction archives: archive='list' shows the "
        "archive files, archive='YYYY-MM' reads one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "archive": {
                "type": "string",
                "description": "Omit for the current file. 'list' to list archives; 'YYYY-MM' to read that archive.",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        archive = (params.get("archive") or "").strip()
        if archive:
            archive_dir = Path(DATA_DIR) / "memories_archive"
            if archive == "list":
                files = sorted(archive_dir.glob("*.md")) if archive_dir.is_dir() else []
                if not files:
                    return "(no memory archives yet)"
                return "Memory archives:\n" + "\n".join(
                    f"- {p.name} ({p.stat().st_size} bytes)" for p in files)
            if not re.match(r"^\d{4}-\d{2}$", archive):
                return "Error: archive must be 'list' or 'YYYY-MM'."
            path = archive_dir / f"{archive}.md"
            if not path.exists():
                return f"No archive for {archive}."
            return path.read_text()
        content = _read_file("MEMORIES.md")
        if not content:
            return "(MEMORIES.md is empty or does not exist)"
        return content


@register_tool("add_memory")
class AddMemory(BaseTool):
    """Append a concise learning to the agent's memory file."""

    name = "add_memory"
    description = (
        "Append a concise learning to MEMORIES.md. Use this when you learn something important "
        "from a conversation — user preferences, useful facts, technical discoveries, or task outcomes. "
        "Keep each entry to 1-2 lines. Add under the appropriate section header. "
        "IMPORTANT: Do NOT include a date in the memory text — the tool prepends today's date "
        "automatically. Just write the learning itself (e.g. 'User prefers terse responses'), "
        "not '[2026-04-20] User prefers terse responses'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": ["User Preferences", "Facts & Knowledge", "Technical Notes", "Task Learnings"],
                "description": "Which section to add the memory under.",
            },
            "memory": {
                "type": "string",
                "description": "The concise learning to save (1-2 lines).",
            },
        },
        "required": ["section", "memory"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        section = params["section"]
        memory = params["memory"]

        memory = re.sub(r"^\s*\[\d{4}-\d{2}-\d{2}\]\s*", "", memory.lstrip("- ").lstrip())

        content = _read_file("MEMORIES.md")
        if not content:
            content = (
                "# Agent Memories\n\n"
                "## User Preferences\n\n"
                "## Facts & Knowledge\n\n"
                "## Technical Notes\n\n"
                "## Task Learnings\n"
            )

        # Find the section and append the memory after it
        section_header = f"## {section}"
        if section_header in content:
            parts = content.split(section_header, 1)
            after_header = parts[1]
            date_str = datetime.now().strftime("%Y-%m-%d")
            new_entry = f"\n- [{date_str}] {memory}"
            parts[1] = after_header.rstrip("\n") + new_entry + "\n"
            content = section_header.join(parts)
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
            content += f"\n## {section}\n\n- [{date_str}] {memory}\n"

        path = _write_file("MEMORIES.md", content)
        autocommit("MEMORIES.md", f"Add memory: {memory[:50]}")
        return f"Memory saved to {section}: {memory[:80]}"


_CANONICAL_MEMORY_SECTIONS = ("## User Preferences", "## Facts & Knowledge",
                              "## Technical Notes", "## Task Learnings")


@register_tool("compact_memories")
class CompactMemories(BaseTool):
    """Replace MEMORIES.md with a consolidated version; old content is archived."""

    name = "compact_memories"
    description = (
        "Replace MEMORIES.md with a consolidated (merged/pruned) version. The current "
        "file is archived VERBATIM to memories_archive/YYYY-MM.md first — nothing is "
        "lost, so prune aggressively. Read read_skill('memory-maintenance') for the "
        "protocol. The new content must be SHORTER than the current file and keep the "
        "canonical section headers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "new_content": {
                "type": "string",
                "description": "The full consolidated MEMORIES.md content.",
            },
        },
        "required": ["new_content"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        new_content = params["new_content"]
        old = _read_file("MEMORIES.md")

        if len(new_content) >= len(old):
            return (f"Error: new content ({len(new_content)} chars) is not shorter than the "
                    f"current file ({len(old)} chars) — compaction must reduce size.")
        missing = [h for h in _CANONICAL_MEMORY_SECTIONS if h not in new_content]
        if missing:
            return f"Error: new content is missing canonical section(s): {', '.join(missing)}."

        # Archive the old content VERBATIM (deterministic — not agent-trusted).
        archive_dir = Path(DATA_DIR) / "memories_archive"
        os.makedirs(archive_dir, exist_ok=True)
        now = datetime.now()
        archive_path = archive_dir / f"{now.strftime('%Y-%m')}.md"
        with open(archive_path, "a") as f:
            f.write(f"\n\n# Snapshot {now.strftime('%Y-%m-%d %H:%M')} (pre-compaction, verbatim)\n\n")
            f.write(old)

        # Ensure the new content references the archives so future compactions
        # (and the agent) know where pruned memories live.
        if "## Archives" not in new_content:
            lines = new_content.split("\n")
            insert_at = 1 if lines and lines[0].startswith("#") else 0
            archives_ref = (
                "\n## Archives\n"
                "- Compacted memories preserved verbatim in memories_archive/ "
                "(monthly files, newest snapshot last). Load with read_memories(archive='YYYY-MM').\n"
            )
            lines.insert(insert_at, archives_ref)
            new_content = "\n".join(lines)

        _write_file("MEMORIES.md", new_content)
        import json as _json
        with open(archive_dir / ".last_compaction", "w") as f:
            _json.dump({"ts": now.timestamp(), "old_len": len(old), "new_len": len(new_content)}, f)
        autocommit("MEMORIES.md", f"Compact memories ({len(old)} -> {len(new_content)} chars)")
        return (f"Memories compacted: {len(old)} -> {len(new_content)} chars. "
                f"Previous content archived to memories_archive/{archive_path.name}.")
