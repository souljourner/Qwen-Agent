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
        "Read the latest MEMORIES.md. Your memories are already in the system prompt, "
        "so only use this to check if a mid-session add_memory was saved correctly."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
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
