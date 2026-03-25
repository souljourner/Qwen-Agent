"""Tools that allow the agent to read and update its own configuration files (SOUL.md, HEARTBEAT.md)."""

import os
from pathlib import Path
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR
from sandbox_agent.tools.git_autocommit import autocommit

# Editable files and their fallback locations (bundled defaults)
_EDITABLE_FILES = {
    "SOUL.md": Path(__file__).parent.parent / "SOUL.md",
    "HEARTBEAT.md": Path(__file__).parent.parent / "heartbeat" / "HEARTBEAT.md",
    "MEMORIES.md": Path(__file__).parent.parent / "heartbeat" / "MEMORIES.md",
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


@register_tool("read_soul")
class ReadSoul(BaseTool):
    """Read the current SOUL.md (agent identity and instructions)."""

    name = "read_soul"
    description = "Read your current SOUL.md identity file. Use this to review your own instructions and personality."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        content = _read_file("SOUL.md")
        if not content:
            return "(SOUL.md is empty or does not exist)"
        return content


@register_tool("update_soul")
class UpdateSoul(BaseTool):
    """Update the SOUL.md identity file."""

    name = "update_soul"
    description = (
        "Update your SOUL.md identity file. This changes your personality, capabilities description, "
        "and behavioral boundaries. The new content takes effect on the next conversation or heartbeat. "
        "Always read_soul first to see the current content before making changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The complete new content for SOUL.md (replaces the entire file).",
            },
        },
        "required": ["content"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        path = _write_file("SOUL.md", params["content"])
        autocommit("SOUL.md", "Update SOUL.md (agent self-edit)")
        return f"SOUL.md updated at {path}. Changes take effect on next session."


@register_tool("read_heartbeat")
class ReadHeartbeat(BaseTool):
    """Read the current HEARTBEAT.md checklist."""

    name = "read_heartbeat"
    description = "Read your current HEARTBEAT.md checklist. Use this to see what background checks are configured."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        content = _read_file("HEARTBEAT.md")
        if not content:
            return "(HEARTBEAT.md is empty or does not exist)"
        return content


@register_tool("update_heartbeat")
class UpdateHeartbeat(BaseTool):
    """Update the HEARTBEAT.md checklist."""

    name = "update_heartbeat"
    description = (
        "Update your HEARTBEAT.md checklist. This controls what background checks run every 30 minutes. "
        "Use '- [ ]' for pending items and '- [x]' for completed items. "
        "Always read_heartbeat first to see the current checklist before making changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The complete new content for HEARTBEAT.md (replaces the entire file).",
            },
        },
        "required": ["content"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        path = _write_file("HEARTBEAT.md", params["content"])
        autocommit("HEARTBEAT.md", "Update HEARTBEAT.md (agent self-edit)")
        return f"HEARTBEAT.md updated at {path}. Changes take effect on next heartbeat cycle."


@register_tool("read_memories")
class ReadMemories(BaseTool):
    """Read the agent's memory file."""

    name = "read_memories"
    description = "Read your MEMORIES.md file to recall important learnings from past conversations."
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
        "Keep each entry to 1-2 lines. Add under the appropriate section header."
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
            # Insert the memory line after the section header
            parts = content.split(section_header, 1)
            # Find the end of the section header line
            after_header = parts[1]
            # Append the new memory as a bullet point
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            new_entry = f"\n- [{date_str}] {memory}"
            parts[1] = after_header.rstrip("\n") + new_entry + "\n"
            content = section_header.join(parts)
        else:
            # Section not found — append at end
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")
            content += f"\n## {section}\n\n- [{date_str}] {memory}\n"

        path = _write_file("MEMORIES.md", content)
        autocommit("MEMORIES.md", f"Add memory: {memory[:50]}")
        return f"Memory saved to {section}: {memory[:80]}"
