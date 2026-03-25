"""Daily markdown chat logger — appends all conversations to a daily log file.

Logs are stored in DATA_DIR/chat_logs/ as YYYY-MM-DD.md files.
The agent can read these via the list_chat_logs and read_chat_log tools.
"""

import logging
import os
from datetime import datetime
from typing import List, Union

from qwen_agent.llm.schema import Message
from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

CHAT_LOGS_DIR = os.path.join(DATA_DIR, "chat_logs")


def _ensure_dir():
    os.makedirs(CHAT_LOGS_DIR, exist_ok=True)


def _today_path() -> str:
    return os.path.join(CHAT_LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.md")


def _format_content(msg: Message) -> str:
    """Extract text content from a message."""
    if isinstance(msg.content, str):
        return msg.content
    elif isinstance(msg.content, list):
        parts = []
        for item in msg.content:
            if hasattr(item, "text") and item.text:
                parts.append(item.text)
        return "\n".join(parts)
    return ""


def log_turn(user_message: Message, response_messages: List[Message]) -> None:
    """Append a user turn + agent response to the daily log."""
    _ensure_dir()
    path = _today_path()

    # Check if file is new (needs header)
    is_new = not os.path.exists(path)

    with open(path, "a") as f:
        if is_new:
            f.write(f"# Chat Log — {datetime.now().strftime('%Y-%m-%d')}\n\n")

        timestamp = datetime.now().strftime("%H:%M:%S")
        f.write(f"---\n\n### [{timestamp}] User\n\n")
        f.write(_format_content(user_message) + "\n\n")

        for msg in response_messages:
            if msg.role == "assistant":
                content = _format_content(msg)
                if content:
                    f.write(f"### [{timestamp}] Agent\n\n")
                    f.write(content + "\n\n")
            elif msg.role == "function":
                name = msg.name or "tool"
                content = _format_content(msg)
                # Truncate long tool results in the log
                if len(content) > 500:
                    content = content[:500] + "\n... (truncated in log)"
                f.write(f"**Tool: {name}**\n\n```\n{content}\n```\n\n")


def log_background_task(task_name: str, task_id: str, result: str) -> None:
    """Log a background task execution to the daily log."""
    _ensure_dir()
    path = _today_path()

    is_new = not os.path.exists(path)

    with open(path, "a") as f:
        if is_new:
            f.write(f"# Chat Log — {datetime.now().strftime('%Y-%m-%d')}\n\n")

        timestamp = datetime.now().strftime("%H:%M:%S")
        f.write(f"---\n\n### [{timestamp}] Background Task: {task_name} [{task_id}]\n\n")
        if len(result) > 1000:
            result = result[:1000] + "\n... (truncated in log)"
        f.write(result + "\n\n")


@register_tool("list_chat_logs")
class ListChatLogs(BaseTool):
    """List available daily chat log files."""

    name = "list_chat_logs"
    description = "List available chat log files by date. Use this to find which days have conversation history."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        _ensure_dir()
        files = sorted(
            [f for f in os.listdir(CHAT_LOGS_DIR) if f.endswith(".md")],
            reverse=True,
        )
        if not files:
            return "No chat logs found."
        return "Available chat logs:\n" + "\n".join(f"- {f}" for f in files)


@register_tool("read_chat_log")
class ReadChatLog(BaseTool):
    """Read a daily chat log to recall earlier conversations."""

    name = "read_chat_log"
    description = (
        "Read a chat log file to recall earlier conversations. "
        "Use list_chat_logs first to see available dates. "
        "Pass the date as YYYY-MM-DD, or 'today' for the current day's log."
    )
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date of the log to read: 'today' or YYYY-MM-DD (e.g., '2024-03-15').",
            },
        },
        "required": ["date"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        date_str = params["date"]

        if date_str.lower() == "today":
            date_str = datetime.now().strftime("%Y-%m-%d")

        filename = f"{date_str}.md"
        filepath = os.path.join(CHAT_LOGS_DIR, filename)

        if not os.path.exists(filepath):
            return f"No chat log found for {date_str}."

        content = open(filepath, "r").read()
        # Cap at ~4000 tokens to avoid blowing up context
        max_chars = 16000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... (log truncated — use code_interpreter to read the full file)"
        return content
