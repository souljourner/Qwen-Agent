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


def _format_content(msg) -> str:
    """Extract text content from a Message OR dict-shaped message.

    Defensive about input shape: qwen-agent yields `Message` objects, but
    some call paths (Gradio's history rebuild, Chainlit thread resume) pass
    plain dicts. Either is fine.
    """
    if msg is None:
        return ""
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            # ContentItem with `.text`
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
                continue
            # plain dict {"text": "..."} or {"type": "text", "text": "..."}
            if isinstance(item, dict):
                t = item.get("text")
                if t:
                    parts.append(t)
        return "\n".join(parts)
    return str(content)


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated in log)"


def log_turn(user_message, response_messages: List) -> None:
    """Append a user turn + agent response to the daily log.

    Captures every channel of the agent's reply — assistant text, reasoning,
    tool calls, tool results — so the markdown is a complete archive that can
    be exported or replayed in a different harness.
    """
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

        for msg in response_messages or []:
            role = getattr(msg, "role", None) if not isinstance(msg, dict) else msg.get("role")
            if role == "assistant":
                content = _format_content(msg)
                reasoning = (
                    getattr(msg, "reasoning_content", None) if not isinstance(msg, dict)
                    else msg.get("reasoning_content")
                ) or ""
                fc = (
                    getattr(msg, "function_call", None) if not isinstance(msg, dict)
                    else msg.get("function_call")
                )

                # Reasoning rendered as a collapsible block so the visible
                # transcript stays clean, but the thinking is preserved.
                if reasoning:
                    f.write(f"<details><summary>[{timestamp}] Agent thinking</summary>\n\n")
                    f.write(_truncate(reasoning, 2000) + "\n\n</details>\n\n")

                if content:
                    f.write(f"### [{timestamp}] Agent\n\n")
                    f.write(content + "\n\n")

                # Tool call (assistant message that requests a tool)
                if fc:
                    name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else "tool")
                    args = getattr(fc, "arguments", None) or (fc.get("arguments") if isinstance(fc, dict) else "")
                    f.write(f"**Tool call: {name}**\n\n```json\n{_truncate(str(args), 500)}\n```\n\n")

            elif role == "function":
                name = (
                    getattr(msg, "name", None) if not isinstance(msg, dict) else msg.get("name")
                ) or "tool"
                content = _format_content(msg)
                f.write(f"**Tool: {name} result**\n\n```\n{_truncate(content, 500)}\n```\n\n")


def log_feedback_note(sentiment: str, comment: str, rated_excerpt: str) -> None:
    """Append a user-feedback note (👍/👎 from the Chainlit UI) to the daily
    log so the agent surfaces it when it reads chat history."""
    _ensure_dir()
    path = _today_path()
    is_new = not os.path.exists(path)
    with open(path, "a") as f:
        if is_new:
            f.write(f"# Chat Log — {datetime.now().strftime('%Y-%m-%d')}\n\n")
        ts = datetime.now().strftime("%H:%M:%S")
        f.write(f"---\n\n### [{ts}] User Feedback: {sentiment}\n\n")
        if comment:
            f.write(f"> Comment: {comment}\n\n")
        if rated_excerpt:
            f.write(f"On message: _{_truncate(rated_excerpt, 300)}_\n\n")


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

        with open(filepath, "r") as f:
            content = f.read()
        # Cap at ~4000 tokens to avoid blowing up context
        max_chars = 16000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... (log truncated — use code_interpreter to read the full file)"
        return content
