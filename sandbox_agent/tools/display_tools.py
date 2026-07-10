"""display_doc — show a file to the user in the Chainlit UI without spending context.

The agent often produces documents (reports, drafts, generated charts, PDFs) the
user wants to SEE in full. Returning the file as a normal tool result would dump
the whole thing into the LLM context (a function-role message). `display_doc`
instead pushes the content to the chat surface out-of-band — a thread-keyed hook
that chat_app registers per turn and drains into a cl.Message / cl.Image / cl.Pdf
— and returns only a tiny confirmation string. So the content never enters the
context window, live or on reload (chat_app marks the message author="document"
and on_chat_resume skips it).

Addressed by {project, path}, mirroring project_read_file.
"""

import os
import threading
from typing import Callable, Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.tools.project_tools import _project_dir

# Thread-ident → callable(payload: dict). chat_app registers one for the worker
# thread running a chat turn (see register_display_hook); the hook ships the doc
# to that turn's UI queue. No hook registered (cron / heartbeat / REPL) → there's
# no chat surface, so display_doc just reports that.
_display_hooks: "dict[int, Callable[[dict], None]]" = {}

# Text shown inline as message content persists as steps.output (reloadable, no
# storage client needed). Larger than this → present as a downloadable file
# element instead of inlining a huge blob into the chat row + socket.
_INLINE_TEXT_MAX = 200_000

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_TEXT_EXTS = {".md", ".markdown", ".txt", ".py", ".js", ".ts", ".tsx", ".json",
              ".yaml", ".yml", ".toml", ".csv", ".tsv", ".html", ".css", ".sh",
              ".sql", ".log", ".rst", ".ini", ".cfg", ".xml", ".tex", ".c", ".cpp",
              ".go", ".rs", ".java", ".rb"}


def register_display_hook(fn: Callable[[dict], None]) -> None:
    _display_hooks[threading.get_ident()] = fn


def unregister_display_hook() -> None:
    _display_hooks.pop(threading.get_ident(), None)


def _classify(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext in _TEXT_EXTS:
        return "text"
    return "file"


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


@register_tool("display_doc")
class DisplayDoc(BaseTool):
    """Show a file to the user in the chat, full content, without reading it into context."""

    name = "display_doc"
    description = (
        "Show a file to the user in the chat — full content, rendered (markdown text, image, or PDF). "
        "Use this to PRESENT a finished report, generated document, chart/figure image, or PDF that the "
        "user should see in full. "
        "IMPORTANT: this does NOT load the file into YOUR context — the content is shown to the user "
        "directly, not returned to you. If you need to read, quote, or reason about the content yourself, "
        "use project_read_file instead (or in addition). "
        "Addressed like project_read_file: by `project` + `path`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project name."},
            "path": {"type": "string", "description": "File path within the project."},
        },
        "required": ["project", "path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            pdir = _project_dir(params["project"])
        except ValueError as e:
            return f"Invalid project: {e}"

        rel = os.path.normpath(params["path"])
        if rel.startswith("..") or rel.startswith("/"):
            return "Invalid path: must be relative within the project."
        full_path = os.path.join(pdir, rel)
        if not os.path.isfile(full_path):
            return f"File not found: {params['path']} in project '{params['project']}'"

        name = os.path.basename(full_path)
        size = os.path.getsize(full_path)
        kind = _classify(full_path)
        payload = {"path": full_path, "name": name, "kind": kind, "size": size}

        if kind == "text":
            try:
                with open(full_path, "r", errors="replace") as f:
                    text = f.read()
            except OSError as e:
                return f"Could not read '{name}': {e}"
            if len(text) > _INLINE_TEXT_MAX:
                payload["kind"] = "file"  # too big to inline — offer as a download element
            else:
                payload["text"] = text

        hook = _display_hooks.get(threading.get_ident())
        if hook is None:
            return (f"display_doc: no interactive chat surface to display '{name}' to "
                    f"(this run isn't attached to a chat session).")
        try:
            hook(payload)
        except Exception as e:  # noqa: BLE001
            return f"display_doc: failed to display '{name}': {e}"
        return (
            f"Displayed '{name}' ({_human_size(size)}, {payload['kind']}) to the user in the chat. "
            f"The content was shown to the user directly and is NOT in your context — "
            f"use project_read_file if you need to read it yourself."
        )


@register_tool("download_file")
class DownloadFile(BaseTool):
    """Offer a file from a project directory for download in the chat UI."""

    name = "download_file"
    description = (
        "Offer a file from a project directory for download in the chat UI. "
        "Use for binary files (archives, executables, etc.) that don't render as text, image, or PDF. "
        "For documents, charts, or images, use display_doc instead. "
        "Addressed like display_doc: by `project` + `path`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project name."},
            "path": {"type": "string", "description": "File path within the project."},
        },
        "required": ["project", "path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            pdir = _project_dir(params["project"])
        except ValueError as e:
            return f"Invalid project: {e}"

        rel = os.path.normpath(params["path"])
        if rel.startswith("..") or rel.startswith("/"):
            return "Invalid path: must be relative within the project."
        full_path = os.path.join(pdir, rel)
        if not os.path.isfile(full_path):
            return f"File not found: {params['path']} in project '{params['project']}'"

        name = os.path.basename(full_path)
        size = os.path.getsize(full_path)
        payload = {"path": full_path, "name": name, "kind": "file", "size": size}

        hook = _display_hooks.get(threading.get_ident())
        if hook is None:
            return f"download_file: no interactive chat surface (not in a chat session)."
        try:
            hook(payload)
        except Exception as e:
            return f"download_file: failed to offer '{name}': {e}"
        return f"Download link offered for '{name}' ({_human_size(size)}) in the chat."
