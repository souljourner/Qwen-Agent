"""session_search — full-text recall over past chat sessions and task results.

Weakness #2B: everything the agent ever concluded is persisted (chat.db steps,
tasks_completed.json) but nothing could FIND it — "what did we decide three
weeks ago?" meant hand-grepping markdown logs. This tool maintains an FTS5
index (DATA_DIR/session_index.db, incrementally refreshed on each call) over
user/assistant messages, tool outputs, and completed-task results. Thought
steps are never indexed (internal reasoning stays internal).
"""

import json
import logging
import os
import sqlite3
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

_INDEXED_STEP_TYPES = ("user_message", "assistant_message", "tool")
_MAX_ENTRY_CHARS = 20_000


def _index_path() -> str:
    return os.path.join(DATA_DIR, "session_index.db")


def _open_index() -> sqlite3.Connection:
    conn = sqlite3.connect(_index_path())
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS entries USING fts5(
        content, source UNINDEXED, ref UNINDEXED, label UNINDEXED,
        created_at UNINDEXED)""")
    conn.execute("CREATE TABLE IF NOT EXISTS index_state (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def _get_state(conn, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM index_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def _set_state(conn, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO index_state VALUES (?, ?)", (key, value))


def _refresh_chat(conn) -> int:
    """Pull steps newer than the last indexed createdAt from chat.db."""
    chat_db = os.path.join(DATA_DIR, "chat.db")
    if not os.path.exists(chat_db):
        return 0
    last = _get_state(conn, "chat_last_created_at")
    try:
        src = sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True)
        placeholders = ",".join("?" for _ in _INDEXED_STEP_TYPES)
        rows = src.execute(
            f'SELECT "threadId", "type", "name", "output", "createdAt" FROM steps '
            f'WHERE "createdAt" > ? AND "type" IN ({placeholders}) '
            f'AND "output" IS NOT NULL AND "output" != ""',
            (last, *_INDEXED_STEP_TYPES)).fetchall()
        src.close()
    except sqlite3.Error:
        logger.exception("session_search: cannot read chat.db")
        return 0
    newest = last
    for thread_id, stype, name, output, created_at in rows:
        # display_doc bodies are shown to the user, not agent context — but
        # they ARE past knowledge; index them like everything else except
        # internal reasoning (thought steps are excluded by type filter).
        conn.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
            (str(output)[:_MAX_ENTRY_CHARS], "chat", thread_id,
             name or stype, created_at))
        if created_at and created_at > newest:
            newest = created_at
    if rows:
        _set_state(conn, "chat_last_created_at", newest)
    return len(rows)


def _refresh_tasks(conn) -> int:
    """Index completed-task results newer than the last indexed updated_at."""
    path = os.path.join(DATA_DIR, "tasks_completed.json")
    try:
        mtime = str(os.path.getmtime(path))
    except OSError:
        return 0
    if _get_state(conn, "tasks_mtime") == mtime:
        return 0
    try:
        with open(path) as f:
            tasks = json.load(f)
    except (OSError, ValueError):
        return 0
    last = _get_state(conn, "tasks_last_updated_at")
    newest = last
    n = 0
    for t in tasks:
        updated = str(t.get("updated_at") or t.get("created_at") or "")
        if updated <= last:
            continue
        text = " — ".join(filter(None, [t.get("name"), t.get("result")]))
        if not text:
            continue
        conn.execute("INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
                     (text[:_MAX_ENTRY_CHARS], "task", t.get("id", "?"),
                      t.get("name", "task"), updated))
        n += 1
        if updated > newest:
            newest = updated
    _set_state(conn, "tasks_last_updated_at", newest)
    _set_state(conn, "tasks_mtime", mtime)
    return n


def _fts_query(query: str) -> str:
    """Quote each token so user text can't break FTS5 syntax (implicit AND)."""
    tokens = [t.replace('"', "") for t in query.split()]
    return " ".join(f'"{t}"' for t in tokens if t)


@register_tool("session_search")
class SessionSearch(BaseTool):
    """Search past conversations and completed-task results."""

    name = "session_search"
    description = (
        "Full-text search over ALL past chat sessions and completed background-task "
        "results. Use when the user references earlier work ('we tried this', 'last "
        "time', 'what did we decide about X', 'remember when') or when you suspect "
        "relevant prior context exists — search before asking the user to repeat "
        "themselves or redoing research. Returns dated snippets with thread/task "
        "references; follow up with read_chat_log (by date) or list_tasks for detail."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms (matched against messages, tool outputs, task results).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 8).",
            },
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = (params.get("query") or "").strip()
        if not query:
            return "Error: query is required."
        limit = int(params.get("limit") or 8)

        conn = _open_index()
        try:
            _refresh_chat(conn)
            _refresh_tasks(conn)
            conn.commit()
            fts = _fts_query(query)
            if not fts:
                return "Error: query contained no searchable terms."
            try:
                rows = conn.execute(
                    "SELECT snippet(entries, 0, '»', '«', ' … ', 16), "
                    "source, ref, label, created_at FROM entries WHERE entries MATCH ? "
                    "ORDER BY rank LIMIT ?", (fts, limit)).fetchall()
            except sqlite3.OperationalError:
                # Pathological query even after quoting — degrade to LIKE.
                like = f"%{query}%"
                rows = conn.execute(
                    "SELECT substr(content, 1, 200), source, ref, label, created_at "
                    "FROM entries WHERE content LIKE ? LIMIT ?", (like, limit)).fetchall()
        finally:
            conn.close()

        if not rows:
            return (f"No matches for '{query}' in past sessions or task results. "
                    f"Try broader/different terms.")
        lines = [f"Matches for '{query}' (newest context refs first help most):"]
        for snippet, source, ref, label, created_at in rows:
            date = str(created_at)[:10]
            where = f"thread {ref}" if source == "chat" else f"task {ref}"
            lines.append(f"- [{date}] ({where}, {label}) {snippet}")
        return "\n".join(lines)
