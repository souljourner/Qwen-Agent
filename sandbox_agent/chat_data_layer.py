"""Chainlit persistence wiring.

Provides a SQLAlchemy-backed data layer (SQLite by default) so chat threads,
messages, and steps survive process restarts and browser refreshes.

Connection string resolution order:
  1. `CHAT_DB_URL` env var (if set)
  2. `DATABASE_URL` env var (Chainlit convention)
  3. `sqlite+aiosqlite:///<DATA_DIR>/chat.db` — default

`DATA_DIR` resolves the same way as the rest of sandbox_agent (env var,
falling back to a project-local default in dev).

Chainlit's SQLAlchemy data layer assumes the schema is already in place —
it does NOT auto-create tables. We bootstrap the schema synchronously at
import time using sqlite3 (for SQLite URLs) so the async layer has tables
to query against.
"""

import logging
import os
import re
import sqlite3

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

logger = logging.getLogger(__name__)


# Schema published by Chainlit-community for the SQLAlchemy data layer:
# https://github.com/Chainlit/chainlit-community/blob/main/packages/data_layers/sqlalchemy/schema.sql
# Postgres types like UUID / JSONB / TEXT[] are accepted by SQLite (it
# stores everything as TEXT/BLOB regardless of the declared affinity).
_CHAINLIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    -- Newer Chainlit versions include these fields in step_dict; the
    -- chainlit-community schema we forked from predates them. Without these
    -- columns, every UPSERT fails ("table steps has no column named
    -- autoCollapse") and assistant_message rows persist with empty output.
    "autoCollapse" BOOLEAN,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" UUID PRIMARY KEY,
    "forId" UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
"""


def _default_sqlite_url() -> str:
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        # Dev fallback — a project-local path so a host-side `chainlit run`
        # works without the Docker bind mount.
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sandbox_agent_dev_data")
        os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.abspath(os.path.join(data_dir, "chat.db"))
    return f"sqlite+aiosqlite:///{db_path}"


def resolve_conninfo() -> str:
    return (
        os.environ.get("CHAT_DB_URL", "").strip()
        or os.environ.get("DATABASE_URL", "").strip()
        or _default_sqlite_url()
    )


def _sqlite_path_from_conninfo(conninfo: str) -> str | None:
    """Return the on-disk SQLite path if `conninfo` is a SQLite URL, else None.

    SQLAlchemy URL conventions:
      - `sqlite:///relative/path.db`    → 3 slashes after `://`, relative path
      - `sqlite:////absolute/path.db`   → 4 slashes, absolute path (the 4th
        slash IS the path's leading `/`)
    Same applies to the `+aiosqlite` variant.
    """
    m = re.match(r"sqlite(?:\+\w+)?:///(/?.*)$", conninfo)
    if not m:
        return None
    return m.group(1)


def _ensure_sqlite_schema(conninfo: str) -> None:
    """If `conninfo` points to a SQLite DB, create the Chainlit tables on
    first run AND apply additive migrations for columns added in newer
    Chainlit versions. No-op for any other backend (assume the operator
    pre-applied the schema)."""
    db_path = _sqlite_path_from_conninfo(conninfo)
    if not db_path:
        return
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # WAL is a persistent property of the DB file. Switch it on once here so
        # concurrent readers don't block the single writer — without it every
        # Chainlit persist (fire-and-forget create_step/update_step tasks, dozens
        # per multi-step turn) contends on a global lock, exhausting the
        # connection pool and starving the event loop that streams tokens.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_CHAINLIT_SCHEMA_SQL)
        conn.commit()

        # Additive migrations for older databases. SQLite ALTER TABLE only
        # supports ADD COLUMN, but that's all we need — Chainlit's data layer
        # does UPSERT with whatever columns step_dict happens to contain.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(steps)").fetchall()}
        for col, ddl in [
            ("autoCollapse", 'ALTER TABLE steps ADD COLUMN "autoCollapse" BOOLEAN'),
            ("modes",        'ALTER TABLE steps ADD COLUMN "modes" TEXT'),
        ]:
            if col not in existing_cols:
                conn.execute(ddl)
                logger.info("Chainlit SQLite migration: added steps.%s", col)
        conn.commit()
    logger.info("Chainlit SQLite schema applied at %s", db_path)


def _harden_sqlite_engine(engine) -> None:
    """Apply SQLite pragmas to every pooled connection.

    Chainlit persists each step send/update via a fire-and-forget
    `asyncio.create_task(data_layer.update_step(...))` — a chatty multi-step
    turn spawns dozens of concurrent DB writes. In the default journal_mode
    (DELETE) each write locks the whole file, so those tasks block on the lock
    while holding one of the engine's 15 pooled connections; the pool exhausts
    ("QueuePool limit ... reached, connection timed out") and the aiosqlite
    worker threads thrash the GIL, starving the event loop that emits stream
    tokens — the chat lags to ~1 token / 5s. WAL lets readers run during a
    write; busy_timeout makes the rare writer-vs-writer clash wait instead of
    erroring; synchronous=NORMAL drops the per-write fsync (safe under WAL).
    """
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.execute("PRAGMA synchronous=NORMAL")
        finally:
            cur.close()


class _SandboxDataLayer(SQLAlchemyDataLayer):
    """SQLAlchemyDataLayer with thread deletion that actually finishes the job.

    Chainlit's DELETE /project/thread is a plain HTTP endpoint: it removes the
    row and returns, never touching the websocket session. So after deleting
    the thread you are CURRENTLY viewing:

      * `session.thread_id` still points at the deleted thread, and
        `has_first_interaction` is still True — so `init_thread()` (the only
        place that sets a thread's name and userId) never fires again;
      * the next message calls create_step -> update_thread(thread_id) with no
        name and no user_id, which re-inserts the row with **userId NULL**;
      * list_threads filters on `WHERE "userId" = :user_id OR "id" = :thread_id`,
        so a NULL-userId thread can NEVER appear in the sidebar.

    Observed live 2026-08-06: 9 such orphaned threads, the newest holding 13
    steps of a real conversation that was invisible in the UI.

    Two further leaks, both of which mean "delete" did not delete: the
    in-session agent history stayed in memory, and the agent-side sidecar
    stayed on disk — so the apparently-fresh chat still carried the full
    context of the conversation the user had just deleted.
    """

    async def delete_thread(self, thread_id: str):
        await super().delete_thread(thread_id)
        _reset_sessions_on_thread(thread_id)
        _delete_history_sidecar(thread_id)

    async def update_thread(self, thread_id: str, name=None, user_id=None,
                            metadata=None, tags=None):
        """Backstop: never let a thread row be created without an owner.

        create_step() calls update_thread(thread_id) with no user_id. If the
        row does not exist that INSERTs an ownerless, permanently invisible
        thread. Filling the owner from the live session makes that impossible
        regardless of how we got here.
        """
        if user_id is None:
            user_id = _current_user_id()
        return await super().update_thread(thread_id, name=name, user_id=user_id,
                                           metadata=metadata, tags=tags)


def _current_user_id():
    """PersistedUser.id for the active session, or None outside a context."""
    try:
        from chainlit.context import context
        user = getattr(getattr(context, "session", None), "user", None)
        return getattr(user, "id", None)
    except Exception:  # noqa: BLE001 — no context (HTTP/background); caller copes
        return None


def _reset_sessions_on_thread(thread_id: str) -> None:
    """Give any live session viewing `thread_id` a genuinely fresh thread.

    Without this the UI looks like a new chat while the session still writes
    into the deleted thread id.
    """
    import uuid
    try:
        from chainlit.session import ws_sessions_id
        from chainlit.user_session import user_sessions
    except Exception:  # noqa: BLE001
        return
    for session in list(ws_sessions_id.values()):
        if getattr(session, "thread_id", None) != thread_id:
            continue
        session.thread_id = str(uuid.uuid4())
        session.thread_id_to_resume = None
        # False so the next message runs init_thread() again, which is what
        # stamps name + userId onto the new row.
        session.has_first_interaction = False
        # Drop the agent-side history: the deleted conversation must not keep
        # feeding the model through an apparently-new chat.
        store = user_sessions.get(session.id)
        if isinstance(store, dict):
            for key in ("history", "_frozen_metadata"):
                store.pop(key, None)
        logger.info("Thread %s deleted — session %s reset to fresh thread %s",
                    thread_id, session.id, session.thread_id)


def _delete_history_sidecar(thread_id: str) -> None:
    """Remove the agent-side history file; deleting a chat must delete it."""
    try:
        from sandbox_agent.chat_history import _path
        path = _path(thread_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Removed agent history sidecar for deleted thread %s", thread_id)
    except Exception:  # noqa: BLE001 — never fail a delete over cleanup
        logger.exception("Could not remove sidecar for deleted thread %s", thread_id)


def make_data_layer() -> SQLAlchemyDataLayer:
    conninfo = resolve_conninfo()
    _ensure_sqlite_schema(conninfo)
    # storage_provider lets create_element persist element blobs (images/PDFs from
    # display_doc) so they survive a reload — without it Chainlit drops elements
    # entirely. LocalFsStorageClient stores them under DATA_DIR/.cl_elements.
    from sandbox_agent.chat_storage import LocalFsStorageClient
    dl = _SandboxDataLayer(conninfo=conninfo, storage_provider=LocalFsStorageClient())
    # Harden the SQLite connection pool against write-lock contention (the cause
    # of the QueuePool-exhaustion + streaming-lag). No-op for non-SQLite backends.
    if _sqlite_path_from_conninfo(conninfo):
        _harden_sqlite_engine(dl.engine)
    return dl
