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


def make_data_layer() -> SQLAlchemyDataLayer:
    conninfo = resolve_conninfo()
    _ensure_sqlite_schema(conninfo)
    # storage_provider lets create_element persist element blobs (images/PDFs from
    # display_doc) so they survive a reload — without it Chainlit drops elements
    # entirely. LocalFsStorageClient stores them under DATA_DIR/.cl_elements.
    from sandbox_agent.chat_storage import LocalFsStorageClient
    dl = SQLAlchemyDataLayer(conninfo=conninfo, storage_provider=LocalFsStorageClient())
    # Harden the SQLite connection pool against write-lock contention (the cause
    # of the QueuePool-exhaustion + streaming-lag). No-op for non-SQLite backends.
    if _sqlite_path_from_conninfo(conninfo):
        _harden_sqlite_engine(dl.engine)
    return dl
