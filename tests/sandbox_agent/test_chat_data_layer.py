"""Tests for chat_data_layer SQLite concurrency hardening.

Root cause of the Chainlit streaming lag + repeated `QueuePool limit reached`
warnings: the SQLite data layer ran in journal_mode=delete with busy_timeout=0,
so Chainlit's fire-and-forget persist tasks (one per step send/update) blocked
on the global write lock, exhausting the 15-connection pool and starving the
event loop. The fix puts the DB in WAL mode with a busy_timeout so writes don't
lock the whole file and contended writes wait briefly instead of piling up.
"""

import asyncio
import sqlite3


def test_make_data_layer_enables_wal_and_busy_timeout(tmp_path, monkeypatch):
    db = tmp_path / "chat.db"
    monkeypatch.setenv("CHAT_DB_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    import importlib
    import sandbox_agent.chat_data_layer as cdl
    importlib.reload(cdl)
    dl = cdl.make_data_layer()

    # WAL is a persistent property of the DB file — a fresh sqlite3 connection
    # should report it.
    jm = sqlite3.connect(str(db)).execute("PRAGMA journal_mode").fetchone()[0]
    assert jm.lower() == "wal", f"expected WAL, got {jm!r}"

    # Each engine connection (what Chainlit's data layer actually uses) must get
    # WAL + a non-zero busy_timeout applied.
    async def _probe():
        async with dl.engine.connect() as conn:
            bt = (await conn.exec_driver_sql("PRAGMA busy_timeout")).scalar()
            mode = (await conn.exec_driver_sql("PRAGMA journal_mode")).scalar()
            return int(bt), str(mode).lower()

    try:
        busy_timeout, mode = asyncio.run(_probe())
    finally:
        asyncio.run(dl.engine.dispose())

    assert busy_timeout >= 30000, f"busy_timeout too low: {busy_timeout}"
    assert mode == "wal", f"engine connection not in WAL: {mode!r}"
