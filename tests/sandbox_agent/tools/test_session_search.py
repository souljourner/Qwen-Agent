"""Weakness #2B: the agent must be able to search its own past sessions."""

import json
import os
import sqlite3
from datetime import datetime, timedelta

import pytest

import sandbox_agent.tools.session_search_tools as ss
from sandbox_agent.tools.session_search_tools import SessionSearch


def _make_chat_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute('''CREATE TABLE steps (
        "id" TEXT, "threadId" TEXT, "type" TEXT, "name" TEXT,
        "input" TEXT, "output" TEXT, "createdAt" TEXT)''')
    conn.executemany(
        'INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
    conn.commit()
    conn.close()


def _ts(days_ago=0):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DATA_DIR", str(tmp_path))
    _make_chat_db(str(tmp_path / "chat.db"), [
        ("s1", "th-1", "user_message", None, None, "what is the SOXS decay rate?", _ts(20)),
        ("s2", "th-1", "tool", "web_search", '{"query":"SOXS"}',
         "SOXS loses roughly seven percent monthly to leveraged decay", _ts(20)),
        ("s3", "th-1", "assistant_message", "Assistant", None,
         "Conclusion: SOXS decays about 7% per month; shorting it compounds.", _ts(20)),
        ("s4", "th-2", "assistant_message", "Assistant", None,
         "Cocoa futures speculators are net short.", _ts(2)),
        ("s5", "th-2", "thought", "thinking", None, "secret internal reasoning", _ts(2)),
    ])
    with open(tmp_path / "tasks_completed.json", "w") as f:
        json.dump([{
            "id": "task-9", "name": "Weekly COT Contrarian Scan",
            "description": "scan cot", "status": "completed",
            "result": "Sugar #11 speculators unwinding shorts rapidly",
            "updated_at": _ts(5),
        }], f)
    return tmp_path


def test_finds_old_conclusion(env):
    out = SessionSearch().call({"query": "SOXS decay"})
    assert "7%" in out or "seven percent" in out
    assert "th-1" in out                      # thread ref for follow-up


def test_finds_completed_task_results(env):
    out = SessionSearch().call({"query": "sugar speculators"})
    assert "#11" in out and "unwinding shorts" in out    # snippet marks the matched words
    assert "Weekly COT" in out


def test_thoughts_not_indexed(env):
    # The thought step's text must not be in the INDEX — the only place the
    # words can appear is the no-matches message echoing the query back.
    out = SessionSearch().call({"query": "secret internal reasoning"})
    assert out.lower().startswith("no matches")


def test_no_results_message(env):
    out = SessionSearch().call({"query": "quantum blockchain zebras"})
    assert "no matches" in out.lower()


def test_hostile_query_does_not_raise(env):
    out = SessionSearch().call({"query": 'AND OR "unbalanced ( NEAR'})
    assert isinstance(out, str)


def test_incremental_reindex_picks_up_new_rows(env):
    SessionSearch().call({"query": "SOXS"})           # builds index
    conn = sqlite3.connect(str(env / "chat.db"))
    conn.execute('INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, ?)',
                 ("s6", "th-3", "assistant_message", "Assistant", None,
                  "New finding: VIX correlation flipped positive.", _ts(0)))
    conn.commit()
    conn.close()
    out = SessionSearch().call({"query": "VIX correlation flipped"})
    assert "th-3" in out


def test_registered():
    from qwen_agent.tools.base import TOOL_REGISTRY
    import sandbox_agent.config as cfg
    assert "session_search" in TOOL_REGISTRY
    assert "session_search" in cfg.TOOL_LIST
