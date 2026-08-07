"""Deleting the currently-open chat must leave a genuinely fresh session.

Live bug 2026-08-06: delete the open conversation, send a message, and the
new chat never appeared in the left nav. Chainlit's DELETE /project/thread
is a plain HTTP endpoint that never touches the websocket session, so:

  * session.thread_id still pointed at the deleted thread and
    has_first_interaction was still True, so init_thread() — the only place
    that stamps a thread's name and userId — never fired again;
  * the next message's create_step -> update_thread(thread_id) re-INSERTed
    the row with userId NULL (None values are stripped from the INSERT);
  * list_threads filters on "userId" = :user_id, so the thread was
    permanently invisible.

The real database held 9 such threads, the newest with 13 steps.
"""

import asyncio
import os
import uuid

import pytest

import sandbox_agent.chat_data_layer as dl


class _FakeSession:
    def __init__(self, thread_id, sid="sess-1"):
        self.id = sid
        self.thread_id = thread_id
        self.thread_id_to_resume = thread_id
        self.has_first_interaction = True


@pytest.fixture
def live_session(monkeypatch):
    import chainlit.session as cs
    # NB: `import chainlit.user_session as x` binds the UserSession INSTANCE
    # (the package re-exports it under that name), not the module. The dict
    # only reachable via a from-import.
    from chainlit.user_session import user_sessions as cus
    tid = str(uuid.uuid4())
    sess = _FakeSession(tid)
    monkeypatch.setitem(cs.ws_sessions_id, sess.id, sess)
    cus[sess.id] = {"history": ["old turn"], "_frozen_metadata": {"x": 1}}
    yield tid, sess
    cus.pop(sess.id, None)


class TestSessionResetOnDelete:

    def test_thread_id_changes(self, live_session):
        tid, sess = live_session
        dl._reset_sessions_on_thread(tid)
        assert sess.thread_id != tid, "session still writes into the deleted thread"
        assert sess.thread_id_to_resume is None

    def test_first_interaction_flag_is_cleared(self, live_session):
        """This is the flag that gates init_thread(), which sets name+userId.
        Left True, the recreated row is ownerless and invisible."""
        tid, sess = live_session
        dl._reset_sessions_on_thread(tid)
        assert sess.has_first_interaction is False

    def test_agent_history_is_dropped(self, live_session):
        """Otherwise the 'new' chat still feeds the model the conversation
        the user just deleted."""
        from chainlit.user_session import user_sessions
        tid, sess = live_session
        dl._reset_sessions_on_thread(tid)
        assert user_sessions[sess.id].get("history") is None
        assert user_sessions[sess.id].get("_frozen_metadata") is None

    def test_other_sessions_are_untouched(self, live_session, monkeypatch):
        import chainlit.session as cs
        tid, sess = live_session
        other = _FakeSession("someone-elses-thread", sid="sess-2")
        monkeypatch.setitem(cs.ws_sessions_id, other.id, other)
        dl._reset_sessions_on_thread(tid)
        assert other.thread_id == "someone-elses-thread"
        assert other.has_first_interaction is True


class TestSidecarRemoval:

    def test_sidecar_is_deleted(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_history as ch
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        tid = "thread-abc"
        os.makedirs(os.path.dirname(ch._path(tid)), exist_ok=True)
        with open(ch._path(tid), "w") as f:
            f.write("[]")
        dl._delete_history_sidecar(tid)
        assert not os.path.exists(ch._path(tid)), (
            "agent history survived deletion — 'delete' did not delete")

    def test_missing_sidecar_is_not_an_error(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_history as ch
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        dl._delete_history_sidecar("never-existed")  # must not raise


class TestOwnerlessThreadsAreImpossible:

    def test_update_thread_fills_user_id_from_session(self, monkeypatch):
        """create_step() calls update_thread(thread_id) with no user_id. If
        the row is missing that INSERTs an ownerless, invisible thread."""
        seen = {}

        async def fake_super(self, thread_id, name=None, user_id=None,
                             metadata=None, tags=None):
            seen["user_id"] = user_id

        monkeypatch.setattr(dl.SQLAlchemyDataLayer, "update_thread", fake_super)
        monkeypatch.setattr(dl, "_current_user_id", lambda: "user-123")
        layer = dl._SandboxDataLayer.__new__(dl._SandboxDataLayer)
        asyncio.run(layer.update_thread("t1"))
        assert seen["user_id"] == "user-123", "thread would be created ownerless"

    def test_explicit_user_id_wins(self, monkeypatch):
        seen = {}

        async def fake_super(self, thread_id, name=None, user_id=None,
                             metadata=None, tags=None):
            seen["user_id"] = user_id

        monkeypatch.setattr(dl.SQLAlchemyDataLayer, "update_thread", fake_super)
        monkeypatch.setattr(dl, "_current_user_id", lambda: "wrong")
        layer = dl._SandboxDataLayer.__new__(dl._SandboxDataLayer)
        asyncio.run(layer.update_thread("t1", user_id="explicit"))
        assert seen["user_id"] == "explicit"

    def test_no_context_is_survivable(self, monkeypatch):
        """Background/HTTP paths have no session; must not raise."""
        assert dl._current_user_id() is None or isinstance(dl._current_user_id(), str)


class TestDeleteWiring:
    """The overrides must actually be on the layer the app builds."""

    def test_make_data_layer_returns_the_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHAT_DB_URL", f"sqlite+aiosqlite:///{tmp_path}/c.db")
        import importlib
        importlib.reload(dl)
        layer = dl.make_data_layer()
        assert isinstance(layer, dl._SandboxDataLayer), (
            "app is using the stock data layer; the fix is not wired in")
        importlib.reload(dl)


class TestDeleteThreadDrivesEverything:
    """Drives delete_thread() itself. The helper tests above pass even if the
    calls are removed from delete_thread — this is the one that doesn't."""

    def test_delete_thread_resets_session_and_removes_sidecar(
            self, live_session, tmp_path, monkeypatch):
        import sandbox_agent.chat_history as ch
        from chainlit.user_session import user_sessions
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        tid, sess = live_session

        os.makedirs(os.path.dirname(ch._path(tid)), exist_ok=True)
        with open(ch._path(tid), "w") as f:
            f.write('[{"role": "user", "content": "secret"}]')

        async def fake_super(self, thread_id):
            return None

        monkeypatch.setattr(dl.SQLAlchemyDataLayer, "delete_thread", fake_super)
        layer = dl._SandboxDataLayer.__new__(dl._SandboxDataLayer)
        asyncio.run(layer.delete_thread(tid))

        assert sess.thread_id != tid, "delete_thread left the session on the dead thread"
        assert sess.has_first_interaction is False, (
            "delete_thread did not clear has_first_interaction — the next "
            "message recreates an ownerless, invisible thread")
        assert user_sessions[sess.id].get("history") is None
        assert not os.path.exists(ch._path(tid)), "delete_thread left the sidecar"
