"""Tests for tiered model concurrency: laguna-s-2.1 primary (3 turn slots),
qwen3.6-27b-linux secondary/overflow (10 turn slots).

Selection is per-TURN: a slot is held for the whole streamed turn. Chat
prefers primary and spills to secondary; background work prefers secondary
and spills INTO primary only when all 10 secondary slots are busy; when all
13 are busy, callers wait (no more ungated pile-on)."""

import threading
import time
from threading import BoundedSemaphore

import pytest

from qwen_agent.llm.schema import Message

import sandbox_agent.main as m


@pytest.fixture
def fresh_locks(monkeypatch):
    """Small real semaphores so tests exercise true acquire/release."""
    primary = BoundedSemaphore(3)
    secondary = BoundedSemaphore(10)
    monkeypatch.setattr(m, "_primary_model_lock", primary)
    monkeypatch.setattr(m, "_secondary_model_lock", secondary)
    return primary, secondary


class TestAcquireTurnSlot:

    def test_prefers_primary_then_spills(self, fresh_locks):
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(4)]
        tiers = [g[0] for g in grants]
        assert tiers == ["primary", "primary", "primary", "secondary"]
        for _, release in grants:
            release()

    def test_prefer_secondary_order(self, fresh_locks):
        tier, release = m._acquire_turn_slot(blocking=False, prefer="secondary")
        assert tier == "secondary"
        release()

    def test_none_when_all_thirteen_held(self, fresh_locks):
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(13)]
        assert all(g is not None for g in grants)
        assert m._acquire_turn_slot(blocking=False) is None
        for _, release in grants:
            release()

    def test_blocking_wakes_when_any_tier_frees(self, fresh_locks):
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(13)]
        released = grants[5]  # a secondary slot

        def free_one():
            time.sleep(0.05)
            released[1]()

        t = threading.Thread(target=free_one)
        t.start()
        tier, release = m._acquire_turn_slot(blocking=True, poll_interval=0.01)
        t.join()
        assert tier == "secondary"
        release()
        for i, (tier_name, rel) in enumerate(grants):
            if i != 5:
                rel()

    def test_release_is_idempotent(self, fresh_locks):
        primary, _ = fresh_locks
        tier, release = m._acquire_turn_slot(blocking=False)
        release()
        release()  # second call must be a no-op, not a BoundedSemaphore error
        # all 3 primary slots must be available again — not 4
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(3)]
        assert [g[0] for g in grants] == ["primary"] * 3
        for _, rel in grants:
            rel()


class _FakeAgent:
    """Stands in for a model-bound Assistant."""

    def __init__(self, name, replies=("hello",)):
        self.name = name
        self.calls = 0
        self._replies = list(replies)
        self.llm = type("L", (), {"generate_cfg": {}})()

    def run(self, messages, **kwargs):
        self.calls += 1
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        yield [Message(role="assistant", content=reply)]

    def run_nonstream(self, messages, **kwargs):
        self.calls += 1
        return [Message(role="assistant", content="ns")]


class TestLockingAgentRouting:

    def _agent(self):
        return m.LockingAgent(_FakeAgent("primary-model"), _FakeAgent("secondary-model"))

    def _drain(self, gen):
        out = []
        for out in gen:
            pass
        return out

    def test_routes_primary_when_free(self, fresh_locks):
        agent = self._agent()
        self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        assert agent._inner.calls == 1
        assert agent._backup.calls == 0

    def test_spills_to_secondary_when_primary_exhausted(self, fresh_locks):
        primary, _ = fresh_locks
        for _ in range(3):
            primary.acquire(blocking=False)
        agent = self._agent()
        self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        assert agent._backup.calls == 1
        assert agent._inner.calls == 0
        for _ in range(3):
            primary.release()

    def test_slot_released_after_turn(self, fresh_locks):
        agent = self._agent()
        self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        # all three primary slots free again
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(3)]
        assert [g[0] for g in grants] == ["primary"] * 3
        for _, rel in grants:
            rel()

    def test_slot_released_on_midstream_exception(self, fresh_locks):
        class _Boom(_FakeAgent):
            def run(self, messages, **kwargs):
                yield [Message(role="assistant", content="partial")]
                raise RuntimeError("boom")

        agent = m.LockingAgent(_Boom("primary-model"), _FakeAgent("secondary-model"))
        with pytest.raises(RuntimeError):
            self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(3)]
        assert [g[0] for g in grants] == ["primary"] * 3
        for _, rel in grants:
            rel()

    def test_retry_on_empty_uses_other_tier_and_restores_counts(self, fresh_locks):
        primary, secondary = fresh_locks
        empty_then = _FakeAgent("primary-model", replies=("",))
        agent = m.LockingAgent(empty_then, _FakeAgent("secondary-model"))
        out = self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        assert agent._backup.calls == 1  # retry ran on the other tier
        # semaphore counts fully restored
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(13)]
        assert all(g is not None for g in grants)
        for _, rel in grants:
            rel()


class TestRunOnBestAvailable:

    def test_background_prefers_secondary(self, fresh_locks, monkeypatch):
        used_cfgs = []

        def fake_create_agent(system_message, llm_cfg=None):
            used_cfgs.append(llm_cfg["model"])
            return _FakeAgent(llm_cfg["model"])

        monkeypatch.setattr(m, "create_agent", fake_create_agent)
        monkeypatch.setattr(m, "model_start", lambda *a, **k: None)
        done = []
        monkeypatch.setattr(m, "model_done", lambda model: done.append(model))
        m.run_on_best_available("sys", [Message(role="user", content="task")],
                                task_label="test-task")
        from sandbox_agent.config import BACKGROUND_LLM_CFG
        assert used_cfgs == [BACKGROUND_LLM_CFG["model"]]
        assert done == [BACKGROUND_LLM_CFG["model"]]

    def test_background_spills_into_primary_when_secondary_full(self, fresh_locks, monkeypatch):
        _, secondary = fresh_locks
        for _ in range(10):
            secondary.acquire(blocking=False)
        used_cfgs = []

        def fake_create_agent(system_message, llm_cfg=None):
            used_cfgs.append(llm_cfg["model"])
            return _FakeAgent(llm_cfg["model"])

        monkeypatch.setattr(m, "create_agent", fake_create_agent)
        monkeypatch.setattr(m, "model_start", lambda *a, **k: None)
        monkeypatch.setattr(m, "model_done", lambda *a, **k: None)
        m.run_on_best_available("sys", [Message(role="user", content="task")],
                                task_label="test-task")
        from sandbox_agent.config import PRIMARY_LLM_CFG
        assert used_cfgs == [PRIMARY_LLM_CFG["model"]]
        for _ in range(10):
            secondary.release()


class TestConfigAndBridge:

    def test_tier_models_and_concurrency(self):
        from sandbox_agent.config import (BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG,
                                          PRIMARY_MODEL_CONCURRENCY,
                                          SECONDARY_MODEL_CONCURRENCY)
        assert PRIMARY_LLM_CFG["model"] == "laguna-s-2.1"
        assert BACKGROUND_LLM_CFG["model"] == "qwen3.6-27b-linux"
        assert PRIMARY_MODEL_CONCURRENCY == 3
        assert SECONDARY_MODEL_CONCURRENCY == 10

    def test_bridge_chain_is_secondary_first(self):
        from sandbox_agent.tools.llm_bridge import _build_fallback_chain
        chain = _build_fallback_chain(None)
        models = [c[0] for c in chain]
        # bulk/unslotted traffic belongs on the high-concurrency tier
        assert models[0] == "qwen3.6-27b-linux"
        assert models[1] == "laguna-s-2.1"
