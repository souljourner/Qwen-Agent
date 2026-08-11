"""Tests for tiered model concurrency: inkling-small primary (2 turn slots),
qwen3.6-27b-linux secondary/overflow (10 turn slots).

Selection is per-TURN: a slot is held for the whole streamed turn. Chat
prefers primary and spills to secondary. Background work ALSO prefers primary
(changed 2026-08-08 — unattended work gets the better model) and spills to
secondary when the 2 primary slots are full; when every slot is busy, callers
wait (no more ungated pile-on).

Both tiers run qwen3.6-27b and both are multimodal, so there is deliberately
NO vision pin — image turns route like any other turn."""

import threading
import time
from threading import BoundedSemaphore

import pytest

from qwen_agent.llm.schema import Message

import sandbox_agent.main as m
from sandbox_agent.config import (PRIMARY_MODEL_CONCURRENCY,
                                  SECONDARY_MODEL_CONCURRENCY)

TOTAL_SLOTS = PRIMARY_MODEL_CONCURRENCY + SECONDARY_MODEL_CONCURRENCY


@pytest.fixture(autouse=True)
def _clean_health():
    """model_health is process-global and reorders tiers. Without this, a
    leaked unhealthy tier from another test silently flips the routing these
    tests assert on — test_background_prefers_* passed in the full suite and
    failed in isolation for exactly that reason."""
    from sandbox_agent import model_health
    model_health.reset()
    yield
    model_health.reset()


@pytest.fixture
def fresh_locks(monkeypatch):
    """Small real semaphores so tests exercise true acquire/release.

    Sized FROM CONFIG, not hardcoded: these previously hardcoded 3 primary
    slots, so changing PRIMARY_MODEL_CONCURRENCY silently left the tests
    asserting the old topology."""
    primary = BoundedSemaphore(PRIMARY_MODEL_CONCURRENCY)
    secondary = BoundedSemaphore(SECONDARY_MODEL_CONCURRENCY)
    monkeypatch.setattr(m, "_primary_model_lock", primary)
    monkeypatch.setattr(m, "_secondary_model_lock", secondary)
    return primary, secondary


class TestAcquireTurnSlot:

    def test_prefers_primary_then_spills(self, fresh_locks):
        grants = [m._acquire_turn_slot(blocking=False)
                  for _ in range(PRIMARY_MODEL_CONCURRENCY + 1)]
        tiers = [g[0] for g in grants]
        assert tiers == ["primary"] * PRIMARY_MODEL_CONCURRENCY + ["secondary"]
        for _, release in grants:
            release()

    def test_prefer_secondary_order(self, fresh_locks):
        tier, release = m._acquire_turn_slot(blocking=False, prefer="secondary")
        assert tier == "secondary"
        release()

    def test_none_when_every_slot_held(self, fresh_locks):
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(TOTAL_SLOTS)]
        assert all(g is not None for g in grants)
        assert m._acquire_turn_slot(blocking=False) is None
        for _, release in grants:
            release()

    def test_blocking_wakes_when_any_tier_frees(self, fresh_locks):
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(TOTAL_SLOTS)]
        released = grants[PRIMARY_MODEL_CONCURRENCY + 1]  # a secondary slot

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
            if i != PRIMARY_MODEL_CONCURRENCY + 1:
                rel()

    def test_release_is_idempotent(self, fresh_locks):
        primary, _ = fresh_locks
        tier, release = m._acquire_turn_slot(blocking=False)
        release()
        release()  # second call must be a no-op, not a BoundedSemaphore error
        # every primary slot must be available again — and no more than that
        grants = [m._acquire_turn_slot(blocking=False)
                  for _ in range(PRIMARY_MODEL_CONCURRENCY)]
        assert [g[0] for g in grants] == ["primary"] * PRIMARY_MODEL_CONCURRENCY
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
        for _ in range(PRIMARY_MODEL_CONCURRENCY):
            primary.acquire(blocking=False)
        agent = self._agent()
        self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        assert agent._backup.calls == 1
        assert agent._inner.calls == 0
        for _ in range(PRIMARY_MODEL_CONCURRENCY):
            primary.release()

    def test_slot_released_after_turn(self, fresh_locks):
        agent = self._agent()
        self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        # every primary slot free again
        grants = [m._acquire_turn_slot(blocking=False)
                  for _ in range(PRIMARY_MODEL_CONCURRENCY)]
        assert [g[0] for g in grants] == ["primary"] * PRIMARY_MODEL_CONCURRENCY
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
        grants = [m._acquire_turn_slot(blocking=False)
                  for _ in range(PRIMARY_MODEL_CONCURRENCY)]
        assert [g[0] for g in grants] == ["primary"] * PRIMARY_MODEL_CONCURRENCY
        for _, rel in grants:
            rel()

    def test_retry_on_empty_uses_other_tier_and_restores_counts(self, fresh_locks):
        primary, secondary = fresh_locks
        empty_then = _FakeAgent("primary-model", replies=("",))
        agent = m.LockingAgent(empty_then, _FakeAgent("secondary-model"))
        out = self._drain(agent.run(messages=[Message(role="user", content="hi")]))
        assert agent._backup.calls == 1  # retry ran on the other tier
        # semaphore counts fully restored
        grants = [m._acquire_turn_slot(blocking=False) for _ in range(TOTAL_SLOTS)]
        assert all(g is not None for g in grants)
        for _, rel in grants:
            rel()


class TestRunOnBestAvailable:

    def test_background_prefers_primary(self, fresh_locks, monkeypatch):
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
        from sandbox_agent.config import PRIMARY_LLM_CFG
        assert used_cfgs == [PRIMARY_LLM_CFG["model"]]
        assert done == [PRIMARY_LLM_CFG["model"]]

    def test_background_spills_into_secondary_when_primary_full(self, fresh_locks, monkeypatch):
        primary, _ = fresh_locks
        for _ in range(PRIMARY_MODEL_CONCURRENCY):
            primary.acquire(blocking=False)
        used_cfgs = []

        def fake_create_agent(system_message, llm_cfg=None):
            used_cfgs.append(llm_cfg["model"])
            return _FakeAgent(llm_cfg["model"])

        monkeypatch.setattr(m, "create_agent", fake_create_agent)
        monkeypatch.setattr(m, "model_start", lambda *a, **k: None)
        monkeypatch.setattr(m, "model_done", lambda *a, **k: None)
        m.run_on_best_available("sys", [Message(role="user", content="task")],
                                task_label="test-task")
        from sandbox_agent.config import BACKGROUND_LLM_CFG
        assert used_cfgs == [BACKGROUND_LLM_CFG["model"]]
        for _ in range(PRIMARY_MODEL_CONCURRENCY):
            primary.release()


class TestConfigAndBridge:

    def test_tier_models_and_concurrency(self):
        from sandbox_agent.config import (BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG,
                                          PRIMARY_MODEL_CONCURRENCY,
                                          SECONDARY_MODEL_CONCURRENCY)
        assert PRIMARY_LLM_CFG["model"] == "inkling-small"
        assert BACKGROUND_LLM_CFG["model"] == "qwen3.6-27b-linux"
        assert PRIMARY_MODEL_CONCURRENCY == 2
        assert SECONDARY_MODEL_CONCURRENCY == 10

    def test_bridge_chain_is_secondary_first(self):
        from sandbox_agent.tools.llm_bridge import _build_fallback_chain
        chain = _build_fallback_chain(None)
        models = [c[0] for c in chain]
        # bulk/unslotted traffic belongs on the high-concurrency tier
        assert models[0] == "qwen3.6-27b-linux"
        assert models[1] == "inkling-small"


class TestOnlyTierAcquisition:

    def test_only_primary_never_grants_secondary(self, fresh_locks):
        primary, _ = fresh_locks
        for _ in range(PRIMARY_MODEL_CONCURRENCY):
            primary.acquire(blocking=False)
        assert m._acquire_turn_slot(blocking=False, only="primary") is None

    def test_only_primary_blocks_until_primary_frees(self, fresh_locks):
        primary, _ = fresh_locks
        for _ in range(PRIMARY_MODEL_CONCURRENCY):
            primary.acquire(blocking=False)

        def free_primary():
            time.sleep(0.05)
            primary.release()

        t = threading.Thread(target=free_primary)
        t.start()
        tier, release = m._acquire_turn_slot(blocking=True, only="primary",
                                             poll_interval=0.01)
        t.join()
        assert tier == "primary"
        release()
        for _ in range(PRIMARY_MODEL_CONCURRENCY - 1):
            primary.release()


class TestPinnedRouting:

    def _drain(self, gen):
        out = []
        for out in gen:
            pass
        return out

    def _big_messages(self):
        # > SPILLABLE_CONTEXT_TOKENS estimated
        from sandbox_agent.config import SPILLABLE_CONTEXT_TOKENS
        n_chars = (SPILLABLE_CONTEXT_TOKENS + 20_000) * 4
        return [Message(role="user", content="x" * n_chars)]

    def _image_messages(self):
        from qwen_agent.llm.schema import ContentItem
        return [Message(role="user", content=[ContentItem(image="data:image/png;base64,AAAA"),
                                              ContentItem(text="what is this?")])]

    def test_big_context_pins_to_primary(self, fresh_locks):
        agent = m.LockingAgent(_FakeAgent("primary-model"), _FakeAgent("secondary-model"))
        self._drain(agent.run(messages=self._big_messages()))
        assert agent._inner.calls == 1 and agent._backup.calls == 0

    def test_big_context_waits_for_primary_even_when_secondary_free(self, fresh_locks):
        primary, _ = fresh_locks
        for _ in range(PRIMARY_MODEL_CONCURRENCY):
            primary.acquire(blocking=False)

        agent = m.LockingAgent(_FakeAgent("primary-model"), _FakeAgent("secondary-model"))
        done = []

        def run_pinned():
            self._drain(agent.run(messages=self._big_messages()))
            done.append(True)

        t = threading.Thread(target=run_pinned)
        t.start()
        time.sleep(0.1)
        assert not done  # waiting on primary, NOT running on free secondary
        primary.release()
        t.join(timeout=5)
        assert done and agent._inner.calls == 1 and agent._backup.calls == 0
        for _ in range(PRIMARY_MODEL_CONCURRENCY - 1):
            primary.release()

    def test_image_history_is_not_pinned(self, fresh_locks):
        """Both tiers are multimodal qwen3.6-27b, so an image turn takes the
        preferred (primary) tier like anything else. Pinning it to secondary
        would waste the primary's capacity for no capability gain."""
        agent = m.LockingAgent(_FakeAgent("primary-model"), _FakeAgent("secondary-model"))
        self._drain(agent.run(messages=self._image_messages()))
        assert agent._inner.calls == 1 and agent._backup.calls == 0

    def test_big_context_with_images_still_pins_to_primary(self, fresh_locks):
        """Size is now the ONLY pin. Previously vision outranked it and sent
        this to secondary."""
        msgs = self._big_messages() + self._image_messages()
        agent = m.LockingAgent(_FakeAgent("primary-model"), _FakeAgent("secondary-model"))
        self._drain(agent.run(messages=msgs))
        assert agent._inner.calls == 1 and agent._backup.calls == 0

    def test_image_retry_may_use_either_tier(self, fresh_locks):
        """An empty completion on an image turn retries on the other tier —
        no longer forbidden, since both tiers see images."""
        empty_primary = _FakeAgent("primary-model", replies=("",))
        agent = m.LockingAgent(empty_primary, _FakeAgent("secondary-model"))
        self._drain(agent.run(messages=self._image_messages()))
        assert agent._backup.calls == 1, "retry did not reach the other tier"


class TestCreateAgentBudgetBinding:

    def test_hooks_carry_tier_budget(self, monkeypatch):
        import functools
        from sandbox_agent.config import BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG

        class _StubAssistant:
            def __init__(self, **kwargs):
                self.llm = type("L", (), {"generate_cfg": {}})()
                self._run = lambda *a, **k: iter(())
                self._call_tool = lambda *a, **k: ""

        monkeypatch.setattr(m, "Assistant", _StubAssistant)
        primary_agent = m.create_agent("sys", llm_cfg=PRIMARY_LLM_CFG)
        secondary_agent = m.create_agent("sys", llm_cfg=BACKGROUND_LLM_CFG)
        assert isinstance(primary_agent._precall_compact, functools.partial)
        # Both tiers budgeted at 200k after the 2026-08-05 rollback; the
        # binding mechanism is what's under test, not the numbers.
        assert (primary_agent._precall_compact.keywords["context_tokens"]
                == PRIMARY_LLM_CFG["context_window_tokens"])
        assert (secondary_agent._precall_compact.keywords["context_tokens"]
                == BACKGROUND_LLM_CFG["context_window_tokens"])
