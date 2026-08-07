"""Background work must survive one tier being down.

Before this, `run_on_best_available` was the ONLY LLM path with no cross-tier
failover — and it is the one that runs cron, heartbeat, pipeline stages and
stage evaluation. A hard failure propagated with no retry elsewhere, and the
silent-failure retry deliberately re-used the SAME tier ("same tier — slot
still held"), so it hit the same dead model twice while the other tier sat
idle. Pipeline stages have limited attempts, so an outage could burn a
strategy's attempts and reject it for infrastructure reasons.

Chat (LockingAgent) and llm_call (llm_client._resolve_chain) already failed
over; this brings background work in line, plus a per-tier health breaker so
a known-dead tier stops being tried first.
"""

import threading

import pytest

from qwen_agent.llm.schema import Message

import sandbox_agent.main as m
from sandbox_agent import model_health
from sandbox_agent.config import (PRIMARY_MODEL_CONCURRENCY,
                                  SECONDARY_MODEL_CONCURRENCY)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from threading import BoundedSemaphore
    model_health.reset()
    monkeypatch.setattr(m, "_primary_model_lock", BoundedSemaphore(PRIMARY_MODEL_CONCURRENCY))
    monkeypatch.setattr(m, "_secondary_model_lock", BoundedSemaphore(SECONDARY_MODEL_CONCURRENCY))
    yield
    model_health.reset()


def _reply(text="hello"):
    return [Message(role="assistant", content=text)]


@pytest.fixture
def tiers(monkeypatch):
    """Record which tiers ran; behavior driven per tier."""
    calls = []
    behavior = {}

    def fake_run(tier, system_message, messages, timeout, task_label):
        calls.append(tier)
        b = behavior.get(tier, _reply())
        if isinstance(b, Exception):
            raise b
        return b

    monkeypatch.setattr(m, "_run_on_tier", fake_run)
    monkeypatch.setattr(m, "compute_request_timeout", lambda msgs: 60)
    return calls, behavior


MSGS = [Message(role="user", content="do the thing")]


class TestFailover:

    def test_hard_failure_retries_on_the_other_tier(self, tiers):
        calls, behavior = tiers
        behavior["secondary"] = ConnectionError("linux model is down")
        out = m.run_on_best_available("sys", MSGS)
        assert calls == ["secondary", "primary"], f"tiers tried: {calls}"
        assert out == _reply()

    def test_hard_failure_no_longer_propagates(self, tiers):
        """Previously this raised and failed the cron/pipeline task outright."""
        _, behavior = tiers
        behavior["secondary"] = ConnectionError("down")
        m.run_on_best_available("sys", MSGS)  # must not raise

    def test_empty_response_retries_on_the_other_tier(self, tiers):
        calls, behavior = tiers
        behavior["secondary"] = []
        m.run_on_best_available("sys", MSGS)
        assert calls == ["secondary", "primary"], (
            "silent-failure retry re-used the dead tier")

    def test_both_tiers_down_raises_rather_than_hanging(self, tiers):
        _, behavior = tiers
        behavior["secondary"] = ConnectionError("down")
        behavior["primary"] = ConnectionError("also down")
        with pytest.raises(ConnectionError):
            m.run_on_best_available("sys", MSGS)

    def test_healthy_run_does_not_retry(self, tiers):
        calls, _ = tiers
        m.run_on_best_available("sys", MSGS)
        assert calls == ["secondary"], "retried a successful run"

    def test_slots_are_released_after_failover(self, tiers):
        _, behavior = tiers
        behavior["secondary"] = ConnectionError("down")
        m.run_on_best_available("sys", MSGS)
        grants = [m._acquire_turn_slot(blocking=False)
                  for _ in range(PRIMARY_MODEL_CONCURRENCY + SECONDARY_MODEL_CONCURRENCY)]
        assert all(g is not None for g in grants), "a slot leaked during failover"
        for _, rel in grants:
            rel()


class TestHealthAwareAcquisition:

    def test_failures_mark_a_tier_unhealthy(self, tiers):
        _, behavior = tiers
        behavior["secondary"] = ConnectionError("down")
        for _ in range(2):
            m.run_on_best_available("sys", MSGS)
        assert not model_health.is_healthy("secondary")
        assert model_health.is_healthy("primary")

    def test_unhealthy_tier_is_no_longer_tried_first(self, tiers):
        calls, behavior = tiers
        behavior["secondary"] = ConnectionError("down")
        for _ in range(2):
            m.run_on_best_available("sys", MSGS)
        calls.clear()
        m.run_on_best_available("sys", MSGS)
        assert calls[0] == "primary", (
            f"still tried the known-dead tier first: {calls}")

    def test_success_reopens_the_tier(self, tiers):
        _, behavior = tiers
        behavior["secondary"] = ConnectionError("down")
        for _ in range(2):
            m.run_on_best_available("sys", MSGS)
        assert not model_health.is_healthy("secondary")
        behavior["secondary"] = _reply()
        model_health.record_success("secondary")
        assert model_health.is_healthy("secondary")

    def test_cooldown_expiry_lets_it_be_probed_again(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr(model_health.time, "monotonic", lambda: clock["t"])
        from sandbox_agent.config import MODEL_HEALTH_COOLDOWN, MODEL_HEALTH_THRESHOLD
        for _ in range(MODEL_HEALTH_THRESHOLD):
            model_health.record_failure("secondary")
        assert not model_health.is_healthy("secondary")
        clock["t"] += MODEL_HEALTH_COOLDOWN + 1
        assert model_health.is_healthy("secondary")

    def test_all_tiers_unhealthy_still_routes(self):
        """Refusing to route would turn 'both flaky' into 'agent does nothing'."""
        from sandbox_agent.config import MODEL_HEALTH_THRESHOLD
        for tier in ("primary", "secondary"):
            for _ in range(MODEL_HEALTH_THRESHOLD):
                model_health.record_failure(tier)
        assert model_health.healthy_subset(["primary", "secondary"]) == \
               ["primary", "secondary"]
        assert m._acquire_turn_slot(blocking=False) is not None

    def test_unhealthy_tier_still_used_when_it_is_the_only_capacity(self):
        """Health reorders preference; it must not shrink capacity."""
        from sandbox_agent.config import MODEL_HEALTH_THRESHOLD
        for _ in range(MODEL_HEALTH_THRESHOLD):
            model_health.record_failure("secondary")
        held = [m._acquire_turn_slot(blocking=False, only="primary")
                for _ in range(PRIMARY_MODEL_CONCURRENCY)]
        assert all(h is not None for h in held)
        grant = m._acquire_turn_slot(blocking=False)
        assert grant is not None and grant[0] == "secondary", (
            "unhealthy tier was excluded entirely, losing real capacity")
        grant[1]()
        for _, rel in held:
            rel()

    def test_pinned_turn_ignores_health(self):
        """Pinning is a correctness constraint; health is only a hint."""
        from sandbox_agent.config import MODEL_HEALTH_THRESHOLD
        for _ in range(MODEL_HEALTH_THRESHOLD):
            model_health.record_failure("primary")
        grant = m._acquire_turn_slot(blocking=False, only="primary")
        assert grant is not None and grant[0] == "primary"
        grant[1]()
