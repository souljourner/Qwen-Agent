"""Per-tier circuit breaker for model availability.

Turn slots are BoundedSemaphores — pure local counters with no idea whether
the model behind them is reachable. So a tier that is completely down still
hands out slots, and every caller discovers the outage the hard way: full
request timeout, then failure. With `run_on_best_available` now failing over
across tiers, a dead tier costs every background task a wasted attempt
BEFORE it reaches the healthy one.

This closes that: after a tier fails repeatedly, stop preferring it for a
cooldown, so work goes straight to the tier that works.

Deliberately soft, never a hard block:

  * If EVERY tier is unhealthy we treat them all as healthy again. Refusing
    to route anywhere would turn "both models flaky" into "the agent does
    nothing", which is strictly worse than trying.
  * A pinned turn (`only=`) always gets its tier. Pinning is a correctness
    constraint (the history only fits there); health is an optimization.
  * The breaker only reorders PREFERENCE. Slots, caps and blocking semantics
    are untouched — an unhealthy tier is still used when it is the only one
    with capacity.

Recovery is automatic: the cooldown expires, the next request probes the
tier, and a success closes the breaker.
"""

import logging
import threading
import time

from sandbox_agent.config import (MODEL_HEALTH_COOLDOWN, MODEL_HEALTH_THRESHOLD)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_failures = {}     # tier -> consecutive failure count
_down_until = {}   # tier -> monotonic deadline


def record_failure(tier: str) -> None:
    """A request to `tier` failed outright (exception, or produced nothing)."""
    if not tier:
        return
    with _lock:
        n = _failures.get(tier, 0) + 1
        _failures[tier] = n
        if n >= MODEL_HEALTH_THRESHOLD and tier not in _down_until:
            _down_until[tier] = time.monotonic() + MODEL_HEALTH_COOLDOWN
            logger.error("Tier %s failed %d consecutive times — deprioritizing it "
                         "for %ds. It is still used if it is the only tier with "
                         "capacity, or for pinned turns.",
                         tier, n, MODEL_HEALTH_COOLDOWN)


def record_success(tier: str) -> None:
    if not tier:
        return
    with _lock:
        if _failures.get(tier) or tier in _down_until:
            logger.info("Tier %s recovered — breaker closed", tier)
        _failures.pop(tier, None)
        _down_until.pop(tier, None)


def is_healthy(tier: str) -> bool:
    with _lock:
        return _is_healthy_locked(tier)


def _is_healthy_locked(tier: str) -> bool:
    deadline = _down_until.get(tier)
    if deadline is None:
        return True
    if time.monotonic() >= deadline:
        # Cooldown elapsed — let the next request probe it.
        _down_until.pop(tier, None)
        _failures.pop(tier, None)
        return True
    return False


def healthy_subset(tiers):
    """Filter `tiers` to the healthy ones, or return them ALL if none is.

    Returning everything when all tiers are down is the important part: the
    caller must always have somewhere to go.
    """
    tiers = list(tiers)
    with _lock:
        healthy = [t for t in tiers if _is_healthy_locked(t)]
    return healthy or tiers


def reset() -> None:
    """Test hook."""
    with _lock:
        _failures.clear()
        _down_until.clear()


def status() -> dict:
    with _lock:
        now = time.monotonic()
        return {t: {"failures": _failures.get(t, 0),
                    "down_for_s": max(0, int(d - now))}
                for t, d in list(_down_until.items())} or {"all_tiers": "healthy"}
