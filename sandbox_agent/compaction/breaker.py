"""Circuit breaker for the compaction summarizer.

Compaction is expensive and all-or-nothing. When the summarizer is down or
too slow, EVERY tier-3 attempt costs the full timeout budget — and because
nothing gets compacted, the next turn starts from the same oversized
history and pays it again. Observed live 2026-08-06: turns burned ~2
minutes each, forever, and the user saw "it seems to continuously compact".

Raising the timeout (which was necessary — see COMPACTION_TIMEOUT) makes
that failure mode WORSE, not better: at 600s x 2 attempts x 4 chunks a
single turn would stall for over an hour. The timeout fixes the healthy
path; this fixes the unhealthy one.

So: after a couple of consecutive failures, stop calling for a cooldown.
Compaction then fails FAST instead of slowly. The conversation is still
safe while the breaker is open — the deterministic tiers (pointers, tool
results, narration) still run, and qwen_agent's own max_input_tokens
backstop still bounds the request. We degrade to "not compacting" rather
than to "unusable chat", and say so in the log.

State is process-global, matching the failure it models: the summarizer
endpoint is shared by every thread, so one thread discovering it is down
should spare the others the same discovery.
"""

import logging
import threading
import time

from sandbox_agent.config import (COMPACTION_BREAKER_COOLDOWN,
                                  COMPACTION_BREAKER_THRESHOLD)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_consecutive_failures = 0
_open_until = 0.0


def is_open() -> bool:
    """True if the summarizer should NOT be called right now."""
    with _lock:
        if _open_until and time.monotonic() < _open_until:
            return True
        return False


def record_success() -> None:
    global _consecutive_failures, _open_until
    with _lock:
        if _consecutive_failures or _open_until:
            logger.info("Compaction summarizer recovered — breaker closed")
        _consecutive_failures = 0
        _open_until = 0.0


def record_failure() -> None:
    global _consecutive_failures, _open_until
    with _lock:
        _consecutive_failures += 1
        if _consecutive_failures >= COMPACTION_BREAKER_THRESHOLD:
            _open_until = time.monotonic() + COMPACTION_BREAKER_COOLDOWN
            logger.error(
                "Compaction summarizer failed %d consecutive times — pausing "
                "summarization for %ds. Histories will NOT be summarized until "
                "then; deterministic compaction tiers still apply.",
                _consecutive_failures, COMPACTION_BREAKER_COOLDOWN)


def reset() -> None:
    """Test hook — clear all breaker state."""
    global _consecutive_failures, _open_until
    with _lock:
        _consecutive_failures = 0
        _open_until = 0.0


def status() -> dict:
    with _lock:
        remaining = max(0.0, _open_until - time.monotonic()) if _open_until else 0.0
        return {"open": remaining > 0, "consecutive_failures": _consecutive_failures,
                "cooldown_remaining_s": int(remaining)}
