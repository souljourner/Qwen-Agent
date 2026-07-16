"""Damper for notification-driven agent turns.

Two protections against the 2026-07-16 runaway loop (a failing pipeline stage
rescheduled itself repeatedly; every failure notification spawned a synthetic
agent turn; the agent reacted — sometimes destructively — which caused the
next failure; the chat Stop button only cancels the in-flight turn, so the
loop was unstoppable from the UI):

1. Per-task cooldown — repeated notifications from the SAME task within
   REACT_COOLDOWN_S get delivered to the UI and appended to history, but do
   NOT trigger a synthetic agent turn. The agent still sees them on its next
   real turn.
2. Stop-button mute — pressing Stop mutes ALL synthetic turns for
   STOP_MUTE_S, so the user can actually halt a notification storm.
"""

import time
from typing import Dict, Optional

REACT_COOLDOWN_S = 900   # one synthetic reaction per task per 15 min
STOP_MUTE_S = 600        # Stop button silences synthetic turns for 10 min

_last_reaction: Dict[str, float] = {}
_muted_until: float = 0.0


def should_trigger_synthetic_turn(task_name: str, now: Optional[float] = None) -> bool:
    """True when a notification for `task_name` may spawn an agent turn."""
    t = time.monotonic() if now is None else now
    if t < _muted_until:
        return False
    last = _last_reaction.get(task_name)
    if last is not None and (t - last) < REACT_COOLDOWN_S:
        return False
    _last_reaction[task_name] = t
    return True


def mute_on_stop(now: Optional[float] = None) -> None:
    """User pressed Stop — silence synthetic turns for STOP_MUTE_S."""
    global _muted_until
    t = time.monotonic() if now is None else now
    _muted_until = t + STOP_MUTE_S


def reset() -> None:
    """Test hook."""
    global _muted_until
    _last_reaction.clear()
    _muted_until = 0.0
