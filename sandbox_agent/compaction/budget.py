"""Compaction budget derivation.

The layering used to be INVERTED: compaction aimed at 170k while
qwen_agent's own `max_input_tokens` backstop sat below it at 160k. Since
that backstop deletes OLDEST-first, and a persisted digest lives at
position 0, the digest was the first thing destroyed (verified empirically
2026-08-06: a [digest + 40 turns] history truncated at 160k came back with
the digest gone and the context starting mid-conversation).

Correct ordering, enforced by test:

    target  <  trigger  <  hard  <=  max_input_tokens  <  real window

Compaction always lands well under the backstop, so the backstop only ever
fires when compaction failed outright — never on healthy output.
"""

from dataclasses import dataclass

from sandbox_agent.config import (
    COMPACTION_HARD_RATIO,
    COMPACTION_RESERVE_TOKENS,
    COMPACTION_TARGET_RATIO,
    COMPACTION_TRIGGER_RATIO,
    MAX_CONTEXT_TOKENS,
)


@dataclass(frozen=True)
class Budgets:
    """Token thresholds for one tier."""
    hard: int      # never exceed — compaction MUST get below this
    trigger: int   # start compacting above this
    target: int    # compact down to this (hysteresis: well below trigger)

    def __post_init__(self):
        assert self.target < self.trigger <= self.hard, (
            f"budget ordering violated: target={self.target} "
            f"trigger={self.trigger} hard={self.hard}")


def derive_budgets(context_tokens: int = None, max_input_tokens: int = None,
                   trigger_fraction_of_context: float = None) -> Budgets:
    """Thresholds for a tier, clamped under its real input ceiling.

    `max_input_tokens` is the model cfg's truncation backstop; the ceiling is
    whichever of (window - reserve) and that backstop binds first.

    `trigger_fraction_of_context` expresses "start compacting at N% of the
    real window" directly, instead of the default chain (a fraction of
    `hard`, which is itself a fraction of the ceiling — ~76% of the window,
    not a round number anyone asked for). Clamped to `hard` so the ordering
    invariant cannot be violated by a careless fraction.
    """
    ctx = context_tokens or MAX_CONTEXT_TOKENS
    ceiling = ctx - COMPACTION_RESERVE_TOKENS
    if max_input_tokens:
        ceiling = min(ceiling, max_input_tokens)
    ceiling = max(ceiling, 20_000)  # degenerate configs still get a sane floor
    hard = int(ceiling * COMPACTION_HARD_RATIO)
    if trigger_fraction_of_context:
        trigger = min(int(ctx * trigger_fraction_of_context), hard)
    else:
        trigger = int(hard * COMPACTION_TRIGGER_RATIO)
    target = int(hard * COMPACTION_TARGET_RATIO)
    if target >= trigger:
        # Keep hysteresis meaningful when the trigger is pulled down by the
        # clamp above: compacting to ~the trigger would re-fire immediately.
        target = int(trigger * 0.65)
    return Budgets(hard=hard, trigger=trigger, target=target)
