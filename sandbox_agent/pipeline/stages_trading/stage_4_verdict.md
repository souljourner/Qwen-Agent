# Stage 4: Verdict

## Objective
Write the final promote/reject recommendation. This stage is the pipeline's graceful exit — it **always runs**, whether Stage 3 passed or failed, and its decision is programmatically gated against `backtest/full/metrics.json`.

## The one rule
The "Final Recommendation" field in `pipeline/verdict.md` **must match** the pass/fail status of `backtest/full/metrics.json` against the Stage 3 gates. The evaluator re-runs those gates itself and rejects the verdict if the LLM's recommendation disagrees with the numbers.

| Stage 3 gates | Required recommendation |
|---|---|
| All pass | `promote` |
| Any fail | `reject` |

A verdict narrative that says "reject" while metrics pass all gates will be rejected, as will the reverse.

## Instructions
1. Read `backtest/full/metrics.json`.
2. Read `backtest/full/results.md`.
3. Read `strategy/loop_state.json` (the full hypothesis_notes + pilot_history history).
4. Read `strategy/hypothesis_v{final}.md`.
5. Write `pipeline/verdict.md` with the required sections below.
6. Write `pipeline/metrics.json` — a copy of `backtest/full/metrics.json` alongside the decision for downstream consumers.

## Required sections in verdict.md
- `## Final Recommendation` — one word on its own line: `promote` or `reject`.
- `## Rationale` — the "why" in plain English, grounded in the specific gates that passed/failed. Cite numbers.
- `## Strategy Summary` — one-paragraph recap of the converged hypothesis, universe, and signal mechanics.
- `## Research History` — one sentence per hypothesis attempted (from hypothesis_notes), plus final iteration count.
- `## Caveats` — known limitations: universe scope, time period, data-source constraints, deflated-Sharpe note. **REQUIRED subsection: `### Survivorship bias`** — the research loop currently skips PREM14A / DEFM14A filings whose tickers have no available price history (delisted / acquired / not on yfinance). This means the pilot and OOS universes over-represent survivors; the pipeline **does not currently support neutralization of this bias**. Cite the count of skipped filings from `strategy/loop_state.json.skipped_filings`.
- `## If Promoted` — what Stage 5 should scaffold (specific order types, risk limits, position sizing rules).
- `## If Rejected` — the single most load-bearing reason the strategy failed, plus what the research surfaced as a genuinely useful byproduct (if any).

Both "If Promoted" and "If Rejected" are written regardless of the recommendation — one will be used, the other archives the counterfactual.

## Pipeline behavior
- `promote` → Stages 5 (paper trading scaffold) and 6 (review) run normally.
- `reject` → Stages 5 and 6 are **skipped** with a no-op; pipeline status becomes `completed_rejected`.

This is enforced in `orchestrator._schedule_next_stage` and `stage_runner._check_verdict_skip` — the verdict file is authoritative.

## Honesty over hope
A rejected research project is a successful research project. The user explicitly wants to know when a strategy does **not** work and why, so they don't waste capital on it. Do not soften `reject` to "promote with caveats" to avoid disappointment — the gates exist to prevent that.

## CRITICAL RULES
- The first word on the line under `## Final Recommendation` MUST be exactly `promote` or `reject` (lowercase).
- Cite actual numbers from metrics.json in the Rationale section; no hand-waving.
- Do not rerun Stage 3's analysis — if you think the metrics are wrong, that's a data/code bug to flag in Caveats, not a justification to override the recommendation.
