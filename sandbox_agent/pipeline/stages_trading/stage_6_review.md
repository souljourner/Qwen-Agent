# Stage 6: Review

## Objective
Synthesize all prior stages into a deployment-readiness assessment. Be honest about robustness and shortcomings.

## Runs only if Stage 4 promoted
If `pipeline/verdict.md` has "Final Recommendation: reject", this stage is **skipped** automatically by `stage_runner._check_verdict_skip`. If invoked, the verdict was `promote`.

## Instructions

1. Read every prior artifact:
   - `research/data-landscape.md`
   - `strategy/loop_state.json` (full history — pilot_history, run_notes, hypothesis_notes)
   - `strategy/hypothesis_v{final}.md`
   - `backtest/full/results.md` and `backtest/full/metrics.json`
   - `pipeline/verdict.md`
   - `paper/README.md`
2. Write `pipeline/review.md` with these sections:
   - ## Performance Summary (Sharpe, Max Drawdown, CAGR, Win Rate from `backtest/full/metrics.json` — restate the numbers; include pilot vs. OOS and walk-forward win rate)
   - ## Research Trajectory (how many hypotheses were tried, how many iterations per hypothesis, what hypothesis_notes say about abandoned ones — the story of what this research *learned*)
   - ## Robustness (regime sensitivity, walk-forward degradation, known failure modes from `run_notes`' "what_seemed_off" entries — did the backtest expose any?)
   - ## Deployment Readiness (is this strategy ready for paper trading? For live trading with real money? What risks should the user be aware of?)
   - ## Learnings (what worked, what didn't, what would you change)
   - ## Recommended Next Steps (parameter tuning, regime filters, additional universe diversification, risk overlays)

Do NOT write to `status.md`, `pipeline/state.json`, or any pipeline-state file — pipeline status is tracked automatically by the orchestrator, and writes to those files are rejected.

## If Previous Output Exists
Tighten the assessment with newer data. Preserve honest criticism — don't soften weaknesses just because you're rerunning.

## Output Files
- `pipeline/review.md` — full assessment

## Tools to Use
- `project_read_file` for loading all prior artifacts
- `project_write_file` (append) for building the review

## Quality Bar
- Performance numbers match the backtest results (don't inflate)
- Robustness section names at least 2 concrete risks or weaknesses
- Deployment Readiness gives a clear yes/no/conditional recommendation
- Honest assessment — if Sharpe is 0.2 and max drawdown is -40%, say so; don't frame a weak result as strong
