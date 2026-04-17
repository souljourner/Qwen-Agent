# Stage 6: Review

## Objective
Synthesize all prior stages into a deployment-readiness assessment. Be honest about robustness and shortcomings.

## Instructions

1. Read every prior artifact:
   - `research/strategy-research.md`
   - `strategy/spec.md`
   - `data/README.md`
   - `backtest/results.md`
   - `paper/README.md`
2. Write `pipeline/review.md` with these sections:
   - ## Performance Summary (Sharpe, Max Drawdown, CAGR, Win Rate from backtest — restate the numbers)
   - ## Robustness (regime sensitivity, parameter sensitivity, walk-forward degradation, known failure modes from stage 1 — did the backtest expose any?)
   - ## Deployment Readiness (is this strategy ready for paper trading? For live trading with real money? What risks should the user be aware of?)
   - ## Learnings (what worked, what didn't, what would you change)
   - ## Recommended Next Steps (parameter tuning, regime filters, additional universe diversification, risk overlays)
3. Update `status.md` with the final pipeline status.

## If Previous Output Exists
Tighten the assessment with newer data. Preserve honest criticism — don't soften weaknesses just because you're rerunning.

## Output Files
- `pipeline/review.md` — full assessment
- `status.md` — final pipeline status (auto-generated mostly, but can be enriched)

## Tools to Use
- `project_read_file` for loading all prior artifacts
- `project_write_file` (append) for building the review

## Quality Bar
- Performance numbers match the backtest results (don't inflate)
- Robustness section names at least 2 concrete risks or weaknesses
- Deployment Readiness gives a clear yes/no/conditional recommendation
- Honest assessment — if Sharpe is 0.2 and max drawdown is -40%, say so; don't frame a weak result as strong
