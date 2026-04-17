# Stage 2: Strategy Specification

## Objective
Translate the research into a precise, implementable strategy specification. The spec must be detailed enough that stage 4 can code the backtest directly from it.

## Instructions

1. Read `research/strategy-research.md` and synthesize the hypothesis into a concrete algorithm.
2. Specify entry rules with exact conditions — indicator values, thresholds, timeframes. No vague language like "when momentum is strong"; use "when 20-day return > 5% and 50-day RSI > 60."
3. Specify exit rules — profit targets, stop losses, time-based exits, signal reversal.
4. Specify position sizing — fixed fraction, volatility-targeted, Kelly fraction, etc. Include the exact formula.
5. Specify risk limits — max position size, max portfolio heat, max drawdown circuit breaker.
6. Define rebalance cadence — daily close, weekly, on-signal-only.
7. State the expected performance envelope: target annualized return, Sharpe, max drawdown, win rate. These are hypotheses the backtest will test — be honest (a 5-Sharpe equity strategy is a red flag, not a target).

## If Previous Output Exists
Read `strategy/spec.md`. Tighten ambiguous rules, add missing specifics, preserve anything that was already precise.

## Output Format
Write to `strategy/spec.md` with these sections:
- ## Entry Rules (precise conditions, with indicator parameters)
- ## Exit Rules (stop loss, profit target, time exits, signal reversal)
- ## Position Sizing (exact formula, with example numbers)
- ## Risk Limits (position cap, heat cap, drawdown circuit breaker)
- ## Expected Performance (target Sharpe, max drawdown, CAGR, win rate — as hypotheses to test)

## Writing Strategy
Use `project_write_file(mode='append', ...)` section by section. Don't try to write the whole spec in one call.

## Tools to Use
- project_read_file to load `research/strategy-research.md`
- project_write_file (append for each section; edit for refinements)

## Quality Bar
- Entry and Exit rules are unambiguous — a programmer could implement them without asking questions
- Position sizing includes a concrete formula (e.g., `size = 0.01 * equity / (ATR * multiplier)`)
- Risk limits include specific percentages, not "reasonable"
- Expected Performance targets are realistic for the strategy class (momentum: 0.7–1.5 Sharpe; mean reversion: 0.5–1.2 Sharpe; stat arb: 1.5–3 Sharpe after costs)
