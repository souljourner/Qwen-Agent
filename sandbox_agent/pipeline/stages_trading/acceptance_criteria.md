# Trading Pipeline Acceptance Criteria

You are evaluating the output of a trading pipeline stage. Judge whether the artifact is acceptable quality to proceed.

## Evaluation Criteria

1. **Completeness**: Does the artifact cover all required sections?
2. **Specificity**: Are rules, numbers, and parameters concrete? Vague research is OK in stage 1, but stages 2–5 must be precise.
3. **Runnability**: For code artifacts, can the code be executed? (Stage 3 pipeline.py must run; stage 4 strategy.py must produce metrics; stage 5 deploy.py must parse as Python.)
4. **Honest Metrics**: Stage 4 must contain real numerical performance metrics from an actual backtest, not placeholders.
5. **Actionability**: Could a trader use this to make decisions?

## Stage-Specific Rules

### Stage 4 (Backtest) — ANTI-PLACEHOLDER RULE
The `backtest/results.md` MUST contain actual decimal numbers for Sharpe, Max Drawdown, CAGR, and Win Rate. **Reject** outputs where any required metric is:
- `TBD`, `N/A`, `X.X`, `X.XX`, `???`, `Pending`, `To be computed`, `Coming soon`
- A description of the metric without a value (e.g., "Sharpe ratio: to be calculated from the equity curve")
- A placeholder formula without a result (e.g., "Sharpe = annualized_return / annualized_vol")

Real values look like: `Sharpe: 0.87`, `Max Drawdown: -18.4%`, `CAGR: 14.2%`, `Win Rate: 42.8%`.

### Stage 5 (Paper Trading) — SYNTAX RULE
`paper/deploy.py` must be syntactically valid Python. The evaluator verifies this by running `python -m py_compile`. If the file doesn't parse, fail with specific error location.

## Scoring

- **PASS**: The artifact covers required sections with reasonable depth and any stage-specific rules are met. Minor ambiguities, imperfect formatting, or gaps in non-critical subsections are acceptable — this is an automated draft.
- **FAIL**: Fail only for CRITICAL issues:
  - Missing entire required sections
  - Stage 4 contains placeholder metrics instead of real numbers
  - Stage 5 code doesn't parse
  - Document is truncated mid-sentence or clearly incomplete
  - Content is factually wrong in a trading-critical way (e.g., confusing long and short, inverting risk rules, claiming a 10-Sharpe strategy based on in-sample overfitting without noting it)

Do NOT fail for:
- Modest but honest backtest results (a 0.3 Sharpe is a PASS — the pipeline ran, the number is real; honesty beats fake performance)
- Placeholder ticker names if the selection rule is clearly defined
- Imperfect formatting or minor rule gaps
- Lack of walk-forward analysis (optional in stage 4)

## Response Format

Respond with exactly one line starting with PASS or FAIL, followed by reasoning:

PASS: [Brief reason]

or

FAIL: [Specific critical issue — placeholder metrics, syntax error in code, missing required section, etc.]
