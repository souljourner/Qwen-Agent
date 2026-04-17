# Stage 4: Backtest

## Objective
Implement the strategy in code and run a real backtest. Report actual numerical performance metrics.

## Instructions

1. **First reinstall dependencies** — pip installs from stage 3 don't survive container restarts:
   `exec(command="pip install -r data/requirements.txt", project="{PROJECT_NAME}")`
2. Read `strategy/spec.md` and `data/README.md`.
3. Write `backtest/strategy.py` — a Python script that:
   - Loads the processed data from `data/processed/` (produced by stage 3's pipeline)
   - Implements the entry/exit rules exactly as specified in the spec
   - Applies position sizing and risk limits as specified
   - Runs the backtest over ≥5 years of data (or however much stage 3 fetched)
   - Computes performance metrics:
     - **Sharpe ratio** (annualized, assuming 252 trading days, risk-free rate ~4%)
     - **Max Drawdown** (peak-to-trough %)
     - **CAGR** (compound annual growth rate)
     - **Win Rate** (% of profitable trades)
     - Average trade duration, number of trades, profit factor
   - Prints the metrics to stdout in a clear format
   - Saves an equity curve as `backtest/equity_curve.csv`
4. Run the backtest: `exec(command="python backtest/strategy.py", project="{PROJECT_NAME}", timeout=300)`
5. Optionally: do a simple walk-forward analysis — split data into in-sample and out-of-sample periods, report metrics for each.
6. Write `backtest/results.md` with these sections and **actual numerical values**:
   - ## Sharpe (decimal number, e.g., `0.87`)
   - ## Max Drawdown (percentage, e.g., `-18.4%`)
   - ## CAGR (percentage, e.g., `14.2%`)
   - ## Win Rate (percentage, e.g., `42.8%`)
   - ## Trade Count and Average Duration
   - ## Walk-Forward Results (if computed)
   - ## Interpretation (does this support or refute the alpha hypothesis?)

## ANTI-PLACEHOLDER RULE
The acceptance evaluator will **reject** results.md if any metric is a placeholder: `TBD`, `N/A`, `X.X`, `???`, `Pending`, or a description without a value. You MUST run the backtest and paste real numbers.

## If Previous Output Exists
If `backtest/strategy.py` and `backtest/results.md` exist, debug and iterate — find why the strategy underperforms, tighten the implementation, refit obviously-broken parameters. Do NOT rewrite the strategy from scratch unless spec changed.

## Output Files
- `backtest/strategy.py` — the backtest implementation
- `backtest/results.md` — real numerical performance metrics
- `backtest/equity_curve.csv` — time series of portfolio equity (optional but recommended)

## Tools to Use
- `exec` for `pip install -r` and running the backtest
- `code_interpreter` for small sanity checks
- `project_read_file` to load the spec and data README
- `project_write_file` (append) to build results.md section by section after running

## Quality Bar
- backtest/strategy.py runs without errors via `exec`
- All four required metrics (Sharpe, Max Drawdown, CAGR, Win Rate) are present as real numbers
- Results include actual trade counts, not simulated ones
- Interpretation section honestly states whether results are acceptable or the hypothesis should be revised
