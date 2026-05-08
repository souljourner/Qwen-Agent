# Stage 5: Paper Trading Scaffold

## Objective
Scaffold a deployable paper-trading setup. This stage does NOT place live orders — the container has no broker credentials. The goal is to produce runnable code + a clear README the user can run on their own machine with their own API keys.

## Runs only if Stage 4 promoted
If `pipeline/verdict.md` has "Final Recommendation: reject", this stage is **skipped** automatically by `stage_runner._check_verdict_skip`. You will not be invoked in that case. If you are invoked, the verdict was `promote` — proceed.

## Instructions

1. Read `backtest/full/strategy.py` and `strategy/hypothesis_v{final}.md` to understand exactly what needs to run in production. Also read `pipeline/verdict.md`'s `## If Promoted` section for the specific scaffold requirements the verdict stage specified.
2. Pick a paper-trading broker integration. Good defaults:
   - **Alpaca** (`alpaca-trade-api` or `alpaca-py`) — free paper trading, US equities + crypto
   - **CCXT** with Binance testnet — crypto paper trading
   - **Interactive Brokers paper gateway** — more complex, skip unless user mentioned it
3. Write `paper/deploy.py` — a runnable script that:
   - Loads broker credentials from environment variables (`ALPACA_API_KEY`, `ALPACA_SECRET`) — NEVER hardcoded
   - Fetches recent data via the broker's market data API or yfinance
   - Computes the signal using the exact same logic as the backtest (import from `backtest/strategy.py` where possible)
   - Places paper-trading orders via the broker SDK
   - Logs each decision (timestamp, ticker, action, size, rationale) to `paper/trades.jsonl`
   - Implements a **kill switch** — aborts all trading if daily loss exceeds the spec's drawdown circuit breaker or if connectivity fails
4. Validate the file compiles: `exec(command="python -m py_compile paper/deploy.py", project="{PROJECT_NAME}")`
5. Write `paper/README.md` with these sections:
   - ## Broker Integration (which broker, which SDK, which environment)
   - ## Monitoring (what log files are written, how to tail them, what alerts to set up)
   - ## Kill Switch (under what conditions the script halts, how to manually halt it)
   - ## How to Run (set env vars, install from requirements, run the script, cron it)

## CRITICAL
- Do NOT try to place live or paper orders from inside the container — no credentials are available.
- Do NOT commit any API keys or secrets.
- The acceptance evaluator will run `python -m py_compile paper/deploy.py` to verify the code is syntactically valid.

## If Previous Output Exists
Debug or tighten the existing scaffold — better error handling, more defensive kill switch conditions, clearer logging. Don't rewrite from scratch.

## Output Files
- `paper/deploy.py` — runnable paper-trading script (syntactically valid Python)
- `paper/README.md` — sections: Broker Integration, Monitoring, Kill Switch, How to Run

## Tools to Use
- `project_read_file` for spec and backtest code
- `project_write_file` (append) for building the README
- `exec` for `python -m py_compile` verification

## Quality Bar
- deploy.py parses as Python (py_compile succeeds)
- Credentials loaded from env vars only — grep for any hardcoded keys
- Kill switch logic is present and triggers on at least one concrete condition
- README's "How to Run" is complete enough that a new user could get it running
