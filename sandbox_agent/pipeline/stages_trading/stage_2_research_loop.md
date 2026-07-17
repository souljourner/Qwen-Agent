# Stage 2: Research Loop

## Objective
Either **converge on a strategy that performs on pilot data**, or **conclude no such strategy exists in this problem space**. This stage iterates across many runs — each invocation advances the loop by exactly one step.

## Core rule — one invocation = one step (NOT one iteration)
You get a single invocation per scheduler tick: you write code, trigger `exec`, see its stdout, and the run ends. You cannot read a backtest's metrics and respond to them in the same run — that's the next invocation's job. So every full iteration (hypothesis → data → features → backtest → decide) takes several invocations, with each one emitting `part-completion` when it finishes its step.

## How to decide what to do this run
1. Read `## Loop State` above. It tells you **`next_step`** — that's what you do, nothing else.
2. Read the **`run_notes`** (last 3) and **`hypothesis_notes`** (all) in Loop State. They record what the previous runs tried, what worked, what seemed off, and (for abandoned hypotheses) generalizable lessons.
3. Execute that one step. Write a new `run_note` describing what you did, what worked, what seemed off, and your suggestion for the next step.
4. Append your `run_note` to `strategy/loop_state.json` under `run_notes`. Do NOT manually set `next_step` or `current_phase` — the evaluator writes those back after acceptance.
5. End your response with `part-completion`.

## The `next_step` state machine

| `next_step` | What you do |
|---|---|
| `init` | Read `research/data-landscape.md`. Write `strategy/hypothesis_v1.md` (with `required_data_types` declared — must reference names from the landscape). Write `strategy/universe_v1.json` (list of tickers, or filter rules). Initialize `strategy/loop_state.json` with `oos_cutoff_date` frozen, `hypothesis_count=1`, empty `pilot_history`. |
| `fetch_data` | Write a Python script that fetches all data needed for the current hypothesis and partitions it at `oos_cutoff_date`: pilot rows → `data/processed/pilot/`, oos rows → `data/processed/oos/`. Every row must have a `date` column. **US equity/ETF prices MUST come from the local data store — `from sandbox_agent.trading_data import get_daily, get_universe` (split/dividend-adjusted, reproducible across iterations; raises `DataUnavailable` for symbols not in the store → backfill once via `request_backfill`, retry, then record under `skipped_filings` and move on — see Data availability below). yfinance is BANNED for US equities/ETFs; it remains acceptable only for data the store lacks (futures, COT, FX, indices like ^VIX).** One-time setup per project: `uv pip install pandas requests numpy` via `exec(project=...)`. Call the script via `exec` (≤ 600s). |
| `extract_features` | Write a script that loads `data/processed/pilot/`, extracts plaintext if not already extracted, calls `llm_batch()` (from `sandbox_agent.tools.llm_client`) with a **static system prompt** and a **variable user prompt per item**, and caches results to `data/processed/llm_cache/pilot/{hash}.json`. Never feed HTML/XBRL/PDF/XML to the LLM — extract plaintext first. |
| `run_pilot_backtest` | Write `backtest/pilot/strategy_v{N}.py` that loads `data/processed/pilot/` and `data/processed/llm_cache/pilot/` and emits trade signals. **The strategy code MUST reference `data/processed/llm_cache/pilot/` — the evaluator enforces this as positive lineage.** **All fills/equity/metrics MUST go through the vetted accounting module — `from sandbox_agent.trading_accounting import Ledger, compute_metrics`: `led.fill(...)` per trade, `led.mark(ts, prices)` per bar, then `compute_metrics(led)` IS your metrics dict (write it as `metrics_v{N}.json` / `metrics_latest.json`). Hand-rolled equity curves / return math fail acceptance (evaluator lineage gate + numeric-sanity quarantine). The Ledger halts at blowup (equity ≤ 0) — that's a legitimate failed-hypothesis result; report it, don't work around it.** Run the backtest via `exec` and write `backtest/pilot/results_v{N}.md` + `backtest/pilot/metrics_v{N}.json` + `backtest/pilot/metrics_latest.json`. Append one row to `pilot_history` in loop_state.json with `{hyp, iter, sharpe, trades, return_pct, dd_pct, failure_mode}` taken from `compute_metrics` output. Append one `run_note`. |
| `revise_hypothesis` | Read the last `run_note` + existing `hypothesis_notes`. Write `strategy/hypothesis_v{N+1}.md` with `required_data_types`. Append one `hypothesis_notes` entry summarizing WHY the prior hypothesis was abandoned and what generalizable lessons carry forward. Bump `hypothesis_count`. |
| `revise_data_processing` | Keep the hypothesis. Sharpen the `extract_features` prompt or parser (e.g., add a short-side classification example; switch the parser to one that preserves tables). Delete or mark stale cache entries that used the old prompt. |
| `revise_strategy_code` | Keep the hypothesis AND the cache. Fix the strategy code — wrong signal threshold, bad position sizing, look-ahead bug, OOS leak, etc. |
| `extend_pilot_window` | Append more chronological data to `data/processed/pilot/`. Every row's date must be **strictly less than `oos_cutoff_date`** — the cutoff is frozen; never rewrite it. |

## Never cross the OOS boundary
- `oos_cutoff_date` in `loop_state.json` is **frozen** on the first fetch. You must never rewrite it.
- No pilot/strategy Python file may reference `oos/` or `data/processed/oos/`. The evaluator scans for this.
- `extend_pilot_window` only adds rows where `date < oos_cutoff_date`.

## Data availability — skip, never fabricate
When a PREM14A / DEFM14A filing's ticker has no real price history available (delisted, acquired, never fetched successfully, yfinance returns nothing), **SKIP THAT FILING**. Move on to the next available filing in the universe. Do **not** fabricate synthetic / flat / random prices to "fill the gap" — that produces meaningless Sharpe values and will be rejected by the evaluator (phrase blacklist catches "synthetic", "mock", "fake", "simulated"; the lineage check catches `hashlib`, hardcoded returns, etc.).

Concretely, inside `fetch_data` and `extract_features`:

1. For every ticker in the universe, attempt a real-price fetch — `sandbox_agent.trading_data.get_daily(ticker)` (the local store; catches `DataUnavailable` for missing symbols). yfinance only for non-US-equity data the store lacks.
2. On `DataUnavailable` for a US equity/ETF, backfill it ONCE: `request_backfill(ticker)` (same import — pulls the symbol from EODHD into the store and blocks until the bars are queryable), then retry `get_daily(ticker)`. Record every attempt in `loop_state.json` under `backfills_requested` (list of tickers). **Cap: 25 backfills per run** — once reached, record further missing tickers under `backfill_capped` and skip them without backfilling. Never call `request_backfill` twice for the same ticker in a run.
3. If the fetch still returns empty or fails (delisted, EODHD lacks it, cap reached), record the filing under `skipped_filings` in `loop_state.json` with shape:
   ```json
   {"ticker": "TWTR", "filing_date": "2022-05-17", "reason": "delisted_no_price_data"}
   ```
4. Drop that filing from `data/processed/pilot/` (and from `universe_v{N}.json` for subsequent iterations). Do not write it into `llm_cache/pilot/`.
5. Continue with the filings that DO have real prices. Every `run_note` must report `backfills_requested` and `backfill_capped` counts alongside `skipped_filings`.

If, after skipping, the pilot universe drops below the `trades >= 30` gate, the loop should either `extend_pilot_window` (fetch more filings) or — if the underlying data simply isn't there — report `insufficient_signal` and let the loop terminate into the verdict stage.

### Survivorship bias caveat (mandatory)
Skipping delisted/acquired tickers means the pilot universe over-represents survivors. This is a **known limitation** that we currently cannot neutralize — there is no symmetric source of historical prices for delisted tickers in this pipeline. Every `run_note` that skips filings must mention this, and the verdict stage will carry the caveat through to the final report.

## Writing the `run_note`
After finishing your step, append this shape to `loop_state.json` under `run_notes`:

```json
{
  "id": "r{N}",
  "step": "run_pilot_backtest",
  "hyp": 1,
  "iter": 2,
  "what_i_did": "re-ran extraction with sharper classifier prompt; same universe",
  "what_worked": "trade count 8 → 18; sharpe 0.12 → 0.44",
  "what_seemed_off": "long-side trades dominate; short-side extraction still finds nothing",
  "suggested_next": "iterate: add a short-side example to the classifier prompt"
}
```

`suggested_next` is **advisory**. The evaluator decides the actual `next_step` from `pilot_history` + gate results; your note gives context to the next invocation's prompt.

## Writing `hypothesis_notes`
When `next_step == revise_hypothesis` AND `pilot_history` has entries for the hypothesis being abandoned, append this shape under `hypothesis_notes`:

```json
{
  "hyp": 1,
  "final_sharpe": 0.44,
  "iterations_spent": 5,
  "why_abandoned": "short-side signal doesn't exist in this corpus regardless of prompt",
  "lessons_for_future_hypotheses": [
    "short-side prem14a events are too rare / noisy here",
    "daily noise dominates — consider weekly aggregation",
    "universe of 12 megacaps may be too efficient — try mid-caps"
  ]
}
```

These notes are **injected into every future run's prompt** (bounded by the 3-hypothesis ceiling), so next hypothesis starts informed.

## Ceilings (backstops — plateau detection is the primary stop)
- Per-hypothesis iterations: 8
- Hypothesis count: 3
- Total iterations: 24
- Stage-level: `max_part_completions = 150`, `budget_seconds = 172800` (48h cumulative)

## Output Files
- `strategy/loop_state.json` — the state machine + full history
- `strategy/hypothesis_v{N}.md` — falsifiable hypothesis, with `required_data_types` list
- `strategy/universe_v{N}.json` — tickers
- `data/processed/pilot/…` — partitioned pilot data (date < oos_cutoff_date)
- `data/processed/oos/…` — held out, never read by this stage
- `data/processed/llm_cache/pilot/{hash}.json` — LLM-extracted features
- `backtest/pilot/strategy_v{N}.py` — MUST reference `data/processed/llm_cache/pilot/`
- `backtest/pilot/results_v{N}.md`, `metrics_v{N}.json`, `metrics_latest.json`

## CRITICAL RULES
- Execute exactly **one** `next_step` per invocation, then END YOUR REPLY NORMALLY — the evaluator runs after each completed step and writes the next `next_step`. Never write a part-completion marker yourself (it SKIPS evaluation; the runner adds it automatically only when you run out of tool budget mid-step). If `next_step` shows `null` or `_advance`, do nothing — finish your reply so the evaluator can adjudicate.
- Never rewrite `oos_cutoff_date`.
- Strategy code MUST load from `data/processed/llm_cache/pilot/`. Fabricated signals (`hashlib.md5`, hardcoded randoms, "simulated/mock/synthetic/fake" anything) will be rejected — both by the lineage check and the phrase blacklist.
- Every `exec` call has a hard 600s cap. Chunk.
- Never print raw data / dataframes beyond 5 rows. Write to files; report row counts + metrics only.

## Quality Bar for `converge`
- Pilot **annualized Sortino ≥ 1.0** — Sortino is the KEY metric for comparing strategies (Sharpe is still recorded)
- Trades ≥ 30
- Plateau (|ΔSortino| < 0.15 across last 2 iters within the same hypothesis)
- Every `pilot_history` row MUST include `sortino` and `annualized_return_pct` (both come free from `compute_metrics(led)`)
- All lineage / variance / coverage / OOS-avoidance gates pass

Missing any of those = keep iterating. Hitting ceilings without converging = `terminate` (stage advances and the verdict stage will write "reject" with reasoning).
