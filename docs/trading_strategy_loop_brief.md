# Brief: Build a Trading Strategy Research Loop (standalone)

You are starting in an empty directory. By the end, you will have a
working Python CLI that runs a real trading-strategy research process:

```
hypothesize → gather data → pilot-backtest on a small sample → learn →
revise the hypothesis, data, or strategy code → re-pilot → loop until
diminishing returns → verdict (works / doesn't work) → if works, build
paper-trading scaffolding; otherwise terminate cleanly.
```

This is a research process, not a checklist. Every iteration should
move the strategy meaningfully closer to either "this works" or "this
won't work" — never busy-loop on a stuck hypothesis.

## Environment assumptions

- Python 3.10+
- An OpenAI-compatible LLM endpoint reachable via env vars
  (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). Use the `openai`
  Python SDK to talk to it.
- Internet access is OPTIONAL — the user will provide a data source.
  Don't hard-code Yahoo Finance / SEC EDGAR scraping; accept any
  reasonable source the user supplies in stage 1.
- Tests run under `pytest`.

## Non-negotiable principles

1. **The iteration loop is a state machine on disk.** Each step
   reads `loop_state.json`, executes exactly ONE step (fetch / extract
   / backtest / revise), updates the file, exits. The LLM that drives
   it is called once per step — it cannot read backtest metrics and
   respond to them within a single LLM call. Long-running steps
   (data fetch, feature extraction) MUST chunk and resume.

2. **OOS data must NEVER be seen by the loop.** On first data fetch,
   partition chronologically into `data/processed/pilot/` (first 80%)
   and `data/processed/oos/` (last 20%). Freeze the boundary date in
   `loop_state.json` as `oos_cutoff_date`. Every later step asserts the
   cutoff is unchanged and every pilot row satisfies
   `date < oos_cutoff_date`. Stage 3 (validation) is the FIRST look at
   OOS data.

3. **Decisions come from numbers, not LLM narrative.** Every gate
   (converge / iterate / dead_end / promote / reject) is a structured
   check over persisted metric history. The LLM writes the story; math
   makes the call.

4. **Plateau detection is the primary stop**, integer ceilings are
   hard backstops only.

5. **Lineage is positive.** Strategy code MUST reference the LLM
   feature cache at `data/processed/llm_cache/pilot/`. A strategy that
   doesn't read upstream features fails the lineage check. A negative
   phrase blacklist (mock-signal language) stays as belt-and-suspenders.

## Architecture

```
<run_dir>/                                        per-run state directory
├── research/
│   └── data-landscape.md                         (stage 1) candidate sources
├── strategy/
│   ├── hypothesis_v1.md                          (stage 2 init) declares required_data_types
│   ├── hypothesis_v2.md                          (stage 2 revise_hypothesis)
│   ├── universe_v1.json                          (stage 2 init) tickers / IDs
│   └── loop_state.json                           ← state machine source of truth
├── data/
│   ├── raw/                                      (stage 2 fetch_data)
│   └── processed/
│       ├── pilot/                                date < oos_cutoff_date
│       ├── oos/                                  date ≥ oos_cutoff_date (untouched until stage 3)
│       └── llm_cache/
│           └── pilot/{hash}.json                 LLM extraction cache (resumable)
├── backtest/
│   ├── pilot/
│   │   ├── strategy_v1.py                        MUST reference data/processed/llm_cache/pilot/
│   │   ├── strategy_v2.py
│   │   ├── results_v1.md
│   │   ├── metrics_v1.json
│   │   └── ...
│   └── full/                                     populated only by stage 3
│       ├── results.md
│       └── metrics.json
└── pipeline/
    ├── verdict.md                                (stage 4)
    └── metrics.json                              (stage 4)
```

Implement these as a small Python package:

```
trading_loop/
├── __init__.py
├── llm.py                — llm_call(prompt, system, model) and llm_batch(...)
├── state.py              — load/save loop_state.json with strict schema validation
├── gates.py              — programmatic acceptance gates (each is bool, msg)
├── decision.py           — plateau-based StageDecision over pilot_history
├── steps.py              — step state machine: dispatches on loop_state.next_step
├── verdict.py            — stage-4 verdict logic; writes verdict.md + metrics.json
├── stage1.py             — data landscape one-shot (LLM-driven)
├── stage3.py             — full validation single-pass OOS test
├── stage5.py             — (optional) paper trading scaffold
├── stage6.py             — (optional) final review
└── cli.py                — argparse entrypoint: `python -m trading_loop ...`
tests/
├── test_gates.py
├── test_decision.py
├── test_state.py
├── test_steps.py
└── test_verdict.py
```

The CLI shape:

```
python -m trading_loop init <run_dir> --data-source <description>
python -m trading_loop step <run_dir>          # execute exactly one loop step
python -m trading_loop run  <run_dir>          # run until terminal
python -m trading_loop status <run_dir>        # print current loop state
```

## The 6 stages

| # | Name | Goal | Key output | Iterates? |
|---|---|---|---|---|
| 1 | Data Landscape | Catalog real data sources with plaintext samples | `research/data-landscape.md` | No |
| 2 | Research Loop | Converge on a strategy on pilot data, OR conclude none exists | `strategy/loop_state.json` + `backtest/pilot/*` | **Yes** |
| 3 | Full Validation | Lock strategy, FIRST look at OOS | `backtest/full/metrics.json` | No |
| 4 | Verdict | Promote / reject with programmatic gate | `pipeline/verdict.md` + `pipeline/metrics.json` | No |
| 5 | Paper Trading | Scaffold (only if verdict=promote) | `paper/` | No |
| 6 | Final Review | Learnings (only if verdict=promote) | `pipeline/review.md` | No |

## Stage 1 — Data Landscape (one-shot)

**Goal.** Catalog real data sources with plaintext samples and concrete
acquisition paths. No hypothesis yet.

**Outputs.** `research/data-landscape.md` — for each candidate source:
how to fetch (with a runnable code block), sample 500-char extraction,
preliminary "why this might have LLM-extractable alpha."

**Acceptance** (programmatic only, no LLM eval):
- File exists, ≥ 100 chars.
- ≥ 2 sources named.
- Each source includes a fenced code block showing the fetch call.

## Stage 2 — Research Loop (iteration-aware)

The hardest stage by far. Most of this brief is about it.

### Step granularity: ONE LLM invocation = ONE step

The LLM gets one invocation to produce a single artifact (write a file,
run a backtest, append a run note). It cannot read backtest metrics and
respond to them in the same call — that requires a fresh invocation.
So an "iteration" (hypothesize → data → features → backtest → decide)
is split across several steps. Each step exits cleanly after writing
its output.

Canonical step types (one per invocation):

| Step | Inputs | Outputs | Time cap |
|---|---|---|---|
| `init` | `data-landscape.md` | `strategy/hypothesis_v1.md` (declares `required_data_types`), `strategy/universe_v1.json`, initial `loop_state.json` (with `oos_cutoff_date` frozen) | seconds |
| `fetch_data` | hypothesis, universe | `data/raw/`, `data/processed/pilot/` (date<cutoff), `data/processed/oos/` (date≥cutoff) | ≤ 600s; chunk if larger |
| `extract_features` | pilot data + extraction prompt | `data/processed/llm_cache/pilot/{hash}.json` (cached, resumable) | ≤ 600s; chunk |
| `run_pilot_backtest` | strategy code + llm_cache | `backtest/pilot/results_v{N}.md`, `backtest/pilot/metrics_v{N}.json`, append one `pilot_history` row + one `run_note` to `loop_state.json` | < 600s for pilot universe |
| `revise_hypothesis` | run_notes, pilot_history | `strategy/hypothesis_v{N+1}.md`, bump `hypothesis_count`, append `hypothesis_notes` entry | seconds |
| `revise_data_processing` | run_notes, same hypothesis | new extraction prompt, mark cache stale | seconds |
| `revise_strategy_code` | run_notes, same hypothesis + data | `backtest/pilot/strategy_v{N+1}.py` | seconds |
| `extend_pilot_window` | run_notes | adds rows where `date < oos_cutoff_date` (cutoff is IMMUTABLE) | ≤ 600s |

`steps.py` is a state machine keyed on `loop_state.json.next_step`. On
each invocation it reads the file, executes exactly that step, updates
the file, exits.

### `next_step` is chosen programmatically, NOT by the LLM

After every `run_pilot_backtest`, run the structured Decision gate
(table below) over `pilot_history` and write `next_step` +
`current_phase` directly into `loop_state.json` as part of acceptance.
The LLM writes a `run_note` narrative (what I did / what worked / what
seemed off / suggested next), but it does NOT choose `next_step` —
that's math. The `suggested_next` field in a run_note is advisory
context for the next invocation, not control flow.

### `loop_state.json` schema

```json
{
  "hypothesis_count": 2,
  "iteration_within_hypothesis": 3,
  "total_iterations": 7,
  "next_step": "run_pilot_backtest",
  "current_phase": "pilot | revise_hypothesis | revise_data_processing | revise_strategy_code | extend_pilot_window",
  "oos_cutoff_date": "2024-04-01",
  "pilot_history": [
    {"hyp": 1, "iter": 1, "sharpe": 0.12, "trades": 8, "failure_mode": "insufficient_signal", "run_note_id": "r001"},
    {"hyp": 1, "iter": 2, "sharpe": 0.44, "trades": 18, "failure_mode": null, "run_note_id": "r002"}
  ],
  "run_notes": [
    {"id": "r001", "step": "run_pilot_backtest", "hyp": 1, "iter": 1,
     "what_i_did": "ran v1 on 12 tickers × 6 months",
     "what_worked": "cache hit rate 94%",
     "what_seemed_off": "70% of events classified neutral — prompt too conservative",
     "suggested_next": "revise_data_processing: sharpen classifier prompt"}
  ],
  "hypothesis_notes": [
    {"hyp": 1, "final_sharpe": 0.44, "iterations_spent": 5,
     "why_abandoned": "short-side signal doesn't exist in this data",
     "lessons_for_future_hypotheses": [
       "classifier works directionally but short-side events too rare",
       "consider weekly aggregation — daily noise dominates"
     ]}
  ],
  "last_decision": "iterate"
}
```

**Why three separate logs:**
- `pilot_history` — structured numbers for the programmatic decision gate.
- `run_notes` — narrative for the next invocation; rotates (only last 3
  enter the next prompt). Older notes are intentionally dropped so the
  prompt doesn't bloat across iterations.
- `hypothesis_notes` — distillation that survives across hypotheses; ALL
  entries are injected on every invocation (capped by hypothesis ceiling
  of 3, so always small and finite).

### Decision contract — `decision.py`

```python
@dataclass
class StageDecision:
    passed: bool                     # whether to advance to stage 3
    decision_type: str               # one of the values below
    feedback: str                    # human-readable reason
    metrics: dict                    # snapshot of last pilot row + plateau state
```

| `decision_type` | Trigger | Loop action |
|---|---|---|
| `converge` | Last pilot meets ALL: Sharpe ≥ 0.8, trades ≥ 30, plateau (2 consec iters \|ΔSharpe\| < 0.15), all lineage checks pass | `passed=True`, advance to stage 3 |
| `needs_more_data` | Pilot promising (Sharpe > 1.2) but trades < 30 OR sample < 6mo | `passed=False`, `current_phase=extend_pilot_window` |
| `iterate` | Last iter improved by ≥ 0.15 Sharpe OR reduced a named failure mode | `passed=False`, same hypothesis |
| `infeasible_data` | llm_cache variance fails (one bucket > 60%) OR coverage < 50% | `passed=False`, `current_phase=revise_data_processing` |
| `insufficient_signal` | ≥ 2 iters with < 10 trades | `passed=False`, `current_phase=revise_hypothesis` |
| `dead_end` | 2 consec iters \|ΔSharpe\| < 0.15 AND no failure-mode reduction; current hypothesis exhausted | `passed=False`, bump `hypothesis_count`, start new hypothesis (if < 3) |
| `terminate` | `hypothesis_count >= 3` OR total iterations > 24 OR budget exhausted | `passed=True` with TERMINATED marker; advance — verdict stage handles "no viable strategy" |

### Programmatic gates — `gates.py`

Run in this order; any failure short-circuits the decision:

- `check_hypothesis_data_feasibility(run_dir, loop_state) -> (bool, str)`
  — current `hypothesis_v{N}.md` MUST declare `required_data_types`;
  every entry must appear as a named source in `research/data-landscape.md`.
  Fails fast before fetch/extract burns budget on an unrealizable hypothesis.

- `check_pilot_avoids_oos(run_dir, loop_state) -> (bool, str)` — no
  `oos/` reference in pilot code; every row in `data/processed/pilot/`
  has `date < oos_cutoff_date`; reject if `oos_cutoff_date` was rewritten
  since the previous run.

- `check_strategy_reads_llm_cache(run_dir) -> (bool, str)` — latest
  `backtest/pilot/strategy_v{N}.py` MUST contain the literal string
  `data/processed/llm_cache/pilot/`. Static check; positive lineage.
  A strategy that ignores the upstream LLM-extracted features fails here.

- `check_llm_cache_variance(run_dir) -> (bool, str)` — load cache JSON,
  compute distribution; reject if max bucket > 0.6 or distinct values < 3.
  Catches degenerate classifiers (e.g. "everything is neutral").

- `check_cache_coverage(run_dir, loop_state) -> (bool, str)` — ≥ 80% of
  universe declared in `hypothesis_v{N}.md` has a corresponding cache entry.

- `check_backtest_uses_real_signals(run_dir) -> (bool, str)` — phrase
  blacklist ("mock", "fake", "placeholder", "hashlib.md5", etc.) in
  pilot code. Belt-and-suspenders; the lineage check above is primary.

### Stage 2 ceilings (hard backstops; plateau detection is primary)

- Per-hypothesis iterations: 8
- Hypothesis count: 3
- Total iterations: 24
- Total budget: 48 hours wall-clock

## Stage 3 — Full Validation (single pass, no iteration)

**Goal.** Strategy is locked; FIRST look at OOS data.

**Outputs.** `backtest/full/results.md`, `backtest/full/metrics.json`
(machine-readable), per-window walk-forward results.

**Programmatic gates (ALL must pass for acceptance):**
- `oos_sharpe / pilot_sharpe >= 0.5` — single best honesty check.
- Walk-forward: strategy beats benchmark on risk-adjusted terms in ≥ 60% of rolling windows.
- `total_trades >= 100` over full period.
- `t_stat_daily_returns > 2.0`.
- Deflated Sharpe (López de Prado, adjusted for up to 24 pilot trials) > 0.
- Turnover consistent with the holding-period claim in the converged `hypothesis_v{N}.md` (± 50%).

If acceptance fails here it is NOT a retry — one OOS peek is all you
get. Mark stage `completed-no-more-attempts` with the failing gate in
the feedback; the verdict stage handles the rejection narrative.

## Stage 4 — Verdict (always runs; programmatically gated)

**Goal.** Write `pipeline/verdict.md` and `pipeline/metrics.json` with
the promote/reject call.

**Programmatic gate.** Verdict's "Final Recommendation" field MUST match
`metrics.json`. If all stage 3 gates passed → must be "promote." If any
failed → must be "reject." Mismatch fails acceptance and the stage is
rewritten.

The verdict ALWAYS runs — even if the loop terminated without converging,
write a "reject" verdict explaining why. This is how the pipeline
gracefully concludes a research effort that didn't pan out.

## Stage 5 — Paper Trading Scaffold (only if verdict = promote)

A standard scaffolding stage: write the paper-trading code that polls
live data, runs the strategy, logs trades. Skip entirely if
`verdict.md` says reject — record "skipped (verdict = reject)" and
move on.

## Stage 6 — Final Review (only if verdict = promote)

Captures learnings from the run into `pipeline/review.md`. Same
skip-if-reject guard.

## Tests to write

Use `pytest`. Each gate has dedicated tests:

- `test_gates.py`:
  - Lineage: write a strategy file containing only
    `import hashlib; sig = hashlib.md5(b'a').hexdigest()` —
    `check_strategy_reads_llm_cache` must reject.
  - OOS avoidance: pilot code referencing `data/processed/oos/` must
    be rejected; pilot rows with `date >= oos_cutoff_date` must be
    rejected; `oos_cutoff_date` rewriting must be rejected.
  - Cache variance: 95% of entries identical → rejected.
  - Cache coverage: < 80% universe coverage → rejected.
  - Hypothesis feasibility: hypothesis declares a data type not in
    `data-landscape.md` → rejected.
  - Mock-signal blacklist: "we used a mock signal for testing" → rejected.

- `test_decision.py`:
  - Plateau decision: synthetic `pilot_history` with Sharpes
    `[0.12, 0.40, 0.43, 0.44]` (plateau after iter 3) → expect
    `dead_end` (last Sharpe below 0.8 threshold). Replace the last
    value with `0.85` → expect `converge`.
  - `insufficient_signal`: two iters with `trades < 10` → triggered.
  - `terminate`: `hypothesis_count == 3` and last decision was `dead_end`
    → triggered.

- `test_state.py`: round-trip `loop_state.json`; reject `oos_cutoff_date`
  mutation; reject malformed schema.

- `test_steps.py`: each step type writes the right files; no step
  reads from `data/processed/oos/`.

- `test_verdict.py`: "verdict matches metrics" gate — mismatch fails
  acceptance.

## How to run (target UX)

```bash
# Set up
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini  # or whatever
pip install openai pytest pandas numpy

# Initialize a run
python -m trading_loop init runs/prem14a-2024 \
    --data-source "PREM14A SEC filings 2020-2024 for tender-offer events"

# Run interactively, one step at a time
python -m trading_loop step runs/prem14a-2024     # init
python -m trading_loop step runs/prem14a-2024     # fetch_data
python -m trading_loop step runs/prem14a-2024     # extract_features
python -m trading_loop step runs/prem14a-2024     # run_pilot_backtest
# decision gate decides next_step automatically; keep stepping until terminal

# Or run to completion
python -m trading_loop run runs/prem14a-2024

# Check status anytime
python -m trading_loop status runs/prem14a-2024
```

## How to work

1. **Read this brief end to end before writing code.** The principles
   and the loop_state schema are the spec — implementations are
   variable, the contract isn't.

2. **Bottom-up.** Build in this order:
   1. `state.py` (schema, load/save)
   2. `gates.py` (each gate is small and isolated — write test first)
   3. `decision.py` (pure function over `pilot_history`)
   4. `llm.py` (thin wrapper over openai SDK)
   5. `steps.py` (state machine; calls into the above)
   6. `stage1.py`, `stage3.py`, `verdict.py` (one-shot stages)
   7. `cli.py`
   8. `stage5.py`, `stage6.py` (last; only matter if a strategy promotes)

3. **Use TDD on the gates and decision logic.** They're pure functions
   over filesystem state and JSON — each test is small, each gate's
   correctness is the entire safety story. Don't skip these tests.

4. **Don't over-engineer.** No web UI, no database, no async. Plain
   Python files, plain JSON state, synchronous CLI. Simplicity and
   auditability beat sophistication.

5. **Don't silently work around the spec.** If a constraint here forces
   an ugly compromise, FLAG IT in your README — explain what you did
   and why. The user decides if it's acceptable.

6. **End-to-end smoke test.** Once everything is wired, run the loop
   on a small synthetic data source (~20 news articles, dummy tickers).
   Expect: at least one iteration, real `pilot_history.json` with real
   numbers, a verdict that matches the metrics. If the loop spins
   without progress, your decision gate is wrong; debug there first.

You have full autonomy on names, file structure inside the constraints
above, and any minor refactor that helps. The principles are the spec.
