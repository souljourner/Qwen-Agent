# Pipelines — Startup Builder & Trading Strategy Builder
> Details of the two multi-stage pipelines, task routing, and when to proactively suggest them.

## Startup Builder — start_pipeline
6-stage startup project builder: **Market Research → BRD → PRD → VC Pitch → MVP → Review**. Each stage runs independently with acceptance evaluation. Use this for building startup ideas into fully researched, planned, and coded MVPs. Companion tools: `pipeline_status`, `list_pipelines`.

## Trading Strategy Builder — start_trading_pipeline
6-stage research-loop pipeline: **Data Landscape → Research Loop → Full Validation → Verdict → Paper Trading → Review**. Stage 2 is an iteration-aware research loop (hypothesis → fetch → extract → pilot-backtest → revise), driven by `strategy/loop_state.json` and programmatic gates on pilot_history — decisions come from numbers, not LLM narrative. OOS data is frozen on first fetch and only seen in Stage 3 (single-pass validation). Stage 4 writes a promote/reject verdict gated against `backtest/full/metrics.json`; if `reject`, Stages 5 and 6 are skipped and pipeline status becomes `completed_rejected`. Uses `exec` for yfinance/pandas-ta/backtrader on real historical data.

A global lock serializes all pipelines — only one runs at a time.

**Completion email**: every pipeline (both types) automatically emails the owner its final result — outcome, per-stage summary, and verdict excerpt — on completed / rejected / best-effort-exhausted. Don't send a duplicate email when a pipeline finishes; only email extra detail if the user asks.

## Taking over from a failed stage
When a `pipeline:<project>:stage_N` background task fails and you follow up, first rebuild the sub-agent's context — it worked from materials you don't see in chat:
1. `pipeline_status` + the project's `pipeline/state.json` — per-stage status, attempt counts, and `notes` (the evaluator's feedback for every failed attempt).
2. The stage's instructions: `exec cat /app/sandbox_agent/pipeline/stages_trading/stage_N_*.md` (or stages_startup/), plus any learned overrides in `/app/data/pipeline_stages/<type>/` — together these are exactly what the sub-agent was told to do.
3. Its working artifacts under the project dir (`strategy/loop_state.json`, `research/`, `backtest/`).
4. Trading pipelines use the local data store — read_skill('trading-data') for the client API and policies the sub-agent was bound by.
Only then decide: fix the blocker and let the orchestrator reschedule, or report to the user.
**Never cancel a pipeline stage task** (via cancel_task): it burns one of the stage's 5 attempts and the orchestrator reschedules it anyway — it does NOT stop the pipeline. To stop a pipeline the user explicitly asked to kill, use `cancel_pipeline(project)` — it cancels the stage tasks, marks the pipeline cancelled, and releases the lock. Never cancel a pipeline on your own initiative.

## Task routing
- **Simple tasks** (research a topic, write a report): use project tools directly — no pipeline needed
- **Building a startup idea end-to-end**: use `start_pipeline`
- **Recurring tasks** (daily reports, periodic checks): use `schedule_task` with cron
- Do NOT create monitor tasks for pipelines — the orchestrator advances stages automatically

## Proactive suggestion
When a user describes a business or startup idea with enough detail (who it's for, what problem it solves), suggest using `start_pipeline` to build it end-to-end. Example: "That sounds like a solid idea! Want me to kick off the Startup Builder Pipeline? It will research the market, write a business plan, create a PRD, draft a VC pitch, build an MVP, and review everything — all automatically."
