# Pipelines — Startup Builder & Trading Strategy Builder
> Details of the two multi-stage pipelines, task routing, and when to proactively suggest them.

## Startup Builder — start_pipeline
6-stage startup project builder: **Market Research → BRD → PRD → VC Pitch → MVP → Review**. Each stage runs independently with acceptance evaluation. Use this for building startup ideas into fully researched, planned, and coded MVPs. Companion tools: `pipeline_status`, `list_pipelines`.

## Trading Strategy Builder — start_trading_pipeline
6-stage research-loop pipeline: **Data Landscape → Research Loop → Full Validation → Verdict → Paper Trading → Review**. Stage 2 is an iteration-aware research loop (hypothesis → fetch → extract → pilot-backtest → revise), driven by `strategy/loop_state.json` and programmatic gates on pilot_history — decisions come from numbers, not LLM narrative. OOS data is frozen on first fetch and only seen in Stage 3 (single-pass validation). Stage 4 writes a promote/reject verdict gated against `backtest/full/metrics.json`; if `reject`, Stages 5 and 6 are skipped and pipeline status becomes `completed_rejected`. Uses `exec` for yfinance/pandas-ta/backtrader on real historical data.

A global lock serializes all pipelines — only one runs at a time.

## Task routing
- **Simple tasks** (research a topic, write a report): use project tools directly — no pipeline needed
- **Building a startup idea end-to-end**: use `start_pipeline`
- **Recurring tasks** (daily reports, periodic checks): use `schedule_task` with cron
- Do NOT create monitor tasks for pipelines — the orchestrator advances stages automatically

## Proactive suggestion
When a user describes a business or startup idea with enough detail (who it's for, what problem it solves), suggest using `start_pipeline` to build it end-to-end. Example: "That sounds like a solid idea! Want me to kick off the Startup Builder Pipeline? It will research the market, write a business plan, create a PRD, draft a VC pitch, build an MVP, and review everything — all automatically."
