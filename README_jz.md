# Sandbox Agent — Knowledge Transfer Document

A sandboxed, self-scheduling agent built on [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), with a Chainlit chat surface (replaced Gradio), a code-driven multi-stage pipeline orchestrator, and design inspiration from [OpenClaw](https://docs.openclaw.ai/) and [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Quick Start

```bash
# Foreground run, no rebuild (assumes image already built)
./sandbox_agent/docker/run.sh

# Foreground rebuild + run (pick up code/.env changes)
./sandbox_agent/docker/restart.sh

# Shut down
./sandbox_agent/docker/shutdown.sh

# Chat UI:        http://localhost:7860
# Status server:  http://localhost:7861

# Local tests (host venv with pytest installed)
.venv/bin/python -m pytest tests/sandbox_agent/ -q

# Dev REPL (no container, no Chainlit) — useful for one-off debugging
python -m sandbox_agent.main --mode repl
```

`run.sh` ensures Docker Desktop is up before launching (handles the cold-daemon case on macOS).

## Infrastructure

| Service | Host | Model | Role |
|---|---|---|---|
| **vLLM (primary)** | 192.168.4.66:8000 | qwen3.6-27b-linux | Main conversation + background tasks; vLLM-side concurrency ~16 |
| **vLLM (backup)** | 192.168.4.66:8000 | qwen3.5 (397B MoE) | Chat fallback when primary slots saturated; also fallback for `llm_call()` |
| **Tools API** | localhost:8080 (host) / `host.docker.internal:8080` (container) | — | Web search (Brave), URL fetch with pagination, stock prices via SSE |
| **Browser (in-container)** | Playwright headless Chromium | — | Navigate, screenshot, click, type, scroll. Persistent cookies. Optional headed mode via Xvfb + noVNC on port 6080 |
| **Chainlit chat** | localhost:7860 | — | Chat UI (replaces the legacy Gradio path) |
| **Status server** | localhost:7861 | — | Live activity monitor, digest, agent requests |
| **Data** | `~/sandbox_agent_data/` | — | Bind-mounted volume (Mac Finder–visible). Includes `chat.db`, `.cl_elements/`, `projects/`, `activity.jsonl`, daily logs |

## Architecture

```
Docker container (no-new-privileges, cap_drop: ALL, tmpfs /tmp:10G)
│
├── Chainlit (PID 1, asyncio event loop, port 7860)
│   ├── on_message       → _execute_agent_turn (drains worker→queue, streams to UI)
│   ├── on_chat_start    → fresh history
│   └── on_chat_resume   → rebuild history from chat.db (assistant text + user only;
│                          tools/reasoning intentionally NOT replayed — see shortcomings)
│
├── Worker threads (one per chat turn)
│   └── _run_agent_in_thread runs agent.run(messages), pushes cumulative
│       List[Message] chunks onto an asyncio.Queue via call_soon_threadsafe
│
├── _StreamBridge (asyncio side, per-turn)
│   ├── consume()            — maps cumulative chunks → cl.Message/cl.Step deltas
│   ├── stream_token batching (~12 emits/sec) — client-side render protection
│   ├── _resolved_tool_steps — guard so finished tool steps aren't re-update()d per chunk
│   └── flush_pending_text() — flushes the held-back tail BEFORE the per-turn footer
│
├── Background lanes (in-process threads — Phase B will parallelize pipelines)
│   ├── Heartbeat (1h)       — checks HEARTBEAT.md
│   ├── Cron loop (60s poll) — pulls due tasks, executes one-at-a-time
│   ├── Notifier loop (4s)   — drains task_notify queue, injects [system event]
│   │                          notices into the originating chat (chat_origin routing)
│   ├── LLM bridge (HTTP, localhost) — code_interpreter llm_call() / llm_batch()
│   └── Status server (separate process via multiprocessing, port 7861)
│
└── Data layer
    ├── SQLAlchemyDataLayer over sqlite+aiosqlite (chat.db)
    │   └── HARDENED: WAL journal mode + busy_timeout=30000 + synchronous=NORMAL
    │                 (was journal_mode=delete → QueuePool exhaustion + 2 tok/s lag)
    ├── LocalFsStorageClient under DATA_DIR/.cl_elements/ — image/PDF persistence
    └── activity.jsonl — append-only structured event log
```

### Model routing

- **User chat**: tries to grab one of the 15 primary-model slots (non-blocking semaphore). Slot free → primary (`qwen3.6-27b-linux`). All saturated → falls back to the 397B `qwen3.5` so chat is never queued behind background work.
- **Background tasks (cron + heartbeat)**: always acquire the primary blocking — they wait. The 397B is reserved for chat fallback only.
- **`llm_call()` / `llm_batch()` (inside `code_interpreter`)**: same primary→backup chain via the in-process LLM bridge. Empty-completion retry once per model before falling through.
- **Native vLLM tool calling** (`use_raw_api=True`): `tools=` passed natively to vLLM. Required because vLLM's auto-tool-call-parsing otherwise consumes the `<tool_call>` tokens in streaming and drops them.

### Concurrency primitives

| Lock | Type | Scope | What it does |
|---|---|---|---|
| `_primary_model_lock` | `BoundedSemaphore(15)` | Process-wide | Limits in-flight primary-model calls; overflow → 397B backup |
| `_background_work_lock` | `threading.Lock` | Process-wide | Serializes heartbeat + cron (background sessions don't interleave) |
| Pipeline file lock | `LOCK_FILE` on disk, single holder | Process-wide | **Serializes ALL pipelines** to one at a time (Phase B will make per-project) |
| Cron `ThreadPoolExecutor` | `max_workers=1` | Cron lane | Single worker = one cron task at a time |
| `_get_turn_lock()` | `asyncio.Lock` on `cl.user_session` | **Per-session** | Serializes turns *within* a chat (lets notifier turns wait); different chats run concurrently |

**Multiple concurrent Chainlit chats work today.** The turn lock is per-session and the primary lock allows 15 concurrent LLM calls; overflow routes to the 397B. The current ceiling is *pipeline* concurrency (Phase B).

## Chat surface (Chainlit specifics)

| Concern | How it's handled |
|---|---|
| **Persistence** | `SQLAlchemyDataLayer` (community) over SQLite at `DATA_DIR/chat.db`, **WAL mode + busy_timeout=30000 + synchronous=NORMAL** to prevent connection-pool exhaustion under concurrent step persistence. Schema bootstrapped from `chat_data_layer._CHAINLIT_SCHEMA_SQL`, including additive migrations for `autoCollapse`/`modes` columns. |
| **Elements (images/PDFs)** | `LocalFsStorageClient(BaseStorageClient)` writes blobs under `DATA_DIR/.cl_elements/{user_id}/{element_id}`. A custom `@app.get("/cl-elements/{object_key:path}")` route serves them with a `realpath` containment guard. |
| **Streaming** | Worker pushes cumulative `List[Message]` chunks; drain coalesces (keeps latest chunk) and computes deltas; `stream_token` batched to ~12/sec to protect the React client from per-event markdown re-parse. |
| **Tool step lifecycle** | `cl.Step(type="tool")` opens on assistant function_call, closes on the matching function result. **Each result is applied exactly once** (`_resolved_tool_steps` set) — without this guard, every subsequent chunk re-`update()`d every tool step → 2 tok/s lag on multi-tool turns. |
| **Reasoning** | `reasoning_content` → `cl.Step(type="thought")`, throttled updates at 0.2s. NOT persisted into the agent's next-turn context (decided cost > benefit). |
| **Per-turn footer** | After the turn ends, `flush_pending_text()` flushes the batched tail BEFORE appending the `📊 _last turn: max ctx N / out · chat high-water: M (P% of 256k)_` footer. Without this flush the footer used to splice into the middle of the last sentence. |
| **Token usage** | Captured in `oai.py` from vLLM's `stream_options={"include_usage": True}` trailing chunk. Logged to `activity.jsonl` as `llm_usage` events + bubbled to the on_message handler via `register_usage_hook` for the footer. |
| **Multimodal input** | Image uploads in Chainlit → `ContentItem(image=data:...)` in the user `Message` content list. `convert_messages_to_dicts` in `oai.py` produces OpenAI-shape multimodal payload. All hosted vLLM models are vision-capable. |
| **Stop button** | Cancels the `on_message` coroutine, which calls `cancellation.cancel(run_id)`. The worker's `_compacting_run` raises `RunCancelled` at the next yield via `cancellation.guard`, and any registered child PGID gets `SIGKILL`ed. |
| **Resume** | `on_chat_resume` rebuilds `HISTORY_KEY` from persisted `user_message` + `assistant_message` steps only — **drops all tool steps and reasoning** (intentional, but see "Known shortcomings"). |

### Background-task completion notices (Hermes-style)

When a cron task / pipeline stage completes, `task_notify.put(...)` enqueues an event. A 4s notifier loop drains the queue and:
1. Renders a chat bubble (`author="background task"`) so the user sees it.
2. Injects a `role="user"` message tagged `[system event]` into the originating chat's HISTORY_KEY (vLLM rejects mid-thread `role="system"` with HTTP 400). The agent reads it and decides whether to act.
3. Routes to the right session via `chat_origin` (the worker thread stamps `{session_id, thread_id}` on the task when scheduled; the notifier reads it back).

## Tools (46 registered)

| Category | Tools | Notes |
|---|---|---|
| **Web** | `web_search`, `web_url_fetch`, `stock_price` | Call port 8080 API via SSE; outputs sanitized (`sanitize_web_content`). `web_url_fetch` supports `offset`/`max_chars` for pagination with a trusted hint appended when `has_more`. Brave Search rate-limited to 1 req / 2s. |
| **Browser** | `browser_navigate`, `browser_screenshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_save_credentials`, `browser_get_credentials` | Playwright headless Chromium with persistent cookies (`DATA_DIR/browser_state/`). Stealth mode (disable-blink-features, disable-http2, masked navigator.webdriver). `browser_screenshot` returns `List[ContentItem]` with base64 PNG for vision model. Optional headed mode via Xvfb + noVNC on port 6080 (`XVFB_ENABLED=true`). Encrypted credential storage. |
| **Code execution** | `code_interpreter`, `exec` | Both stateless subprocesses with `start_new_session=True` + killpg on timeout. `code_interpreter`: fresh Python per call (numpy/pandas/requests pre-imported), `llm_call()` via the in-process bridge, 1-hour max timeout. `exec`: any shell command; with `project=` it **runs inside that project's `.venv`** (auto-created with `uv venv`, activated via PATH+VIRTUAL_ENV) — see "Per-project dependency isolation" below. |
| **Scheduling** | `schedule_task`, `list_tasks`, `complete_task`, `cancel_task`, `pause_task`, `resume_task`, `update_task_checkpoint` | JSON-backed queue (`tasks.json`, completed/cancelled archived to separate files). |
| **Pipeline** | `start_pipeline`, `start_trading_pipeline`, `pipeline_status`, `list_pipelines` | Two pipeline kinds (startup builder + trading research); see Pipeline section. |
| **Self-edit** | `update_soul`, `update_heartbeat` | Section patches (not full file replacement); auto-committed to git. |
| **Memory** | `read_memories`, `add_memory` | `MEMORIES.md` dated entries, auto-loaded into the system prompt. |
| **Projects** | `create_project`, `list_projects`, `delete_project`, `rename_project`, `update_project`, `project_write_file`, `project_read_file`, `project_list_files`, `project_delete_file`, `project_apply_patch` | Persistent file workspaces under `projects/<slug>/`. `project_write_file` supports write/append/edit modes. `project_apply_patch` does OpenClaw-style multi-file context-matching patches. `rename_project` renames the folder + updates `.project.json`. `update_project` updates the description without touching files. |
| **Display** | `display_doc`, `download_file` | Out-of-band file display in Chainlit (text/markdown/code, images, PDFs) — content **does NOT enter the agent's context**; a thread-keyed display hook pushes the payload to `_StreamBridge.display_document()`. `download_file` offers project files for download (binary/archives). Both restricted to project directories. Persists as an element so it survives reload. |
| **Filesystem** | `move_file`, `delete_file` | Generic file ops within DATA_DIR. |
| **Chat logs** | `list_chat_logs`, `read_chat_log` | Daily markdown logs at `DATA_DIR/chat_logs/YYYY-MM-DD.md`. |
| **Notifications** | `request_user`, `view_requests`, `resolve_request` | Structured pending/resolved JSON requests with dedup. **File-based only — no push/email/Slack.** |

## Pipeline orchestrator

Two code-driven pipelines, each running stages sequentially with acceptance evaluation, retry on failure, and stage-to-stage artifact passing. Flow controlled by Python, not prompts.

### Startup builder (`start_pipeline`)

| # | Stage | Output | Description |
|---|---|---|---|
| 1 | Market Research | `research/market-research.md` | Market size, competitors, timing |
| 2 | BRD | `business/brd.md` | Branding, legal, scalability, operations, finance |
| 3 | PRD | `product/prd.md` | User stories, MVP scope, technical requirements |
| 4 | VC Pitch | `business/vc-pitch.md` | Elevator + warm-contact pitches |
| 5 | MVP | `mvp/` | Frontend, backend, tests, DB scripts |
| 6 | Review | `pipeline/review.md` | Learnings + instruction-improvement suggestions |

### Trading pipeline (`start_trading_pipeline`)

Sibling pipeline for trading-strategy research, used heavily for the active `shapiro-*` projects. Same orchestrator + retry mechanics.

### Mechanics

- **Stage instructions** live as editable markdown at `sandbox_agent/pipeline/stages/`. Stage 6 evaluates and suggests improvements.
- **Acceptance**: each stage runs (a) programmatic checks (file exists, sections present) then (b) an LLM quality check. PASS → schedule next stage; FAIL with retries → add feedback notes, requeue (up to 5x); FAIL exhausted → mark `completed-no-more-attempts` / `failed-no-more-attempts` and proceed/notify.
- **Job states**: `scheduled → running → completed | part-completion | failed`. `part-completion` is for graceful tool-budget exhaustion (the agent runs a wrap-up call when `MAX_LLM_CALL_PER_RUN` is hit; `stage_runner._detect_part_completion` looks for the `⚠️ ... (part-completion)` marker and re-queues the stage to continue).
- **Serialization (today)**: a single global file lock + `_background_work_lock` + `max_workers=1` executor means **only one pipeline runs at a time** across the whole system. Phase B will make the lock per-project and bound concurrency at 3.
- **Rerun**: `start_pipeline` on a completed project resets all stages; previous artifacts are kept so the agent improves them rather than starting from scratch.

## Per-project dependency isolation (Phase A — pending deploy)

Just-shipped (working tree, requires rebuild):

- `exec` with `project=X` auto-creates `projects/<X>/.venv` via `uv venv` on first use and **activates it** (`VIRTUAL_ENV`, `PATH` prepended, `PYTHONHOME` cleared) — so each project's `pip` / `python` are isolated. `uv` is the default package manager (cache-backed, near-instant re-installs).
- Toggle via `PROJECT_VENV_ENABLED` (default `true`); UV binary path via `UV_BIN` (default `uv`).
- `code_interpreter` still uses the shared global env (its model is ad-hoc, stateless Python with pre-imported numpy/pandas — not project-scoped). SOUL guidance directs project deps through `exec` with `project=`.

**Migration note:** existing projects (shapiro-*, etc.) have no `.venv` and previously installed into the shared global env. After deploy, the first `exec(project=X)` creates a fresh empty `.venv` and activates it — their previously-global deps won't be importable until re-installed (UV-cache fast, but a one-time bump). Set `PROJECT_VENV_ENABLED=false` to keep existing projects on the global env during migration.

## Cancellation model

- `cancellation.begin_run(run_id)` is a context manager that tags **the current thread** with the run id. Used by `_run_agent_in_thread` so any child subprocess registered via `register_child_pgid` is reachable.
- `cancellation.guard()` is called by `_compacting_run` at each generator yield; raises `RunCancelled` if the cancel flag for this thread's run is set.
- The Chainlit Stop button cancels the `on_message` coroutine → `CancelledError` is caught in `_execute_agent_turn`, which calls `cancellation.cancel(run_id)` → the worker's next `guard()` raises and any registered subprocess gets `SIGKILL` via process-group kill. Then the agent unwinds normally.

## Key files

```
sandbox_agent/
├── main.py                      # LockingAgent, bootstrap_background, cron_loop, REPL mode, --mode flag
├── chat_app.py                  # Chainlit entry: on_chat_start/resume/message, _StreamBridge, _execute_agent_turn,
│                                #   notifier loop, /cl-elements route, token-usage footer, multimodal
├── chat_data_layer.py           # SQLAlchemy data layer + WAL/busy_timeout/synchronous=NORMAL hardening
├── chat_storage.py              # LocalFsStorageClient (BaseStorageClient implementation for blobs)
├── chat_logger.py               # Daily chat logs + tools
├── chat_origin.py               # session/thread origin stamping for routing notifier events back
├── task_notify.py               # Background-task completion event queue
├── cancellation.py              # Thread-local run ids + RunCancelled
├── config.py                    # LLM configs, TOOL_LIST (38), token budgets, PROJECT_VENV_ENABLED, UV_BIN
├── SOUL.md                      # Agent identity + tooling guidance (uv, project=, web pagination, display_doc)
├── MEMORIES.md                  # Agent memories (bundled default)
├── token_budget.py              # estimate_tokens, trim_to_budget, compute_request_timeout
├── activity_log.py              # Structured event logging (tool_call, llm_usage, chat_start, exec, etc.)
├── daily_digest.py              # Rolling 3-day digest
├── model_tracker.py             # set_current_tool, set_state, status server data source
├── status_server.py             # separate process: /status, /dashboard, /digest, /requests on 7861
├── tools/
│   ├── api_tools.py             # web_search, web_url_fetch (with pagination), stock_price (port 8080 SSE)
│   ├── sanitizer.py             # Prompt-injection defense
│   ├── code_interpreter.py      # Fresh Python subprocess per call (1h max), llm_call() bridge
│   ├── exec_tool.py             # Shell + per-project venv (uv) + start_new_session + killpg
│   ├── llm_bridge.py            # HTTP bridge → qwen3.6-27b-linux → qwen3.5 fallback
│   ├── self_edit_tools.py       # update_soul, update_heartbeat
│   ├── project_tools.py         # CRUD + rename_project + update_project + project_apply_patch
│   ├── display_tools.py         # display_doc + thread-keyed display hook
│   ├── notification_tools.py    # request_user, view_requests, resolve_request
│   └── git_autocommit.py        # Auto-commit DATA_DIR git repo
├── pipeline/
│   ├── models.py                # PipelineState, StageState
│   ├── orchestrator.py          # State machine + file-based pipeline lock
│   ├── stage_runner.py          # Build prompt, run agent, detect part-completion
│   ├── evaluator.py             # Programmatic + LLM acceptance evaluation
│   ├── pipeline_tools.py        # start_pipeline, start_trading_pipeline, pipeline_status
│   └── stages/                  # Editable markdown stage instructions (startup + trading)
├── compaction/                  # OpenClaw-style three-tier strategy (truncate tools → summarize → trim)
│   ├── __init__.py              # maybe_compact entry point (trigger at ~170k estimated)
│   ├── estimator.py             # select_tier, _estimate_tool_result_reduction
│   ├── compactor.py             # Summarize old turns, keep recent verbatim, tool-pair-aware
│   ├── summarizer.py            # LLM summarization
│   └── checkpoint.py            # Compaction checkpointing
├── heartbeat/
│   └── heartbeat_runner.py      # 1h interval, HEARTBEAT_OK suppression
├── scheduler/
│   ├── models.py                # Task pydantic model
│   ├── task_queue.py            # JSON-backed queue
│   ├── scheduler_tools.py       # schedule/list/complete/cancel/pause/resume tools
│   └── checkpoint.py            # Long-running task state
├── docker/
│   ├── Dockerfile               # python:3.12-slim + Qwen-Agent[gui] + Chainlit + aiosqlite + uv (separate layer)
│   ├── docker-compose.yml       # Security hardening, bind mount, env vars
│   ├── run.sh                   # Fast foreground run (no rebuild) + Docker Desktop wait
│   ├── restart.sh               # Down + build + foreground up (for code/.env changes)
│   └── shutdown.sh              # Compose down
└── scripts/
    └── cleanup_data.py          # Batch reorg script (dry-run default)

qwen_agent/                      # vendored; locally modified:
├── llm/oai.py                   # vLLM tool/stream + multimodal serialization + usage capture
└── agents/fncall_agent.py       # MAX_LLM_CALL_PER_RUN wrap-up (budget exhaustion → part-completion)
```

## Configuration (docker-compose.yml env)

| Env Var | Default | Purpose |
|---|---|---|
| `VLLM_BASE` | `http://192.168.4.66:8000/v1` | Primary + backup endpoint |
| `PRIMARY_MODEL_CONCURRENCY` | `15` | In-flight cap on primary model before chat falls through to 397B |
| `LLM_CALL_MODEL` / `LLM_CALL_BASE` | *(unset)* | Legacy override to pin `llm_call()` to one model/endpoint |
| `TOOLS_API_BASE` | `http://host.docker.internal:8080` | Web tools server |
| `DATA_DIR` | `/app/data` | Bind-mounted writable volume |
| `HEARTBEAT_INTERVAL_SECONDS` | `3600` | 1 hour |
| `QWEN_AGENT_MAX_LLM_CALL_PER_RUN` | `100` | Max tool iterations per agent run; budget-pressure note + graceful wrap-up |
| `QWEN_AGENT_DEFAULT_WORKSPACE` | `/app/data/workspace` | Qwen-Agent RAG workspace |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | *(from `.env`)* | Optional, for HF model downloads |
| `HF_HOME` | `/app/data/hf_cache` | HF cache on bind mount (survives rebuilds) |
| `CHAINLIT_AUTH_SECRET` | *(from `.env` or dev fallback)* | Chainlit session signing |
| `DATABASE_URL` | `sqlite+aiosqlite:////app/data/chat.db` | Chainlit data layer connection string |
| `STATUS_SERVER_PORT` | `7861` | 0 to skip starting the status server in-process |
| `TZ` | `America/Los_Angeles` | Pacific |
| `PROJECT_VENV_ENABLED` | `true` | (new) `exec(project=)` auto-creates + activates per-project `.venv` |
| `UV_BIN` | `uv` | (new) UV binary path |

## Token budgets & timeouts

| Layer | Limit | Mechanism |
|---|---|---|
| Conversation context (target) | 200,000 tokens (`MAX_CONTEXT_TOKENS`) | `maybe_compact` triggers above ~170k (reserve 30k); model context window is 256k |
| Compaction recent turns kept verbatim | 2 (`COMPACTION_RECENT_TURNS_PRESERVE`) | Older turns summarized via LLM |
| Per-tool-result max share | 30% of context (`COMPACTION_TOOL_RESULT_MAX_SHARE`) | Truncate before summarize |
| Per-tool-result hard cap | 40,000 chars (`COMPACTION_TOOL_RESULT_HARD_CAP`) | Beyond this, always truncate |
| Tool outputs (sanitizer) | 16,000 tokens (`MAX_TOOL_OUTPUT_TOKENS`) | `sanitize_web_content` truncates |
| Code interpreter output | 16,000 tokens (`MAX_CODE_OUTPUT_TOKENS`) | `truncate_output` with corrective marker |
| Code interpreter timeout | 10 min – 1 hour (`MIN_TIMEOUT` / `MAX_TIMEOUT=3600`) | Dynamic scale on code size |
| exec timeout | 2 min default, 1 hour max | `DEFAULT_TIMEOUT=120`, `MAX_TIMEOUT=3600` |
| LLM request timeout | 10–30 min | `compute_request_timeout(messages)` |
| Tool-call loop | 100 iterations | `QWEN_AGENT_MAX_LLM_CALL_PER_RUN`; budget-pressure note injected near limit + graceful wrap-up |

## System prompt assembly

```
load_system_message():
  SOUL.md (DATA_DIR override or bundled default)
  + "## Your Memories (auto-loaded from MEMORIES.md)" + MEMORIES.md
  + SYSTEM_PROMPT_SUFFIX  (token-efficiency reminder)

Main chat agent only (Chainlit):
  + session_metadata()    (frozen once per conversation, format: 2026-06-15 03:30pm PDT)
  Cached in cl.user_session["_frozen_metadata"] on first turn. Both agents'
  system_message are updated with this frozen value before each agent.run() call.
  The timestamp never changes mid-conversation → vLLM KV prefix cache stays stable.
  The Gradio/REPL path still uses the one-time boot-time snapshot.
```

Background sessions (cron, heartbeat) use the metadata-free base to preserve the vLLM prefix cache.

## Comparison to OpenClaw / Hermes Agent

### What this agent has that OpenClaw doesn't

| Capability | Description |
|---|---|
| **`llm_call()` / `llm_batch()` inside code execution** | LLM access from inside `code_interpreter` Python loops with the primary→backup chain. |
| **Pipeline orchestrator (×2)** | Startup-builder and trading-research, code-driven with acceptance evaluation, retry, and stage artifact passing. |
| **Self-scheduling** | Agent creates its own cron jobs and pipeline stages. |
| **Self-modification** | `update_soul`, `update_heartbeat`, `add_memory` — agent edits its identity, checklist, and memory at runtime. |
| **Live dashboard** | Separate-process status server reading file-based state. |
| **Multi-model routing with fallback** | Semaphore-based primary→397B routing so background work doesn't queue chat. |
| **Context compaction (3-tier)** | Tool-result truncation → LLM summarization → trim fallback. Tool-pair-aware (keeps assistant tool_call paired with its result). |
| **Background-task chat notifier** | Hermes-style: notifier loop injects `[system event]` notices into the originating chat session. |
| **`display_doc`** | Render files out-of-band to the user without spending agent context. |
| **Streaming bridge with adaptive batching** | Stream-token batching + tool-step apply-once guard + pre-footer flush. |
| **WAL data layer** | SQLite tuned for concurrent step persistence. |
| **Per-project venv (UV)** | (Phase A, pending deploy) per-project dependency isolation with UV. |
| **File edit modes + structured patches** | `project_write_file` write/append/edit + `project_apply_patch`. |
| **Daily digest** | LLM-summarized rolling 3-day activity. |

### Where OpenClaw / Hermes still have things we don't

| Capability | Source | Impact |
|---|---|---|
| **On-resume context restoration** | OpenClaw replays compacted history; Hermes uses `session_search` + a frozen memory snapshot | Our `on_chat_resume` drops all tool/reasoning steps, so the agent "forgets" gathered context across page reloads. See "Known shortcomings". |
| **Background process registry** | OpenClaw auto-backgrounds after 10s with cancellation | We have `start_new_session` + killpg but no first-class registry; a `server &` survives but isn't tracked. |
| **Inner sandbox isolation** | OpenClaw path denylists + seccomp + command-approval | We rely on the outer Docker container only. |
| **Tool approval system** | OpenClaw deny/allowlist/full tiers with interactive prompts | Every tool executes without review. |
| **Push notifications** | Neither has it; ours is file-only | Agent can't actually reach the user when it says "I'll let you know." |
| **Real-time process streaming** | OpenClaw stdout/stderr callbacks during execution | Agent sees `exec` output only after it finishes; `code_interpreter` has a per-call progress hook surfaced to the chat step but it's coarse. |
| **Multi-user / multi-agent** | Both have richer session/auth | Single-user LAN only. |
| **Per-pipeline isolation (containers)** | OpenClaw can run per-task containers | We use one container + (soon) per-project venv. |

## Roadmap / pending work

### Approved, in progress

- **Phase B — parallel pipelines (cap 3).** Make the global pipeline lock **per-project**, raise the cron `ThreadPoolExecutor(max_workers=1)`, scope `_background_work_lock` so heartbeat and cron don't fully serialize, add `MAX_PARALLEL_PIPELINES` env var, audit `task_queue` for concurrent access. Prerequisite: Phase A deployed (so parallel `pip install`s don't corrupt a shared global env).

### Approved, pending design

- **Resume context restoration.** Two candidates were researched in this session: OpenClaw-style replay+compact (reconstruct tool-call / result message pairs from `chat.db` steps, let `maybe_compact` manage size) vs Hermes-style session_search tool (lean resume + on-demand recall). Decision still open. Either restores continuity that the current intentional drop on resume costs.

### Useful next steps

- **`code_interpreter` project-scoped mode** — optionally use the project's `.venv` when a `project=` arg is supplied (parallels the new `exec` behavior). Today `code_interpreter` is global-env only.
- **Push notifications** — Slack/email/ntfy.sh integration so `request_user` actually reaches the user. Currently file-only.
- **Tool approval / destructive-op gating** — at least log all destructive calls (`delete_*`, `pause_*`, etc.); ideally queue for user approval.
- **Lock timeout / watchdog** — `_background_work_lock` and `_primary_model_lock` acquire without timeout. A hung LLM call can block all background work; add `acquire(timeout=N)` or a watchdog.
- **Background process registry** — first-class long-running process management so a dev server started with `&` is trackable, listable, cancellable.
- **`fork_pipeline` / cross-project knowledge** — derive a new project from a completed one, copying artifacts.
- **Multi-language `exec` / `code_interpreter`** — JS/Go/shell beyond Python.
- **Real-time `exec` output streaming to chat** — currently buffered until process exit.

## Known shortcomings (current state)

- **Resume drops tool context.** `on_chat_resume` rebuilds history from user + assistant text only, intentionally dropping all 200k+ chars of tool results and reasoning per long thread. The agent loses everything it gathered whenever the page reloads. Roadmap above has the fix candidates.
- **Pipelines serialize globally.** A single file lock + `_background_work_lock` + `max_workers=1` cron executor means only one pipeline runs at a time across the system. Phase B fixes this.
- **`code_interpreter` shares the global env.** Per-project venv isolation only applies to `exec`. Project deps installed via `exec(project=X, "uv pip install …")` won't be visible to `code_interpreter` in the same project unless the interpreter is also scoped (planned).
- **`_background_work_lock` has no timeout.** A hung LLM call can block heartbeat + cron indefinitely. `clear_lock_on_startup()` only helps after restart.
- **`LockingAgent.run()` releases its semaphore on generator exhaustion.** Early break by a caller = held lock until the generator object is GC'd. Should be a context manager.
- **vLLM contention is real-but-bounded.** Both primary and 397B share one vLLM on one GPU. The 397B request is dispatched correctly when the primary saturates, but vLLM's scheduler can deprioritize it under GPU pressure. User-stated tolerance: ~16 concurrent at marginal cost.
- **Pipeline stage truncation loop (residual).** Long stages occasionally produce output that exceeds the per-call output cap and gets truncated; the `part-completion` mechanism handles most cases but not all.
- **vLLM thinking-token leakage** (model-dependent). Fix requires server-side `--chat-template-kwargs '{"enable_thinking": false}'` or top-level `chat_template_kwargs` in the request (not in `extra_body`). See `feedback_thinking_tokens.md`.
- **Invalid JSON tool arguments.** The model occasionally emits malformed JSON (escaped quotes in long `schedule_task` descriptions). qwen-agent retries internally but it's a recurring failure mode.
- **Single-user.** No auth, no session isolation beyond Chainlit's own. Don't expose beyond LAN.

## Recent fixes (this session — most need a rebuild to deploy)

| Fix | Files | Status |
|---|---|---|
| WAL data layer (QueuePool exhaustion → streaming lag) | `chat_data_layer.py` | Deployed |
| Streaming-lag guard: finished tool steps no longer re-`update()`d every chunk | `chat_app.py` (`_resolved_tool_steps`) | Deployed |
| Footer-spliced-mid-sentence: flush batched tail before appending the per-turn footer | `chat_app.py` (`flush_pending_text`) | Deployed |
| `web_url_fetch` pagination (`offset`, `max_chars`) + content extraction + pagination hint | `api_tools.py`, `SOUL.md` | Deployed |
| `code_interpreter` rewrite to stateless subprocess (was wedging on persistent kernel) | `code_interpreter.py` | Deployed |
| `exec` + `code_interpreter` `start_new_session=True` + killpg | `exec_tool.py`, `code_interpreter.py` | Deployed |
| Vision/multimodal user messages | `oai.py`, `chat_app.py` | Deployed |
| `display_doc` tool + storage client + persisted reload | `display_tools.py`, `chat_storage.py`, `chat_app.py` | Deployed |
| `rename_project`, `update_project` tools | `project_tools.py` | Deployed |
| Token usage capture + per-turn `📊 max ctx N / out · chat high-water` footer | `oai.py`, `chat_app.py` | Deployed |
| Background-task completion notifier → originating chat session | `task_notify.py`, `chat_app.py`, `chat_origin.py` | Deployed |
| Notifier injection role=`user` `[system event]` (vLLM rejects mid-thread role=`system`) | `chat_app.py` | Deployed |
| Per-project `.venv` + `uv` default package manager | `exec_tool.py`, `config.py`, `Dockerfile`, `SOUL.md` | **Pending rebuild** |
| `run.sh` foreground script with Docker Desktop wait | `sandbox_agent/docker/run.sh` | Shipped |

## Data directory layout (`~/sandbox_agent_data/`)

```
~/sandbox_agent_data/
├── MEMORIES.md                # Auto-loaded into system prompt
├── SOUL.md                    # If agent has edited it
├── HEARTBEAT.md               # Agent-edited heartbeat checklist
├── tasks.json                 # Task queue
├── agent_requests.json/.md    # Structured pending/resolved requests
├── activity.jsonl             # Structured event log (chat, tool, llm_usage, …)
├── chat.db                    # Chainlit persistence (SQLite, WAL)
├── chat.db-wal / chat.db-shm  # WAL sidecar files
├── .cl_elements/              # Chainlit element blobs (images, PDFs) → /cl-elements/{user}/{id}
├── chat_logs/YYYY-MM-DD.md    # Daily conversation logs
├── digest/                    # Daily + rolling-3-day LLM-summarized digest
├── projects/<slug>/
│   ├── .project.json          # name, description, created_at, updated_at
│   ├── .venv/                 # (NEW) per-project venv, auto-created via `uv venv` on first project-exec
│   ├── README.md, TODO.md     # Default scaffold
│   ├── research/ business/ product/ mvp/ pipeline/   # Pipeline-stage outputs
│   └── …                      # Anything else the agent writes
├── scratch/                   # code_interpreter temp scripts (auto-deleted)
├── checkpoints/               # Task checkpoints
└── hf_cache/                  # HuggingFace cache (survives rebuilds)
```

## Docker security

- `security_opt: no-new-privileges:true` — prevents privilege escalation
- `cap_drop: ALL` — drops all Linux capabilities
- `tmpfs: /tmp:size=10G` — temp files in RAM
- Non-root `agent` user
- Bind mount only on `~/sandbox_agent_data/`
- LLM bridge secured with per-startup auth token + 1MB request size limit
- `sanitize_web_content` strips chat-template tokens, invisible chars, delimiter injection

## Memory references

Persistent cross-session memory at `~/.claude/projects/-Users-johnzhu-code-Qwen-Agent/memory/`:

- `reference_infrastructure.md` — LAN IPs and services
- `feedback_thinking_tokens.md` — Disabling Qwen 3.5 thinking on vLLM/Ollama
- `feedback_readme.md` — "the readme" means *this* file (`README_jz.md`)
- `project_vllm_contention.md` — Single-GPU vLLM contention pattern
- `project_llm_call_concurrency.md` — Bridge serialization + llm_batch ignoring primary_model_lock
- `project_pipeline_architecture_issues.md` — 7 load-bearing causes of pipeline slowness
- `project_pipeline_bug.md` — `start_pipeline` ignoring `CreateProject` return when project exists
- `project_deferred_bugs.md` — Deferred: lock timeout watchdog + generator lock release
- `project_future_work.md` — Token-based loop limits, user notifications, git diffs on all writes
