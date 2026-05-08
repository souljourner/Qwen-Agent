# Sandbox Agent — Knowledge Transfer Document

A sandboxed, self-scheduling Qwen Agent built on [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), inspired by [OpenClaw](https://docs.openclaw.ai/) patterns.

## Quick Start

```bash
# Build and run (Gradio WebUI on :7860, dashboard on :7861)
cd sandbox_agent/docker && docker compose up --build

# Terminal REPL mode
docker compose run --rm -it agent --mode repl

# Run tests
cd /path/to/Qwen-Agent && source .venv/bin/activate && pytest tests/sandbox_agent/ -v
```

## Infrastructure

| Service | Host | Model | Role |
|---|---|---|---|
| **vLLM** | 192.168.4.66:8000 | qwen3.6-27b-linux | Primary — main conversation + all background tasks; concurrency 15 |
| **vLLM** | 192.168.4.66:8000 | qwen3.5 (397B MoE) | User chat fallback when every primary slot is in use; also backup for `llm_call()` |
| **Tools API** | localhost:8080 | — | Web search (Brave), URL fetch, stock prices via SSE |
| **Gradio** | localhost:7860 | — | Chat WebUI |
| **Dashboard** | localhost:7861 | — | Live activity monitor, digest, agent requests |
| **Data** | ~/sandbox_agent_data/ | — | Bind-mounted volume, visible in Finder |

## Architecture

```
Docker Container (read_only, no-new-privileges, cap_drop: ALL)
├── Main Lane (Gradio WebUI) → primary qwen3.6-27b-linux (15 concurrent slots); falls back to qwen3.5 (397B) when all slots in use
├── Heartbeat Lane (1 hour interval) → checks HEARTBEAT.md
├── Cron Lane (60s poll) → executes scheduled tasks + pipeline stages on the primary
├── Pipeline Orchestrator → code-based 6-stage project builder
├── Status Server (separate process, port 7861) → reads activity.jsonl from disk
└── LLM Bridge (HTTP server on localhost) → vLLM qwen3.6-27b-linux → qwen3.5 (397B) for llm_call()
```

### Model Routing

- **User sends message**: tries to grab one of the 15 primary slots (non-blocking semaphore). If every slot is taken, falls back to the 397B qwen3.5.
- **Background tasks**: always use the primary (blocking acquire — wait for a free slot). The 397B is reserved for user chat fallback only.
- **`llm_call()` inside code_interpreter**: primary qwen3.6-27b-linux → 397B qwen3.5 fallback chain on vLLM (same as chat). Empty-completion retry once per model before falling through.

### Bounded-Semaphore Primary, Single-Mutex Background

The primary lock is a **`BoundedSemaphore(PRIMARY_MODEL_CONCURRENCY)`** (default 15). User chat acquires non-blockingly: if any of the 15 slots is free, the user runs on the primary; otherwise the user is routed to the 397B backup so chat is never queued behind background work. Background callers acquire blocking and wait for a free slot.

Setting `PRIMARY_MODEL_CONCURRENCY` higher than what vLLM actually serves on the primary will move queueing from the client to the server (slow chat) instead of fanning out to the backup. Tune both sides together. A separate `_background_work_lock` (plain `Lock`) ensures the heartbeat and cron lanes don't run two background sessions at the same time — that mutex is about session ordering, not model concurrency.

## Tools (29 registered)

| Category | Tools | Description |
|---|---|---|
| **Web** | `web_search`, `web_url_fetch`, `stock_price` | Call port 8080 API via SSE. All outputs sanitized. Brave Search rate-limited to 1 req/2s. |
| **Code** | `code_interpreter` | Local Jupyter kernel (subprocess). Has `llm_call(prompt, system="", think=False)` via the qwen3.6-27b-linux → qwen3.5 chain. Pre-configured `DATA_DIR`, `PROJECTS_DIR` path vars. Output capped at 4k tokens. |
| **Scheduling** | `schedule_task`, `list_tasks`, `complete_task`, `cancel_task`, `pause_task`, `resume_task`, `update_task_checkpoint` | JSON-backed task queue. Active tasks in `tasks.json`, completed/cancelled archived to separate files. |
| **Pipeline** | `start_pipeline`, `pipeline_status`, `list_pipelines` | Code-based 6-stage project builder (see Pipeline section below). |
| **Self-edit** | `update_soul`, `update_heartbeat` | Patch a section or append a line (not full file replacement). Auto-committed to git. |
| **Memory** | `read_memories`, `add_memory` | MEMORIES.md with dated entries in 4 sections. Auto-loaded into system prompt. |
| **Projects** | `create_project`, `list_projects`, `delete_project`, `project_write_file`, `project_read_file`, `project_list_files`, `project_delete_file`, `project_apply_patch`, `move_file`, `delete_file` | Persistent file workspaces. `project_write_file` supports write/append/edit modes. `project_apply_patch` applies OpenClaw-style structured patches (add/update/delete files with context-based matching). |
| **Chat** | `list_chat_logs`, `read_chat_log` | Daily markdown logs at `DATA_DIR/chat_logs/YYYY-MM-DD.md`. |
| **Notifications** | `request_user`, `view_requests`, `resolve_request` | Structured JSON requests with pending/resolved status. Deduplication on pending subjects. |

## Pipeline Orchestrator

A Python code-based 6-stage pipeline for building startup projects. Each stage runs independently with a clean LLM context, reading artifacts from previous stages. Flow is controlled by Python code, not prompts.

### Stages

| # | Stage | Input | Output | Description |
|---|---|---|---|---|
| 1 | Market Research | idea description | `research/market-research.md` | Market size, competitors, customers, timing, adjacent markets |
| 2 | BRD | market research | `business/brd.md` | Branding, legal, scalability, operations, finance, personnel |
| 3 | PRD | research + BRD | `product/prd.md` | User stories, MVP scope, technical requirements, distribution |
| 4 | VC Pitch | all above | `business/vc-pitch.md` | Elevator pitch + longer warm-contact pitch |
| 5 | MVP | PRD | `mvp/` directory | Frontend, backend, tests, README, DB scripts |
| 6 | Review | all artifacts | `pipeline/review.md` | Learnings, status update, instruction improvement suggestions |

### How It Works

```
start_pipeline(name="pet-ai", description="AI pet health monitoring for busy pet owners...")
  → validates idea clarity → creates project → schedules stage 1

cron_loop detects "pipeline:" prefix task
  → acquires lock → loads stage instructions (markdown) → loads previous artifacts
  → runs agent on primary model → evaluates output (programmatic + LLM check)
  → PASS: schedules next stage | FAIL: adds notes, retries (up to 5x)
```

### Stage Acceptance Evaluation

After each stage, the evaluator runs:
1. **Programmatic checks**: artifact exists, non-empty, required sections present
2. **LLM quality check**: reads output and judges completeness, depth, accuracy
3. **Decision**: pass → next stage, fail + retries left → add feedback notes and retry, fail + 5 attempts → proceed with `completed-no-more-attempts` or `failed-no-more-attempts`

### Job States

`scheduled` → `running` → `completed` / `part-completion` / `failed`

- **`part-completion`**: ran out of tokens/tool calls (MVP stage). Notes track progress, next attempt continues.
- **`completed-no-more-attempts`**: produced output but never passed quality check after 5 tries. Proceeds anyway.
- **`failed-no-more-attempts`**: couldn't produce any artifact after 5 tries. Notifies user, proceeds.

### Self-Improvement

Stage instructions live in editable markdown files at `sandbox_agent/pipeline/stages/`. Stage 6 (Review) evaluates instruction quality and suggests improvements. Updated instructions apply to subsequent pipeline runs.

### Rerun

`start_pipeline` on a completed project resets all stages. Previous artifacts are preserved — instructions tell the agent to read and improve existing files rather than start from scratch.

## Key Files

```
sandbox_agent/
├── config.py                 # LLM configs, tool list, token budgets, load_system_message()
├── main.py                   # Entry point: LockingAgent, run_on_best_available, cron_loop, create_agent
├── SOUL.md                   # Agent identity + token efficiency rules (bundled default)
├── MEMORIES.md               # Agent memories (bundled default)
├── token_budget.py           # estimate_tokens, trim_to_budget, compute_request_timeout, truncate_output
├── activity_log.py           # Structured event logging to activity.jsonl + in-memory state
├── chat_logger.py            # Daily chat logs + list/read tools
├── daily_digest.py           # Rolling 3-day digest, updated after each cron task
├── status_server.py          # Separate process: /status, /dashboard, /digest, /requests on port 7861
├── tools/
│   ├── api_tools.py          # web_search, web_url_fetch, stock_price (port 8080 SSE)
│   ├── sanitizer.py          # Prompt injection defense: strip tokens, invisible chars, delimiters
│   ├── code_interpreter.py   # Local Jupyter kernel with llm_call() bridge
│   ├── llm_bridge.py         # HTTP bridge → qwen3.6-27b-linux → qwen3.5 on vLLM for llm_call()
│   ├── self_edit_tools.py    # update_soul (section patch), update_heartbeat, read/add memory
│   ├── project_tools.py      # Project workspace CRUD + move_file, delete_file, delete_project
│   ├── notification_tools.py # request_user, view_requests, resolve_request
│   └── git_autocommit.py     # Auto-commit file changes in DATA_DIR git repo
├── pipeline/
│   ├── models.py             # PipelineState, StageState pydantic models
│   ├── orchestrator.py       # State machine: advance, lock, stage definitions
│   ├── stage_runner.py       # Build prompts, load artifacts, run agent, detect part-completion
│   ├── evaluator.py          # Programmatic checks + LLM acceptance evaluation
│   ├── pipeline_tools.py     # start_pipeline, pipeline_status, list_pipelines
│   └── stages/               # Editable markdown instruction files (6 stages + acceptance criteria)
├── heartbeat/
│   ├── heartbeat_runner.py   # Isolated sessions, HEARTBEAT_OK suppression, background work lock
│   └── HEARTBEAT.md          # Default checklist
├── scheduler/
│   ├── models.py             # Task pydantic model (cron, project, checkpoint, backoff)
│   ├── task_queue.py         # JSON-backed queue with 3 files (active/completed/cancelled)
│   ├── scheduler_tools.py    # schedule/list/complete/cancel/pause/resume/checkpoint tools
│   └── checkpoint.py         # Task state persistence for long-running work
├── scripts/
│   └── cleanup_data.py       # Batch file reorganization script (standalone, dry-run default)
└── docker/
    ├── Dockerfile            # Python 3.12, Qwen-Agent[gui], Jupyter, beautifulsoup4, lxml
    └── docker-compose.yml    # Security-hardened, bind mount, env vars
```

## Configuration (docker-compose.yml environment)

| Env Var | Value | Purpose |
|---|---|---|
| `VLLM_BASE` | `http://192.168.4.66:8000/v1` | Primary + backup models (qwen3.6-27b-linux, qwen3.5) |
| `PRIMARY_MODEL_CONCURRENCY` | `15` | Number of in-flight requests allowed against the primary before chat falls through to the 397B backup. Match the server-side concurrency. |
| `LLM_CALL_MODEL` | *(unset)* | Legacy override to pin llm_call() to a single model. Default: chain qwen3.6-27b-linux → qwen3.5. |
| `LLM_CALL_BASE` | *(unset)* | Legacy override for llm_call() endpoint. Default: `VLLM_BASE`. |
| `TOOLS_API_BASE` | `http://host.docker.internal:8080` | Web tools |
| `DATA_DIR` | `/app/data` | Writable volume (bind-mounted to ~/sandbox_agent_data/) |
| `HEARTBEAT_INTERVAL_SECONDS` | `3600` | 1 hour |
| `QWEN_AGENT_MAX_LLM_CALL_PER_RUN` | `50` | Max tool iterations per agent run |
| `GRADIO_SERVER_TIMEOUT` | `3600` | 1 hour (prevents connection timeout on long tasks) |
| `TZ` | `America/Los_Angeles` | Pacific time |
| `QWEN_AGENT_DEFAULT_WORKSPACE` | `/app/data/workspace` | Qwen-Agent RAG workspace |

## Token Budget & Timeouts

| Layer | Budget/Limit | Mechanism |
|---|---|---|
| Conversation context | 200k tokens | `trim_to_budget()` drops oldest turns, preserves system prefix |
| Tool outputs | 8k tokens | Sanitizer truncates |
| Code interpreter output | 4k tokens | `truncate_output()` with corrective message |
| Request timeout | 10-30 min (dynamic) | Linear scale based on payload size |
| Code interpreter timeout | 10 min - 2 hr (dynamic) | Scales with code size |
| `llm_call()` prompt | 200k tokens / 1MB | Bridge truncates |
| Tool call loop | 50 iterations | `QWEN_AGENT_MAX_LLM_CALL_PER_RUN` |

## System Prompt Assembly

```
load_system_message():
  SOUL.md (from DATA_DIR or bundled default)
  + "## Your Memories (auto-loaded from MEMORIES.md)" + MEMORIES.md content
  + SYSTEM_PROMPT_SUFFIX (token efficiency reminder)

Main agent only:
  + session_metadata() (date, time, location: San Mateo CA, US/Pacific)
```

Background sessions use the base system message without metadata (static for KV cache).

## Capabilities vs OpenClaw Comparison

### What We Have That OpenClaw Doesn't

| Capability | Description |
|-----------|-------------|
| **Persistent Jupyter kernel** | State (variables, imports, DataFrames) survives across code_interpreter calls. OpenClaw starts fresh every command. |
| **`llm_call()` inside code execution** | Agent calls a background LLM from within Python loops for per-item reasoning. 2-model fallback chain (qwen3.6-27b-linux → qwen3.5) with empty-completion retry. OpenClaw has no LLM access inside execution. |
| **Pipeline orchestrator** | Automated 6-stage startup builder (Market Research → BRD → PRD → VC Pitch → MVP → Review) with acceptance evaluation, retry logic, incremental file writing, and stage-to-stage artifact passing. OpenClaw has no equivalent. |
| **Self-scheduling** | Agent creates its own cron jobs, schedules pipeline stages, runs periodic heartbeat checks autonomously. |
| **Self-modification** | `update_soul`, `update_heartbeat`, `add_memory` — agent edits its own identity, checklist, and persistent memory at runtime. OpenClaw's AGENTS.md is static. |
| **Real-time monitoring dashboard** | Separate process (port 7861) showing active model, current task, tool calls, digest, and pending requests. |
| **Multi-model routing with fallback** | Bounded-semaphore primary → 397B backup routing so background tasks don't block user chat. `llm_call()` uses the same qwen3.6-27b-linux → qwen3.5 chain with empty-completion retry. |
| **Context compaction** | OpenClaw-inspired three-tier strategy (tool truncation → LLM summarization → trim) adapted for Qwen-Agent message format. |
| **Notification queue** | `request_user` with structured pending/resolved JSON requests and markdown view. |
| **Git autocommit** | Every project file write is auto-committed to the data directory git repo. |
| **Daily digest** | LLM-summarized rolling 3-day activity log of all agent work. |
| **File edit tools** | `project_write_file` with write/append/edit modes + `project_apply_patch` for OpenClaw-style multi-file structured patches. |

### What OpenClaw Has That We Don't

| Capability | Description | Impact |
|-----------|-------------|--------|
| **General shell/exec tool** | Runs any shell command (`npm install`, `cargo build`, start servers). We only have Python via Jupyter. | Can't build non-Python apps or run build tools. |
| **Background process management** | Auto-backgrounds after 10s, tracks processes in registry, cancellable. | Can't start dev servers while doing other work. |
| **Inner sandbox isolation** | Docker sandbox with path denylists, env sanitization, seccomp/AppArmor, command approval system. | Code interpreter has full container filesystem access. Prompt injection → code exec → data exfiltration is possible. |
| **Tool approval system** | Multi-tier (`deny`/`allowlist`/`full`) with interactive prompts before dangerous operations. | Every tool call executes without review. |
| **Push notifications** | Not in OpenClaw either, but our `request_user` is file-only — no email, Slack, webhook, desktop alert. | Agent can't actually reach the user when it says "I'll let you know." |
| **Conversation persistence** | OpenClaw has compaction checkpoints and transcript repair for crash recovery. | Gradio crash = chat history gone. |
| **Multi-language execution** | Any language in Docker image (JS, Go, Rust, shell). | Python only. |
| **Real-time output streaming** | stdout/stderr callbacks during execution. | Agent sees output only after code finishes. |
| **Multi-user / multi-agent** | Session isolation, agent routing, multi-agent orchestration. | Single user only. |
| **Token/cost tracking** | Not built-in to either, but OpenClaw has richer observability hooks. | No visibility into token consumption or compute cost. |

### Architecture Comparison

| Aspect | **OpenClaw** | **Sandbox Agent** |
|--------|-------------|-------------------|
| Code execution | Docker + ephemeral shell commands | Persistent Jupyter kernel (subprocess) |
| Languages | Any (shell) | Python only |
| State persistence | None per command | Yes across calls |
| Output limit | 200k chars | 64k chars (raised from 16k) |
| Timeout | Overall + no-output; auto-background at 10s | Dynamic 10min–2hr; OS signal alarm |
| Security model | Docker sandbox + path denylists + env sanitization + seccomp + command approval | Outer Docker container only |
| LLM inside code | None | `llm_call()` with 2-model fallback + empty-completion retry |
| File tools | write, edit, apply_patch (workspace-scoped) | write/append/edit, apply_patch (project-scoped) |
| Long-running work | Auto-background + process registry | Blocks until done or timeout |
| Streaming | Real-time callbacks | Batch after execution |

## Roadmap to Production-Grade Agent

### High Priority

1. **General shell/exec tool** — Add an `exec` tool that runs arbitrary shell commands in the container. Required for building non-Python apps, running build tools, starting servers. Model after OpenClaw's exec with timeout and output limits.

2. **Push notifications** — Slack webhook, email, or ntfy.sh integration so the agent can actually alert the user when tasks complete, pipelines fail, or approvals are needed.

3. **Conversation state persistence** — Save messages to disk on every turn so the agent can resume after Gradio crash or container restart. OpenClaw does this with transcript files.

4. **Silent LLM failure detection** — Add defensive logging when the model returns 200 OK but produces empty/unparseable output. Auto-retry once, then notify user. Currently causes invisible hangs.

### Medium Priority

5. **Inner sandbox isolation** — Path denylists for code interpreter (block `/app/`, `/etc/`, env vars). Prevent prompt injection → code execution → data exfiltration chain.

6. **Tool approval system** — Gate destructive operations (`delete_project`, `delete_file`, scheduled tasks that modify data). At minimum, log all destructive calls; ideally, queue for user approval.

7. **Token/cost tracking** — Count tokens per request, track cumulative usage, show "context X% full" in dashboard. Essential for understanding compute costs.

8. **Background process management** — Allow code interpreter to start long-running processes (dev servers, build watchers) without blocking the agent. Track and cancel from dashboard.

9. **Cross-project knowledge** — `fork_pipeline` tool to derive new projects from completed ones, copying artifacts. Currently pipelines can't read from other projects.

10. **Lock timeout / watchdog** — Add timeout to `_background_work_lock` and `_primary_model_lock`. Detect hung LLM calls and recover instead of blocking all background work forever.

### Lower Priority

11. **Multi-language code execution** — Support JS/shell alongside Python in code interpreter, or add a separate `exec` tool (see #1).

12. **Real-time output streaming** — Stream code interpreter output to the dashboard as it executes, rather than batch after completion.

13. **Multi-user support** — Session isolation, authentication, per-user project separation. Required if exposing beyond localhost.

14. **Multi-agent orchestration** — Spawn sub-agents for parallel research, divide pipeline stages across agents. Currently everything is single-threaded.

15. **Generator lock release fix** — Redesign `LockingAgent.run()` to not rely on generator exhaustion for lock release. Use context manager pattern instead.

## Known Issues & Deferred Work

### Active Issues
- **vLLM thinking tokens**: Model outputs "Thinking Process: ..." inline. Fix requires vLLM server restart with `--chat-template-kwargs '{"enable_thinking": false}'` or passing `chat_template_kwargs: {"enable_thinking": false}` as a **top-level** field in the API request (not nested in `extra_body`).
- **Context bloat stalls**: When the agent makes many tool calls with large outputs (web_url_fetch), the context grows until vLLM becomes very slow or hangs. Need token-based loop limit.
- **Invalid JSON tool arguments**: The model sometimes generates malformed JSON with escaped quotes in `schedule_task` descriptions, causing tool call failures.
- **Chat hangs silently after vLLM 200 OK**: Agent receives user message, vLLM returns HTTP 200, but no output renders in Gradio and no error is logged. The streaming generator likely gets an empty/unparseable response and silently completes. Needs defensive logging in the chat streaming path.
- **Chat slow/unresponsive during pipeline execution**: Both models (qwen3.6-27b-linux, qwen3.5 397B) share one vLLM server on a single Mac Studio. GPU memory contention can make the 397B backup very slow when a large-context pipeline stage is active on the primary. The semaphore fallback works correctly — the 397B request is sent — but vLLM's scheduler deprioritizes it under GPU pressure. Fix: route backup to a different server.
- **No lock timeout on background tasks**: `_background_work_lock` in `main.py` acquired with no timeout. A hung LLM call blocks heartbeat and cron forever. Needs watchdog thread or `lock.acquire(timeout=N)`.
- **Generator pattern delays lock release**: `LockingAgent.run()` (`main.py:128-150`) yields from a generator. Lock in `finally` only releases when generator is fully consumed. Early break by caller = held lock.
- **Pipeline stage truncation loop**: Pipeline Stage 1 (Market Research) repeatedly fails acceptance because the agent generates documents that exceed the output token limit and get truncated mid-sentence. The evaluator correctly rejects them, but the agent doesn't learn to write shorter or use incremental file writes. Seen on `agent-matchmaker` (2 failures so far).

### Deferred Work (saved in memory)
- **Token-based loop limit**: Replace 50-iteration cap with ~200k token budget check per iteration.
- **User notifications**: Slack/webhook integration for real-time alerts (currently file-based `request_user` only — no push, no desktop alert, no email).
- **Git commit on ALL file writes**: code_interpreter, chat_logger, checkpoints not yet auto-committed.
- **Conversation state persistence**: Save messages to disk so agent can resume after Gradio crash.
- **Disable vLLM thinking**: Needs server-side flag change.

### Fixed (Apr 7, 2026)
- **CRITICAL**: `/models` endpoint crashed — `status_server.py` imported `read_model_status_from_file` but function was `read_status_from_file`
- Unguarded JSON parsing in `task_queue.py` and `orchestrator.py` — corrupt files crashed the system
- 9 unclosed file handles across 7 files (stage_runner, evaluator, project_tools, notification_tools, chat_logger, daily_digest, status_server)
- TOCTOU race in status_server.py `/digest` and `/requests` endpoints
- `start_pipeline` ignored `CreateProject` return when project already existed
- Invalid pipeline task names marked "completed" instead of raising (now raises `ValueError`, caught by cron_loop's except block)
- Silent `except: pass` in `log_turn` — now logs with `logger.debug`

### Operational Notes
- **Stuck tasks on restart**: Startup code resets any `running` tasks to `pending`.
- **Brave Search rate limit**: 2-second minimum between requests (enforced in api_tools.py).
- **`llm_call()` fallback chain**: qwen3.6-27b-linux → qwen3.5 (397B) on vLLM (192.168.4.66:8000). Each model retried once on empty completions before falling through. `think` parameter passed through for models that support it.
- **Dashboard process**: Runs as a separate `multiprocessing.Process` to avoid GIL blocking. Reads all data from files (activity.jsonl, digest, requests) — no shared memory.
- **Daily digest**: After each cron task, Ollama summarizes the tool call record (not the raw result text) into 2-3 sentences for the digest.

## Data Directory Layout (~/sandbox_agent_data/)

```
~/sandbox_agent_data/
├── MEMORIES.md                    # Agent's persistent memory (auto-loaded into system prompt)
├── SOUL.md                        # Agent-edited identity (if modified from bundled default)
├── HEARTBEAT.md                   # Agent-edited heartbeat checklist
├── tasks.json                     # Task queue (all scheduled/completed/recurring tasks)
├── agent_requests.json            # Structured requests (pending/resolved)
├── agent_requests.md              # Human-readable request view
├── activity.jsonl                 # Structured event log (tool calls, cron, chat)
├── chat_logs/
│   └── YYYY-MM-DD.md             # Daily conversation logs
├── digest/
│   ├── YYYY-MM-DD.md             # Daily digest entries
│   └── latest.md                 # Rolling 3-day view
├── projects/
│   └── flatsixai/                # Example project workspace
│       ├── .project.json
│       ├── README.md
│       ├── TODO.md
│       ├── research/             # Research files
│       └── demos/                # Demo artifacts
├── code_interpreter/              # Jupyter kernel files
├── checkpoints/                   # Task checkpoint state
└── workspace/                     # Qwen-Agent RAG workspace
```

## Docker Security

- `read_only: true` — cannot write to host filesystem
- `no-new-privileges: true` — prevents privilege escalation
- `cap_drop: ALL` — drops all Linux capabilities
- `tmpfs /tmp:size=100M` — temp files only
- Non-root `agent` user
- Bind mount only on `~/sandbox_agent_data/`
- LLM bridge secured with per-startup auth token + 1MB request size limit
- Content sanitizer strips chat-template tokens, invisible chars, delimiter injection

## Cron Schedule (current)

| Task | Schedule | Project | Description |
|---|---|---|---|
| `flatsixai-heartbeat-research` | `30 * * * *` (hourly at :30) | flatsixai | Research and advance project |
| `flatsixai-deep-research-sprint` | `0 */2 * * *` (every 2hr) | flatsixai | Deep research sessions |
| `Daily Trading Ideas Report` | `30 9 * * 1-5` (9:30am PT weekdays) | — | Trading report saved to `/data/trading_reports/` |
| Main heartbeat (HEARTBEAT.md) | Every 1 hour (thread, not cron) | — | System health check |

## Memory References

See `~/.claude/projects/-Users-johnzhu-code-Qwen-Agent/memory/` for:
- `reference_infrastructure.md` — LAN IPs and services
- `feedback_thinking_tokens.md` — How to disable Qwen 3.5 thinking (verified methods)
- `project_future_work.md` — Deferred improvements (token limits, notifications, git diffs)
