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
| **vLLM** | 192.168.4.66:8000 | qwen3.5 (MoE) | Main conversation + all background tasks |
| **vLLM** | 192.168.4.66:8000 | qwen3.5-27b | User chat fallback when primary is busy |
| **Ollama** | 192.168.4.88:11434 | gemma4:26b | `llm_call()` bridge inside code_interpreter |
| **Tools API** | localhost:8080 | — | Web search (Brave), URL fetch, stock prices via SSE |
| **Gradio** | localhost:7860 | — | Chat WebUI |
| **Dashboard** | localhost:7861 | — | Live activity monitor, digest, agent requests |
| **Data** | ~/sandbox_agent_data/ | — | Bind-mounted volume, visible in Finder |

## Architecture

```
Docker Container (read_only, no-new-privileges, cap_drop: ALL)
├── Main Lane (Gradio WebUI) → vLLM qwen3.5, falls back to qwen3.5-27b when busy
├── Heartbeat Lane (1 hour interval) → checks HEARTBEAT.md
├── Cron Lane (60s poll) → executes scheduled tasks + pipeline stages on vLLM
├── Pipeline Orchestrator → code-based 6-stage project builder
├── Status Server (separate process, port 7861) → reads activity.jsonl from disk
└── LLM Bridge (HTTP server on localhost) → Ollama gemma4:26b for llm_call()
```

### Model Routing

- **User sends message**: tries qwen3.5 (non-blocking lock). If locked by background task, falls back to qwen3.5-27b.
- **Background tasks**: always use qwen3.5 (blocking lock — wait for user to finish). The 27b is reserved for user fallback only.
- **`llm_call()` inside code_interpreter**: always gemma4:26b on Ollama native API.

### First-Come-First-Served vLLM Mutex

The primary model lock is **non-blocking** for the main conversation. If a background task grabbed vLLM and is mid-tool-loop, the user's message either gets the lock (background released it) or proceeds to Ollama. The user is never stuck waiting.

Background tasks release the lock immediately after checking it — they don't hold it during the entire tool loop. This means vLLM may serve both user and background requests concurrently (queued), but the user waits at most ~30s for one background call, not the full 15-minute task.

## Tools (29 registered)

| Category | Tools | Description |
|---|---|---|
| **Web** | `web_search`, `web_url_fetch`, `stock_price` | Call port 8080 API via SSE. All outputs sanitized. Brave Search rate-limited to 1 req/2s. |
| **Code** | `code_interpreter` | Local Jupyter kernel (subprocess). Has `llm_call(prompt, system="", think=False)` via gemma4:26b. Pre-configured `DATA_DIR`, `PROJECTS_DIR` path vars. Output capped at 4k tokens. |
| **Scheduling** | `schedule_task`, `list_tasks`, `complete_task`, `cancel_task`, `pause_task`, `resume_task`, `update_task_checkpoint` | JSON-backed task queue. Active tasks in `tasks.json`, completed/cancelled archived to separate files. |
| **Pipeline** | `start_pipeline`, `pipeline_status`, `list_pipelines` | Code-based 6-stage project builder (see Pipeline section below). |
| **Self-edit** | `update_soul`, `update_heartbeat` | Patch a section or append a line (not full file replacement). Auto-committed to git. |
| **Memory** | `read_memories`, `add_memory` | MEMORIES.md with dated entries in 4 sections. Auto-loaded into system prompt. |
| **Projects** | `create_project`, `list_projects`, `delete_project`, `project_write_file`, `project_read_file`, `project_list_files`, `project_delete_file`, `move_file`, `delete_file` | Persistent file workspaces. `project_list_files` supports `path` param for browsing subdirectories. |
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
│   ├── llm_bridge.py         # HTTP bridge → gemma4:26b on Ollama for llm_call()
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
| `VLLM_BASE` | `http://192.168.4.66:8000/v1` | Primary + backup models (qwen3.5, qwen3.5-27b) |
| `LLM_CALL_MODEL` | `gemma4:26b` | Model for llm_call() in code_interpreter |
| `LLM_CALL_BASE` | `http://192.168.4.88:11434/v1` | Ollama server for llm_call() |
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

## Known Issues & Deferred Work

### Active Issues
- **vLLM thinking tokens**: Model outputs "Thinking Process: ..." inline. Fix requires vLLM server restart with `--chat-template-kwargs '{"enable_thinking": false}'` or passing `chat_template_kwargs: {"enable_thinking": false}` as a **top-level** field in the API request (not nested in `extra_body`).
- **Context bloat stalls**: When the agent makes many tool calls with large outputs (web_url_fetch), the context grows until vLLM becomes very slow or hangs. Need token-based loop limit.
- **Invalid JSON tool arguments**: The model sometimes generates malformed JSON with escaped quotes in `schedule_task` descriptions, causing tool call failures.

### Deferred Work (saved in memory)
- **Token-based loop limit**: Replace 50-iteration cap with ~200k token budget check per iteration.
- **User notifications**: Slack/webhook integration for real-time alerts (currently file-based).
- **Git commit on ALL file writes**: code_interpreter, chat_logger, checkpoints not yet auto-committed.
- **Conversation state persistence**: Save messages to disk so agent can resume after Gradio crash.
- **Disable vLLM thinking**: Needs server-side flag change.

### Operational Notes
- **Stuck tasks on restart**: Startup code resets any `running` tasks to `pending`.
- **Brave Search rate limit**: 2-second minimum between requests (enforced in api_tools.py).
- **Ollama `llm_call()` think parameter**: `think=False` (default) for extraction/classification, `think=True` for complex analysis. Uses Ollama native API, not OpenAI-compat.
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
