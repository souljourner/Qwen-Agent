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
| **vLLM** | 192.168.4.66:8000 | qwen3.5 (397B) | Main interactive conversation |
| **Ollama** | 192.168.4.88:11434 | qwen3.5-9b-256k | Background tasks, `llm_call()` bridge, fallback when vLLM busy |
| **Tools API** | localhost:8080 | — | Web search (Brave), URL fetch, stock prices via SSE |
| **Gradio** | localhost:7860 | — | Chat WebUI |
| **Dashboard** | localhost:7861 | — | Live activity monitor, digest, agent requests |
| **Data** | ~/sandbox_agent_data/ | — | Bind-mounted volume, visible in Finder |

## Architecture

```
Docker Container (read_only, no-new-privileges, cap_drop: ALL)
├── Main Lane (Gradio WebUI) → vLLM, falls back to Ollama when vLLM busy
├── Heartbeat Lane (1 hour interval) → checks HEARTBEAT.md, uses Ollama
├── Cron Lane (60s poll) → executes scheduled tasks, uses Ollama (or vLLM if idle)
├── Status Server (separate process, port 7861) → reads activity.jsonl from disk
└── LLM Bridge (HTTP server on localhost) → Ollama native API with think parameter
```

### Smart Model Routing

- **User sends message**: tries vLLM first. If vLLM lock is held (background task started on it), falls back to Ollama immediately.
- **Background task starts**: checks lock. If free (user not chatting), uses vLLM. Releases lock immediately so it doesn't block user. If user is chatting, uses Ollama.
- **`llm_call()` inside code_interpreter**: always Ollama native API (`/api/chat`) with `think` parameter.

### Key Design Decision: vLLM Never Blocks User

The primary model lock is **non-blocking** for the main conversation. If a background task grabbed vLLM and is mid-tool-loop, the user's message either gets the lock (background released it) or proceeds to Ollama. The user is never stuck waiting.

Background tasks release the lock immediately after checking it — they don't hold it during the entire tool loop. This means vLLM may serve both user and background requests concurrently (queued), but the user waits at most ~30s for one background call, not the full 15-minute task.

## Tools (23 registered)

| Category | Tools | Description |
|---|---|---|
| **Web** | `web_search`, `web_url_fetch`, `stock_price` | Call port 8080 API via SSE. All outputs sanitized. Brave Search rate-limited to 1 req/2s. |
| **Code** | `code_interpreter` | Local Jupyter kernel (subprocess, no Docker-in-Docker). Has `llm_call(prompt, system="", think=False)`. Variables persist. Output capped at 4k tokens. Dynamic timeout (10min-2hr). |
| **Scheduling** | `schedule_task`, `list_tasks`, `complete_task`, `cancel_task`, `update_task_checkpoint` | JSON-backed task queue. Cron/interval/one-shot. Project-scoped. Exponential backoff retry. `cancel_task` permanently removes (including recurring). |
| **Self-edit** | `update_soul`, `update_heartbeat` | Patch a section or append a line (not full file replacement). Auto-committed to git. |
| **Memory** | `read_memories`, `add_memory` | MEMORIES.md with dated entries in 4 sections. Auto-loaded into system prompt. |
| **Projects** | `create_project`, `list_projects`, `project_write_file`, `project_read_file`, `project_list_files`, `project_delete_file` | Persistent file workspaces at `DATA_DIR/projects/<name>/`. Auto-committed to git. |
| **Chat** | `list_chat_logs`, `read_chat_log` | Daily markdown logs at `DATA_DIR/chat_logs/YYYY-MM-DD.md`. |
| **Notifications** | `request_user`, `view_requests`, `resolve_request` | Structured JSON requests with pending/resolved status. Deduplication on pending subjects. |

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
│   ├── llm_bridge.py         # HTTP server → Ollama native API with think=true/false
│   ├── self_edit_tools.py    # update_soul (section patch), update_heartbeat, read/add memory
│   ├── project_tools.py      # Project workspace CRUD (6 tools)
│   ├── notification_tools.py # request_user, view_requests, resolve_request + read_digest
│   └── git_autocommit.py     # Auto-commit file changes in DATA_DIR git repo
├── heartbeat/
│   ├── heartbeat_runner.py   # Isolated sessions, HEARTBEAT_OK suppression, background work lock
│   └── HEARTBEAT.md          # Default checklist
├── scheduler/
│   ├── models.py             # Task pydantic model (cron, project, checkpoint, backoff)
│   ├── task_queue.py         # JSON-backed queue with dependencies, backoff, remove_task
│   ├── scheduler_tools.py    # schedule/list/complete/cancel/checkpoint tools
│   └── checkpoint.py         # Task state persistence for long-running work
└── docker/
    ├── Dockerfile            # Python 3.12, Qwen-Agent[gui], Jupyter, beautifulsoup4, lxml
    └── docker-compose.yml    # Security-hardened, bind mount, env vars
```

## Configuration (docker-compose.yml environment)

| Env Var | Value | Purpose |
|---|---|---|
| `VLLM_BASE` | `http://192.168.4.66:8000/v1` | Primary model |
| `OLLAMA_BASE` | `http://192.168.4.88:11434/v1` | Background model |
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
