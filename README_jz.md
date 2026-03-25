# Sandbox Agent

A sandboxed, self-scheduling Qwen Agent with web tools, code execution, heartbeat monitoring, and vLLM KV cache optimization.

Built on top of [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), inspired by [OpenClaw](https://docs.openclaw.ai/) patterns.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Docker Container (read_only, no-new-privileges, cap_drop: ALL)     │
│                                                                      │
│  ┌─────────────────┐  ┌──────────┐  ┌───────────────┐               │
│  │   Main Lane     │  │ Heartbeat│  │   Cron Lane   │               │
│  │ (Gradio WebUI)  │  │  Lane    │  │ (scheduled    │               │
│  │                 │  │ (30 min) │  │  tasks)       │               │
│  └────────┬────────┘  └────┬─────┘  └──────┬────────┘               │
│           │                │               │                         │
│     ┌─────┴────────────────┴───────────────┴──────┐                  │
│     │         Primary Model Lock                   │                  │
│     │  Main holds lock → background uses Ollama    │                  │
│     │  Main idle → background uses primary vLLM    │                  │
│     └──────────────────────────────────────────────┘                  │
│           │                                                          │
│  ┌────────┴──────────────────────────────────────────────────────┐   │
│  │                    Shared Tool Registry (22 tools)             │   │
│  │  web_search · web_url_fetch · stock_price                     │   │
│  │  code_interpreter (with llm_call() bridge)                    │   │
│  │  schedule_task · list_tasks · complete_task                   │   │
│  │  update_task_checkpoint                                       │   │
│  │  read_soul · update_soul · read_heartbeat · update_heartbeat  │   │
│  │  read_memories · add_memory · list_chat_logs · read_chat_log  │   │
│  │  create_project · list_projects · project_write_file          │   │
│  │  project_read_file · project_list_files · project_delete_file │   │
│  └──────┬──────────────┬──────────────┬──────────────────────────┘   │
│         │              │              │                               │
│  ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴───────────┐                  │
│  │ Tools API   │ │ Task Queue │ │ Jupyter Kernel  │                  │
│  │ :8080 (SSE) │ │ (JSON)     │ │ (subprocess)    │                  │
│  └─────────────┘ │ + Git repo │ │ + LLM Bridge    │                  │
│                  └────────────┘ └─────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
         │                │                    │
    ┌────┴─────┐    ┌─────┴──────┐     ┌──────┴───────┐
    │  vLLM    │    │  vLLM      │     │   Ollama     │
    │ :8000    │    │ 192.168.   │     │  192.168.    │
    │ localhost│    │ 4.66:8000  │     │  4.88:11434  │
    │          │    │ qwen3.5    │     │  qwen3.5-9b  │
    └──────────┘    │ (primary)  │     │ (background) │
                    └────────────┘     └──────────────┘
```

### Dual-LLM Design

| Model | Hardware | Role | Why |
|---|---|---|---|
| `qwen3.5` via vLLM | 192.168.4.66:8000 | Interactive conversation | Best model, KV cache optimized |
| `qwen3.5-9b-256k` via Ollama | 192.168.4.88:11434 | Heartbeat, cron, `llm_call()` | Separate hardware avoids queue contention |

The vLLM server is single-queue — heartbeat/cron requests would block interactive use. The Ollama instance on a second machine handles all background work independently.

### Smart Model Sharing

A lock controls access to the primary model:
- **User is chatting** → main lane holds lock → heartbeat/cron fall back to Ollama
- **User is idle** → lock is free → heartbeat/cron use the primary vLLM (better quality)
- **No deadlocks** — background lanes never block on the lock, they fall back immediately

### Lane-Based Concurrency (OpenClaw Pattern)

Three threads run independently:

- **Main lane**: Persistent conversation via Gradio WebUI. Uses the primary vLLM model. Messages accumulate across turns.
- **Heartbeat lane**: Every 30 minutes, creates a fresh isolated agent session. Reads `HEARTBEAT.md`, checks for issues, suppresses `HEARTBEAT_OK` responses silently.
- **Cron lane**: Polls the task queue every 60 seconds. Executes due tasks in isolated sessions. Supports exponential backoff retry on failure.

Isolated sessions mean heartbeat/cron never pollute the main conversation history and cost ~2-5K tokens each instead of replaying the full context.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) for Python package management
- vLLM running on 192.168.4.66:8000 with `qwen3.5`
- Ollama running on 192.168.4.88:11434 with `qwen3.5-9b-256k:latest`
- Tools API running on localhost:8080 (provides `web_search`, `web_url_fetch`, `stock_price`)
- Docker (for sandboxed deployment)

### Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[gui]" croniter numpy soundfile ipykernel jupyter_client pandas
```

### Run Locally (Development)

```bash
# Gradio WebUI (default) — opens browser at http://localhost:7860
python -m sandbox_agent.main

# Terminal REPL
python -m sandbox_agent.main --mode repl

# Custom port
python -m sandbox_agent.main --port 8888
```

### Run in Docker Sandbox

```bash
cd sandbox_agent/docker

# Gradio WebUI (default) — access at http://localhost:7860
docker compose up --build

# Terminal REPL (interactive)
docker compose run --rm -it agent --mode repl

# Background only (headless, heartbeat/cron)
docker compose up -d
```

Note: `docker compose up` works for Gradio mode (no stdin needed). For REPL mode, use `docker compose run -it` to connect your terminal.

### Run Tests

```bash
source .venv/bin/activate
pytest tests/sandbox_agent/ -v
```

102 tests covering sanitizer, API tools, code interpreter, task queue, checkpoint, scheduler tools, heartbeat runner, self-edit tools, token budget, and chat logging.

## Configuration

All configuration is in `sandbox_agent/config.py` and can be overridden via environment variables:

| Env Var | Default | Description |
|---|---|---|
| `VLLM_BASE` | `http://192.168.4.66:8000/v1` | Primary vLLM endpoint |
| `OLLAMA_BASE` | `http://192.168.4.88:11434/v1` | Background Ollama endpoint |
| `TOOLS_API_BASE` | `http://localhost:8080` | Web tools API endpoint |
| `HEARTBEAT_INTERVAL_SECONDS` | `1800` | Heartbeat check interval (30 min) |
| `DATA_DIR` | `./data` (local) / `/app/data` (Docker) | Writable directory for task queue, checkpoints, git repo |
| `MAX_CONTEXT_TOKENS` | `200000` | Token budget for conversation context (256k model limit) |
| `MAX_TOOL_OUTPUT_TOKENS` | `8000` | Max tokens per tool result |
| `MAX_CODE_OUTPUT_TOKENS` | `4000` | Max tokens per code_interpreter execution |

## Components

### SOUL.md — Agent Identity

`sandbox_agent/SOUL.md` defines the agent's identity and is injected as the first part of every system prompt. Follows the OpenClaw SOUL.md pattern. Contains:

- **Core identity** and personality
- **Token efficiency rules** — teaches the agent to offload heavy content to code_interpreter, use `llm_call()` for per-item processing, and schedule heavy work as background tasks
- **`llm_call()` usage patterns** — with examples of extraction, classification, and batch processing
- **Capabilities list** — all 22 tools described
- **When to save memories** — guidelines for what's worth persisting
- **Boundaries** — including self-edit safety rules (never remove critical sections)

The agent can update its own SOUL.md via the `update_soul` tool. Changes take effect on new background sessions and after restart, not the current conversation.

### Web Tools (Port 8080)

Three tools registered via `@register_tool` that call the local API:

| Tool | Parameters | Description |
|---|---|---|
| `web_search` | `query: str` | Brave web search (overwrites built-in Serper-based tool) |
| `web_url_fetch` | `url: str` | Fetch URL content as markdown |
| `stock_price` | `symbol: str` | Stock price and market data (e.g., AAPL, GOOGL) |

All tools communicate with the API via SSE (`POST /api/tools/execute` with `{"name": ..., "arguments": {...}}`).

#### Content Sanitization

All web tool outputs pass through `sandbox_agent/tools/sanitizer.py` before reaching the LLM:

1. **Strip chat-template tokens** — removes `<|im_start|>`, `<|im_end|>`, `[INST]`, `### SYSTEM:`, etc. that could enable role-switching attacks
2. **Remove invisible characters** — zero-width spaces, RTL overrides, BOM, and other formatting chars used to hide injected content
3. **Strip delimiter injection** — removes `[TOOL_OUTPUT]`/`[/TOOL_OUTPUT]` from content to prevent breakout from the tool output wrapper
4. **Truncate** — caps output to `MAX_TOOL_OUTPUT_TOKENS` (default 8000 tokens / ~32k chars)
5. **Delimit** — wraps output in `[TOOL_OUTPUT]...[/TOOL_OUTPUT]` so the model treats it as tool output, not instructions

### Code Interpreter

A local Jupyter kernel running as a subprocess inside the container (no Docker-in-Docker). The agent can write and execute arbitrary Python code.

**Pre-installed packages**: numpy, pandas, requests, Pillow

**Persistent state**: Variables survive between calls within the same session.

**`llm_call()` bridge**: Inside code_interpreter, the agent has access to `llm_call(prompt, system="")` which calls the background Ollama model. This enables:

```python
# The agent writes code like this:
import requests

urls = ["https://example.com/page1", "https://example.com/page2", ...]
results = []
for url in urls:
    html = requests.get(url, timeout=30).text
    # LLM processes each page with a task-specific prompt
    extracted = llm_call(
        f"Extract the product name and price from this page:\n{html[:4000]}",
        system="Return JSON: {product_name, price}. Nothing else."
    )
    results.append({"url": url, "data": extracted})

import json
print(json.dumps(results, indent=2))  # Only this enters the conversation
```

Each `llm_call()` runs on the background model — it does NOT add tokens to the main conversation. The prompt can be anything: extraction, classification, translation, comparison, reformatting, etc.

**Security**: The LLM bridge uses a shared secret token generated at startup (injected into the kernel). Only the kernel process can call it. Request body is capped at 1MB.

**Output cap**: Code execution output is truncated to `MAX_CODE_OUTPUT_TOKENS` (default 4000 tokens).

### Self-Edit Tools

The agent can read and modify its own configuration files:

| Tool | Description |
|---|---|
| `read_soul` | Read current SOUL.md |
| `update_soul` | Replace SOUL.md content (auto-committed to git) |
| `read_heartbeat` | Read current HEARTBEAT.md checklist |
| `update_heartbeat` | Replace HEARTBEAT.md content (auto-committed to git) |

Files are written to `DATA_DIR` (the writable Docker volume). Every update is automatically committed to a private git repo in `DATA_DIR` so the agent's edit history is tracked.

Changes take effect on new background sessions (heartbeat/cron) and after restart, but NOT the current interactive conversation.

**Safety**: SOUL.md instructs the agent to never remove the Token Efficiency Rules, Capabilities, or Boundaries sections.

### Persistent Memory (MEMORIES.md)

The agent has long-term memory that persists across sessions:

| Tool | Description |
|---|---|
| `read_memories` | Read all saved memories |
| `add_memory` | Append a concise learning under a specific section |

Memories are organized into four sections:
- **User Preferences** — how the user likes things done, topics of interest
- **Facts & Knowledge** — important facts, domain knowledge, corrections
- **Technical Notes** — system configurations, API quirks, what works/doesn't
- **Task Learnings** — outcomes of tasks, useful data sources, patterns that worked

Each entry is dated (e.g., `- [2026-03-24] User prefers concise bullet-point reports`) and auto-committed to git. The agent is instructed to save important learnings immediately during conversations, not wait until the end.

### Chat Logs

All conversations are logged daily as markdown files in `DATA_DIR/chat_logs/YYYY-MM-DD.md`. Background task results are also logged.

| Tool | Description |
|---|---|
| `list_chat_logs` | List available log files by date |
| `read_chat_log` | Read a specific day's log (`"today"` or `"2026-03-24"`) |

The agent can use these to recall earlier conversations when the user references past sessions.

### Session Metadata

The main interactive agent receives one-time metadata in its system prompt at startup:
- Current date and time
- Location: San Mateo, California
- Timezone: US/Pacific

This is NOT included in background sessions (heartbeat/cron) to keep their system prompts static for KV cache efficiency.

### Project Workspaces

Persistent file storage organized by project for long-running, multi-session work.

| Tool | Description |
|---|---|
| `create_project` | Create a named workspace with description |
| `list_projects` | Show all projects with file counts |
| `project_write_file` | Write/update files (supports subdirs like `research/competitors.md`) |
| `project_read_file` | Read a project file |
| `project_list_files` | List all files in a project |
| `project_delete_file` | Remove a file |

All writes are auto-committed to git. Projects live in `DATA_DIR/projects/<name>/`. The agent is taught to create a project at the start of any multi-session effort and save all artifacts there.

### Task Scheduler

The agent can schedule its own tasks, optionally scoped to a project:

| Tool | Description |
|---|---|
| `schedule_task` | Create a new task (one-shot, interval, or cron), optionally scoped to a project |
| `list_tasks` | List tasks, optionally filtered by status and/or project |
| `complete_task` | Mark a task as done |
| `update_task_checkpoint` | Save progress for long-running tasks |

#### Schedule Types (OpenClaw Pattern)

- **`at`** — One-shot: runs once at a specified time (or immediately if no `run_at`)
- **`every`** — Interval: runs repeatedly every N seconds
- **`cron`** — Cron expression: standard 5-field cron (e.g., `0 * * * *` for hourly)

#### Task Dependencies

Tasks can declare `depends_on: [task_id, ...]`. A task won't execute until all dependencies are completed.

#### Exponential Backoff Retry

Failed tasks retry with increasing delays: 30s → 1m → 5m → 15m → 60m. One-shot tasks stop after `max_retries` (default 3). Recurring tasks retry indefinitely.

#### Checkpointing

Long-running tasks can save intermediate state via `update_task_checkpoint`. On interruption, the checkpoint is loaded and injected into the task prompt so the agent can resume from where it left off.

#### Persistence

The task queue is backed by `DATA_DIR/tasks.json` with auto-commit to git. Survives container restarts via the Docker named volume.

### Heartbeat (OpenClaw Pattern)

**How it works:**

1. 60-second startup delay, then every 30 minutes the heartbeat timer fires
2. Loads `HEARTBEAT.md` (user-editable checklist) from `DATA_DIR`
3. Also checks the task queue for due scheduled tasks
4. If there are items to check: creates an isolated agent session with the prompt: *"Read the HEARTBEAT checklist. Follow it strictly. If nothing needs attention, reply HEARTBEAT_OK."*
5. If the agent responds with `HEARTBEAT_OK` (≤300 chars): silently suppressed, user never sees it
6. If the agent finds something: logs a `HEARTBEAT ALERT`

The agent can also edit the checklist itself via `update_heartbeat`.

**Editing the checklist:**

```markdown
# Heartbeat Checklist
- [ ] Check for due scheduled tasks and execute them
- [ ] Review task queue for failed tasks that should be retried
- [ ] Check AAPL stock price and alert if below $200
- [x] This item is done and will be skipped
```

## Token Budget Management

All models have a 256k token limit. The system targets 200k to leave room for generation.

| Layer | Budget | Mechanism |
|---|---|---|
| **Conversation context** | 200k tokens | `trim_to_budget()` drops oldest turns, preserves system prefix for KV cache. Inserts a note: *"[Note: N earlier messages were trimmed...]"* |
| **Tool outputs** (web_search, web_url_fetch, stock_price) | 8k tokens (~32k chars) | Sanitizer truncates |
| **Code interpreter output** | 4k tokens (~16k chars) | Output capped after execution |
| **llm_call() prompt** | 200k tokens | Bridge truncates oversized prompts |
| **Request timeout** | 10-30 min (dynamic) | Scales linearly with payload: 10 min at ~0 tokens, 30 min at full 200k context |
| **Code interpreter timeout** | 10 min - 2 hr (dynamic) | Scales with code size to accommodate `llm_call()` loops |

Token estimation uses **1 token ≈ 4 characters** (conservative heuristic, no expensive tokenization). All limits configurable via env vars.

## KV Cache Optimization (vLLM Prefix Caching)

The configuration is tuned to maximize vLLM prefix cache hits during the agent's function-calling loop:

| Setting | Value | Why |
|---|---|---|
| `max_input_tokens` | `0` | Disables Qwen-Agent's client-side truncation. Without this, truncation removes messages from the middle, breaking the prefix. |
| Prompt-based function calling | `use_raw_api=False` (default) | vLLM does not support native OpenAI `tool_calls`. Qwen-Agent injects tool definitions into the system message and parses calls from text output. The system message is stable within a `_run()` loop. |
| Static system message | SOUL.md + fixed suffix | No dynamic content in the base system message. Session metadata (date/time) is only added once for the main agent. |
| Append-only message loop | (built into FnCallAgent) | Only appends to the message list — never modifies or reorders earlier messages. |
| Conversation trimming | Drops oldest turns | When trimming for token budget, removes from the front (after system message), preserving the most recent prefix. |

**Net effect**: Within a single agent `_run()` loop (e.g., search → read result → search again), each successive LLM call reuses the full KV cache of all previous messages.

## Tool Call Logging

Every tool invocation is logged with the tool name, arguments (first 200 chars), and result (first 200 chars):

```
Tool call: project_read_file({"project": "flatsixai", "path": "TODO.md"})
Tool result: project_read_file -> # FlatSixAI - Active TODOs...

Tool call: web_search({"query": "AI governance 2026"})
Tool result: web_search -> [TOOL_OUTPUT] [1]"JetStream raises $34M..."...
```

This applies to all agents — main conversation, heartbeat, and cron tasks. View via `docker compose logs`.

## Gradio Timeout

`GRADIO_SERVER_TIMEOUT=3600` (1 hour) is set in docker-compose to prevent "Connection errored out" errors during long-running tasks. The default Gradio timeout is too short for code_interpreter sessions that process many URLs with `llm_call()`.

## Docker Security

The container runs with:
- `read_only: true` — cannot write to the host filesystem
- `no-new-privileges: true` — prevents privilege escalation
- `cap_drop: ALL` — drops all Linux capabilities
- `tmpfs` on `/tmp` with 100M limit
- Non-root `agent` user
- Named volume `agent_data` on `/app/data` — only writable persistent storage
- Port 7860 exposed for Gradio WebUI

## File Structure

```
sandbox_agent/
├── __init__.py
├── config.py                         # Dual-LLM config, token budgets, env vars
├── main.py                           # Gradio/REPL entry, 3 lanes, model lock
├── token_budget.py                   # Token estimation, conversation trimming
├── chat_logger.py                   # Daily chat logs + list/read tools
├── SOUL.md                           # Agent identity + token efficiency rules
├── tools/
│   ├── __init__.py
│   ├── api_tools.py                  # 3 tools calling port 8080 via SSE
│   ├── sanitizer.py                  # Prompt injection defense
│   ├── code_interpreter.py           # Local Jupyter kernel (no Docker-in-Docker)
│   ├── llm_bridge.py                 # HTTP bridge for llm_call() in kernel
│   ├── self_edit_tools.py            # read/update SOUL.md, HEARTBEAT.md, MEMORIES.md
│   ├── project_tools.py             # Project workspace CRUD (6 tools)
│   └── git_autocommit.py            # Auto-commit data changes to git
├── heartbeat/
│   ├── __init__.py
│   ├── heartbeat_runner.py           # Isolated-session heartbeat loop
│   ├── HEARTBEAT.md                  # Default checklist template
│   └── MEMORIES.md                   # Default persistent memory file
├── scheduler/
│   ├── __init__.py
│   ├── models.py                     # Task pydantic model
│   ├── task_queue.py                 # JSON-backed queue with cron + backoff
│   ├── scheduler_tools.py           # 4 registered tools for self-scheduling
│   └── checkpoint.py                 # Long-running task state persistence
└── docker/
    ├── Dockerfile                    # Sandboxed container with Gradio + Jupyter
    └── docker-compose.yml            # Security-hardened compose config

tests/sandbox_agent/
├── tools/
│   ├── test_sanitizer.py             # 20 tests
│   ├── test_api_tools.py             # 12 tests
│   ├── test_code_interpreter.py      # 13 tests
│   └── test_self_edit_tools.py       # 10 tests
├── scheduler/
│   └── test_scheduler.py             # 23 tests
├── heartbeat/
│   └── test_heartbeat.py             # 15 tests
└── test_token_budget.py              # 9 tests
```
