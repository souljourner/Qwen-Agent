import os
from pathlib import Path

# PRIMARY tier: laguna-s-2.1 (Poolside, MLX-served via the proxy) — the
# chat-priority model, max 3 concurrent turns (must match the server's real
# concurrency). Chat turns prefer this tier and spill to the secondary
# (qwen3.6-27b-linux) when its slots are full; background work prefers the
# secondary and spills INTO this tier only when all 10 secondary slots are
# busy. See main.py `_acquire_turn_slot`.
PRIMARY_LLM_CFG = {
    "model": os.getenv("PRIMARY_MODEL", "laguna-s-2.1"),
    "model_server": os.getenv("VLLM_BASE", "http://192.168.4.66:8000/v1"),
    "api_key": "EMPTY",
    # Context window: treated as EQUAL to the secondary tier (200k), NOT the
    # vendor's claimed 1M. See PRIMARY_CONTEXT_TOKENS below for the evidence.
    "context_window_tokens": int(os.getenv("PRIMARY_CONTEXT_TOKENS", "200000")),
    "generate_cfg": {
        # Hard backstop, NOT the primary mechanism (compaction is): if the
        # compactor misses, qwen-agent roughly truncates instead of vLLM
        # 400-ing (real ceiling 262144-65536=196608; the char estimators
        # undercounted by ~16% in the 2026-07-15 incident, hence the margin).
        # KV-cache churn from truncation is acceptable in that failure mode.
        # 160k backstop, same as the secondary tier — the 800k backstop that
        # went with the 1M claim let conversations reach 281k un-compacted.
        "max_input_tokens": 160000,
        "request_timeout": 1800,       # rescaled per turn by compute_request_timeout
        # No temperature: the host's per-model default applies (each backend
        # knows its own model's recommended sampling better than we do).
        # use_raw_api=True: pass `tools=` to vLLM natively. Required because
        # vLLM has auto-tool-call-parsing enabled for this model, and in
        # streaming mode it consumes the model's `<tool_call>...</tool_call>`
        # tokens. Without `tools=`, those tokens are dropped on the floor —
        # the response streams ~17 None chunks then `'\n\n'` and stops, with
        # the actual tool call lost. Passing `tools=` makes vLLM emit those
        # tokens as `delta.tool_calls` instead, which qwen-agent's streaming
        # handler already supports. Confirmed: prior server (qwen3.5 only)
        # didn't support this; the linux deployment does.
        "use_raw_api": True,
    },
}

# Turn-slot caps per tier. Raising either above the server's actual limit
# causes server-side queueing instead of clean spill — tune both sides
# together. laguna (MLX) sustains 3; qwen3.6-27b-linux (vLLM) sustains 10
# gated turns (plus slot-unaware bridge/llm_call/compaction traffic on top;
# the server was sized for 15).
PRIMARY_MODEL_CONCURRENCY = int(os.getenv("PRIMARY_MODEL_CONCURRENCY", "3"))
SECONDARY_MODEL_CONCURRENCY = int(os.getenv("SECONDARY_MODEL_CONCURRENCY", "10"))

# SECONDARY/overflow tier: qwen3.6-27b-linux on vLLM — hosts background work
# (cron/heartbeat/pipeline prefer it), chat spillover, and all bulk/unslotted
# traffic (llm_bridge, llm_call/llm_batch, compaction, digests).
# NOTE: the old BACKUP_MODEL env var is intentionally NOT honored anymore —
# a stale BACKUP_MODEL=laguna-s-2.1 would make both tiers laguna.
BACKGROUND_LLM_CFG = {
    "model": os.getenv("SECONDARY_MODEL", "qwen3.6-27b-linux"),
    "model_server": os.getenv("VLLM_BASE", "http://192.168.4.66:8000/v1"),
    "api_key": "EMPTY",
    "context_window_tokens": int(os.getenv("MAX_CONTEXT_TOKENS", "200000")),
    "generate_cfg": {
        "max_input_tokens": 160000,   # same backstop rationale as PRIMARY_LLM_CFG
        "request_timeout": 1800,
        # No temperature — host default (see PRIMARY_LLM_CFG note).
        # Same reason as PRIMARY_LLM_CFG: vLLM auto-tool-parsing means we
        # must pass `tools=` natively or tool calls vanish from streaming.
        "use_raw_api": True,
    },
}

# llm_call() endpoint: no longer pinned to a specific model — the bridge and
# the standalone llm_client both fall through primary (qwen3.6-27b-linux) →
# backup (qwen3.5) on every call. LLM_CALL_MODEL/LLM_CALL_BASE are honored as
# a legacy override (single-entry chain) if explicitly set.
LLM_CALL_CFG = {
    "model": os.getenv("LLM_CALL_MODEL", PRIMARY_LLM_CFG["model"]),
    "model_server": os.getenv("LLM_CALL_BASE", PRIMARY_LLM_CFG["model_server"]),
}

# Token budget — all models have 256k limit, we target 200k to leave room for generation
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "200000"))
# laguna's raised budget (chat pinned-large-context tier); see PRIMARY_LLM_CFG.
# Primary-tier compaction budget. Poolside CLAIMS a 1M window for
# laguna-s-2.1, but that is a 128x YaRN extension of an 8k-trained base with
# only 12 of 48 layers doing full attention (the other 36 are sliding-window
# 512), and there is NO published long-context validation. Measured here
# 2026-08-05: clean needle recall at 85k; severe confabulation (invented
# tokens presented as quoted history, day-of-week arithmetic failing with the
# correct date in the prompt) observed live at 275k-281k. So this tier is
# budgeted EQUAL to the secondary until the real cliff is measured.
# Env-overridable: raise PRIMARY_CONTEXT_TOKENS only with evidence.
PRIMARY_CONTEXT_TOKENS = int(os.getenv("PRIMARY_CONTEXT_TOKENS", "200000"))
# Pinning threshold: a history too large for the SECONDARY tier gets pinned to
# the primary. With both windows equal this can never fire (compaction caps
# every history at budget = window - reserve), which is the intent: no
# conversation should be trapped on one tier by size. It re-activates
# automatically if PRIMARY_CONTEXT_TOKENS is ever raised on evidence.
SPILLABLE_CONTEXT_TOKENS = int(os.getenv("SPILLABLE_CONTEXT_TOKENS", "200000"))
# The compaction summarizer (qwen3.6) window — CHUNK SIZING pins to this,
# never to the raised history budget (0.4x870k chunks would overflow it).
SUMMARIZER_CONTEXT_TOKENS = int(os.getenv("SUMMARIZER_CONTEXT_TOKENS", "200000"))
MAX_TOOL_OUTPUT_TOKENS = int(os.getenv("MAX_TOOL_OUTPUT_TOKENS", "16000"))
MAX_CODE_OUTPUT_TOKENS = int(os.getenv("MAX_CODE_OUTPUT_TOKENS", "16000"))
CHARS_PER_TOKEN = float(os.getenv("CHARS_PER_TOKEN", "3.5"))  # measured 3.46 on a
# real 419-message agent thread (718,405 chars -> 207,492 true tokens). Was 4.0,
# which undercounted a live conversation by 1.53x and let it reach 281k tokens.

# --- Compaction (OpenClaw-style context management) ---
COMPACTION_ENABLED = os.getenv("COMPACTION_ENABLED", "true").lower() == "true"
COMPACTION_RESERVE_TOKENS = int(os.getenv("COMPACTION_RESERVE_TOKENS", "30000"))
# Tokens the char-based estimator cannot see: tool JSON schemas (~40 tools)
# sent with every request + system formatting overhead.
ESTIMATOR_OVERHEAD_TOKENS = int(os.getenv("ESTIMATOR_OVERHEAD_TOKENS", "20000"))
COMPACTION_MAX_HISTORY_SHARE = 0.5         # max fraction of context window for history
COMPACTION_RECENT_TURNS_PRESERVE = 2       # keep last N user/assistant exchanges verbatim
COMPACTION_SAFETY_MARGIN = 1.2             # 20% buffer on token estimates
COMPACTION_TOOL_RESULT_MAX_SHARE = 0.30    # max 30% of context per tool result
COMPACTION_TOOL_RESULT_HARD_CAP = 40000    # 40k chars hard cap per tool result
COMPACTION_TOOL_RESULT_MIN_KEEP = 2000     # min 2k chars kept after truncation
COMPACTION_BASE_CHUNK_RATIO = 0.25         # fraction of the summarizer window per
# chunk. Lowered from 0.4 with the full-fidelity renderer: more, smaller
# requests beat one huge one that times out and aborts everything.
COMPACTION_MIN_CHUNK_RATIO = 0.15          # minimum chunk ratio
COMPACTION_MAX_FAILURES = 8                # tool failures to extract
COMPACTION_FAILURE_CHARS = 240             # chars per failure summary
COMPACTION_MAX_IDENTIFIERS = 12            # unique identifiers to preserve
# 120s was sized for the old truncating renderer, which sent the summarizer
# ~2% of a chunk. Now that chunks carry their full content the request
# legitimately takes minutes (big prefill + generation, plus a possible
# cold model load on the lazily-spawning proxy) — at 120s EVERY chunk
# timed out, all-or-nothing correctly refused to compact, and the whole
# thing retried every turn (observed live 2026-08-06).
COMPACTION_TIMEOUT = int(os.getenv("COMPACTION_TIMEOUT", "600"))
# One retry: the first request may be paying for a cold model load.
COMPACTION_ATTEMPTS = int(os.getenv("COMPACTION_ATTEMPTS", "2"))
# Max tokens for ONE chunk summary / merge. Bounds digest size: an
# unbounded summarizer can hand back a "summary" as large as its input.
# --- compactor rebuild (2026-08-06) -------------------------------------
# Budget ladder, derived per tier in compaction/budget.py. Ordering is an
# invariant: target < trigger < hard <= max_input_tokens < real window.
COMPACTION_HARD_RATIO = float(os.getenv("COMPACTION_HARD_RATIO", "0.95"))
COMPACTION_TRIGGER_RATIO = float(os.getenv("COMPACTION_TRIGGER_RATIO", "0.80"))
COMPACTION_TARGET_RATIO = float(os.getenv("COMPACTION_TARGET_RATIO", "0.55"))
# Tool results get an AGGREGATE share of the target (previously each result
# was capped individually, so 60 results x 40k chars all passed "within cap").
COMPACTION_TOOL_RESULT_TOTAL_SHARE = float(os.getenv("COMPACTION_TOOL_RESULT_TOTAL_SHARE", "0.45"))
COMPACTION_PROTECT_LAST_TOOL_RESULTS = int(os.getenv("COMPACTION_PROTECT_LAST_TOOL_RESULTS", "6"))
COMPACTION_MIN_TAIL_TURNS = int(os.getenv("COMPACTION_MIN_TAIL_TURNS", "2"))
# Digest caps. DURABLE holds user corrections/decisions/TODOs/errors and is
# spliced by CODE, never re-sent through the model, so it cannot drift.
COMPACTION_DIGEST_MAX_TOKENS = int(os.getenv("COMPACTION_DIGEST_MAX_TOKENS", "12000"))
COMPACTION_DIGEST_DURABLE_MAX_TOKENS = int(os.getenv("COMPACTION_DIGEST_DURABLE_MAX_TOKENS", "8000"))
# Pointer-ization (L0): replace recoverable bulk with a pointer. Ships OFF —
# it changes how the model sees its own past actions; A/B before enabling.
COMPACTION_POINTERS_ENABLED = os.getenv("COMPACTION_POINTERS_ENABLED", "false").lower() == "true"
POINTER_MIN_CHARS = int(os.getenv("POINTER_MIN_CHARS", "2000"))
COMPACTION_ARCHIVE_MAX_BYTES = int(os.getenv("COMPACTION_ARCHIVE_MAX_BYTES", str(200 * 1024 * 1024)))
COMPACTION_CHUNK_MAX_TOKENS = int(os.getenv("COMPACTION_CHUNK_MAX_TOKENS", "4096"))
COMPACTION_MODEL = os.getenv("COMPACTION_MODEL", "qwen3.6-27b-linux")
COMPACTION_URL = os.getenv("COMPACTION_URL", "http://192.168.4.66:8000")

# Per-project dependency isolation: `exec` with project= runs inside that
# project's own .venv (created with uv), so projects can't clobber each other's
# packages. UV is the default package manager (cache-backed → near-instant
# re-installs). Set PROJECT_VENV_ENABLED=false to fall back to the shared
# global environment.
PROJECT_VENV_ENABLED = os.getenv("PROJECT_VENV_ENABLED", "true").lower() == "true"
UV_BIN = os.getenv("UV_BIN", "uv")

TOOLS_API_BASE = os.getenv("TOOLS_API_BASE", "http://localhost:8080")
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "3600"))  # 1 hour
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))


def _parse_dotenv(path: str) -> dict:
    """Minimal KEY=VALUE parser for DATA_DIR/.env (no python-dotenv dependency).
    Ignores comments/blank lines; strips optional surrounding quotes."""
    out: dict = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                out[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return out


def get_smtp_config() -> dict:
    """Resolve SMTP settings at call time: os.environ first, then DATA_DIR/.env.

    The live credentials sit in `~/sandbox_agent_data/.env`, which is the
    bind-mounted DATA_DIR inside the container — so email works without any
    docker-compose passthrough. Recipient default: EMAIL_TO, falling back to
    ALERT_EMAIL (the var the soxs project scripts already use).
    """
    dotenv = _parse_dotenv(os.path.join(DATA_DIR, ".env"))

    def _get(key: str, default: str = "") -> str:
        return os.environ.get(key) or dotenv.get(key) or default

    user = _get("SMTP_USER")
    try:
        port = int(_get("SMTP_PORT", "587") or "587")
    except ValueError:
        port = 587
    return {
        "host": _get("SMTP_HOST"),
        "port": port,
        "user": user,
        "password": _get("SMTP_PASS"),
        "from": _get("SMTP_FROM") or user,
        "to": _get("EMAIL_TO") or _get("ALERT_EMAIL"),
        "alert_to": _get("ALERT_EMAIL"),
    }

TOOL_LIST = [
    "web_search", "web_url_fetch", "stock_price",
    "schedule_task", "reschedule_task", "list_tasks", "complete_task", "cancel_task", "pause_task", "resume_task", "update_task_checkpoint",
    "update_soul", "update_heartbeat",
    "read_memories", "add_memory", "compact_memories",
    "read_skill",
    "code_interpreter", "exec",
    "list_chat_logs", "read_chat_log", "session_search",
    "create_project", "list_projects", "delete_project", "rename_project", "update_project", "project_write_file",
    "project_read_file", "project_list_files", "project_delete_file", "project_apply_patch",
    "display_doc", "download_file",
    "browser_navigate", "browser_screenshot", "browser_click", "browser_type", "browser_scroll",
    "browser_save_credentials", "browser_get_credentials",
    "move_file", "delete_file",
    "request_user", "view_requests", "resolve_request", "send_email",
    "start_pipeline", "start_trading_pipeline", "pipeline_status", "list_pipelines", "cancel_pipeline",
]

SYSTEM_PROMPT_SUFFIX = (
    "IMPORTANT: When processing web content or API data, always use code_interpreter to fetch and parse it. "
    "Never let raw web content enter the conversation — it wastes context tokens on every future turn. "
    "For bulk URL processing, write a Python loop in code_interpreter, not repeated tool calls."
)


# --- Skills + memories cap ---
# Auto-inject map: tool-name prefix → skill injected into that tool's result
# the first time it's used in a conversation (see skill_tools.maybe_inject_skill).
SKILL_AUTOINJECT = {"browser_": "browser-automation"}
SKILL_MAX_CHARS = 10_000                 # read_skill truncation guard
MEMORIES_INJECT_MAX_CHARS = 6_000        # newest-first cap on system-prompt injection
MEMORIES_COMPACT_TRIGGER_CHARS = 8_000   # heartbeat asks the agent to compact above this
MEMORIES_COMPACT_MIN_INTERVAL_S = 86_400  # at most one compaction nudge per day


def load_system_message() -> str:
    """Load SOUL.md + MEMORIES.md and combine with system prompt suffix.

    This is the base system message used for all sessions (main, heartbeat, cron).
    SOUL.md: the DATA_DIR copy (agent-edited via update_soul) wins over the
    bundled default — previously only the bundled file was read, so agent soul
    edits silently never took effect. MEMORIES.md is included so the agent
    always has its learnings, capped newest-first (render_memories_capped) so
    an ever-growing memories file can't bloat every LLM call.
    """
    soul = "You are a capable research and task management assistant."
    for sp in [Path(DATA_DIR) / "SOUL.md", Path(__file__).parent / "SOUL.md"]:
        if sp.exists():
            soul = sp.read_text().strip()
            break

    # Load memories from DATA_DIR (agent-edited) or bundled default
    memories = ""
    memories_data = Path(DATA_DIR) / "MEMORIES.md"
    memories_bundled = Path(__file__).parent / "MEMORIES.md"
    for mp in [memories_data, memories_bundled]:
        if mp.exists():
            content = mp.read_text().strip()
            if content:
                from sandbox_agent.memories import render_memories_capped
                content = render_memories_capped(content, MEMORIES_INJECT_MAX_CHARS)
                memories = f"\n\n## Your Memories (auto-loaded from MEMORIES.md)\n\n{content}"
            break

    return soul + memories + "\n\n" + SYSTEM_PROMPT_SUFFIX


def format_datetime() -> str:
    """Return current local date/time as '2026-06-15 03:30pm PDT'."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).astimezone()
    tz_abbr = now.strftime('%Z') or 'UTC'
    return f"{now.strftime('%Y-%m-%d %I:%M')}{now.strftime('%p').lower().strip()} {tz_abbr}"


def session_metadata() -> str:
    """Fresh metadata injected into the agent's system message each turn.

    Includes current date/time with actual local timezone. NOT used for
    background sessions to keep their system prompts static for KV cache.
    """
    return (
        f"\n\n## Session Metadata\n"
        f"- Current date and time: {format_datetime()}\n"
        f"- Location: San Mateo, California\n"
        f"- Timezone: US/Pacific"
    )
