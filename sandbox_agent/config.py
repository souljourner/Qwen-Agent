import os
from pathlib import Path

# Primary LLM: qwen3.6-27b-linux on vLLM (192.168.4.66) — supports concurrency 15
# Used for main interactive conversation AND background tasks; up to 15 in flight
# at once. Saturating all slots routes user chat to the backup (see main.py
# `_primary_model_lock`, sized as BoundedSemaphore(PRIMARY_MODEL_CONCURRENCY)).
PRIMARY_LLM_CFG = {
    "model": "qwen3.6-27b-linux",
    "model_server": os.getenv("VLLM_BASE", "http://192.168.4.66:8000/v1"),
    "api_key": "EMPTY",
    "generate_cfg": {
        "max_input_tokens": 0,        # Disable client-side truncation for KV cache stability
        "request_timeout": 1800,       # 30min — covers full 200k context window worst case
        "temperature": 0.6,
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

# How many primary-model requests may be in flight at once. The 27b-linux model
# is configured for concurrency 15 on the vLLM side; raising this above the
# server's actual limit will cause queueing on the server instead of clean
# fallback to the backup. Tune both sides together.
PRIMARY_MODEL_CONCURRENCY = int(os.getenv("PRIMARY_MODEL_CONCURRENCY", "15"))

# Backup LLM: qwen3.5 (the 397B MoE) on same vLLM server.
# Reserved for user chat fallback when all primary slots are occupied by
# background work. Background tasks NEVER use this directly — they always
# acquire a primary-model slot.
BACKGROUND_LLM_CFG = {
    "model": "qwen3.5",
    "model_server": os.getenv("VLLM_BASE", "http://192.168.4.66:8000/v1"),
    "api_key": "EMPTY",
    "generate_cfg": {
        "max_input_tokens": 0,
        "request_timeout": 1800,
        "temperature": 0.6,
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
MAX_TOOL_OUTPUT_TOKENS = int(os.getenv("MAX_TOOL_OUTPUT_TOKENS", "16000"))
MAX_CODE_OUTPUT_TOKENS = int(os.getenv("MAX_CODE_OUTPUT_TOKENS", "16000"))
CHARS_PER_TOKEN = 4  # Rough estimate for English/code; conservative for CJK

# --- Compaction (OpenClaw-style context management) ---
COMPACTION_ENABLED = os.getenv("COMPACTION_ENABLED", "true").lower() == "true"
COMPACTION_RESERVE_TOKENS = int(os.getenv("COMPACTION_RESERVE_TOKENS", "30000"))
COMPACTION_MAX_HISTORY_SHARE = 0.5         # max fraction of context window for history
COMPACTION_RECENT_TURNS_PRESERVE = 2       # keep last N user/assistant exchanges verbatim
COMPACTION_SAFETY_MARGIN = 1.2             # 20% buffer on token estimates
COMPACTION_TOOL_RESULT_MAX_SHARE = 0.30    # max 30% of context per tool result
COMPACTION_TOOL_RESULT_HARD_CAP = 40000    # 40k chars hard cap per tool result
COMPACTION_TOOL_RESULT_MIN_KEEP = 2000     # min 2k chars kept after truncation
COMPACTION_BASE_CHUNK_RATIO = 0.4          # base chunk size as fraction of context
COMPACTION_MIN_CHUNK_RATIO = 0.15          # minimum chunk ratio
COMPACTION_MAX_FAILURES = 8                # tool failures to extract
COMPACTION_FAILURE_CHARS = 240             # chars per failure summary
COMPACTION_MAX_IDENTIFIERS = 12            # unique identifiers to preserve
COMPACTION_TIMEOUT = int(os.getenv("COMPACTION_TIMEOUT", "120"))
COMPACTION_MODEL = os.getenv("COMPACTION_MODEL", "qwen3.6-27b-linux")
COMPACTION_URL = os.getenv("COMPACTION_URL", "http://192.168.4.66:8000")

TOOLS_API_BASE = os.getenv("TOOLS_API_BASE", "http://localhost:8080")
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "3600"))  # 1 hour
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

TOOL_LIST = [
    "web_search", "web_url_fetch", "stock_price",
    "schedule_task", "list_tasks", "complete_task", "cancel_task", "pause_task", "resume_task", "update_task_checkpoint",
    "update_soul", "update_heartbeat",
    "read_memories", "add_memory",
    "code_interpreter", "exec",
    "list_chat_logs", "read_chat_log",
    "create_project", "list_projects", "delete_project", "project_write_file",
    "project_read_file", "project_list_files", "project_delete_file", "project_apply_patch",
    "display_doc",
    "move_file", "delete_file",
    "request_user", "view_requests", "resolve_request",
    "start_pipeline", "start_trading_pipeline", "pipeline_status", "list_pipelines",
]

SYSTEM_PROMPT_SUFFIX = (
    "IMPORTANT: When processing web content or API data, always use code_interpreter to fetch and parse it. "
    "Never let raw web content enter the conversation — it wastes context tokens on every future turn. "
    "For bulk URL processing, write a Python loop in code_interpreter, not repeated tool calls."
)


def load_system_message() -> str:
    """Load SOUL.md + MEMORIES.md and combine with system prompt suffix.

    This is the base system message used for all sessions (main, heartbeat, cron).
    MEMORIES.md is included so the agent always has its learnings without needing
    to call read_memories as a tool.
    """
    soul_path = Path(__file__).parent / "SOUL.md"
    if soul_path.exists():
        soul = soul_path.read_text().strip()
    else:
        soul = "You are a capable research and task management assistant."

    # Load memories from DATA_DIR (agent-edited) or bundled default
    memories = ""
    memories_data = Path(DATA_DIR) / "MEMORIES.md"
    memories_bundled = Path(__file__).parent / "MEMORIES.md"
    for mp in [memories_data, memories_bundled]:
        if mp.exists():
            content = mp.read_text().strip()
            if content:
                memories = f"\n\n## Your Memories (auto-loaded from MEMORIES.md)\n\n{content}"
            break

    return soul + memories + "\n\n" + SYSTEM_PROMPT_SUFFIX


def session_metadata() -> str:
    """One-time metadata injected only into the main agent's first system message.

    Includes current date/time and location. NOT used for background sessions
    to keep their system prompts static for KV cache.
    """
    from datetime import datetime
    now = datetime.now()
    return (
        f"\n\n## Session Metadata\n"
        f"- Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
        f"- Location: San Mateo, California\n"
        f"- Timezone: US/Pacific"
    )
