import os
from pathlib import Path

# Primary LLM: vLLM on 192.168.4.66 (single-queue, no concurrency)
# Used for main interactive conversation only
PRIMARY_LLM_CFG = {
    "model": "qwen3.5",
    "model_server": os.getenv("VLLM_BASE", "http://192.168.4.66:8000/v1"),
    "api_key": "EMPTY",
    "generate_cfg": {
        "max_input_tokens": 0,        # Disable client-side truncation for KV cache stability
        "request_timeout": 1800,       # 30min — covers full 200k context window worst case
        # Note: use_raw_api is NOT set (defaults to False) because vLLM does not support
        # native OpenAI tool_calls. Qwen-Agent uses prompt-based function calling instead,
        # injecting tool definitions into the system message and parsing calls from text.
    },
}

# Backup LLM: qwen3.5-27b on same vLLM server
# Reserved for user chat fallback when the big model is busy with background tasks.
# Background tasks NEVER use this — they always use the primary model.
BACKGROUND_LLM_CFG = {
    "model": "qwen3.5-27b",
    "model_server": os.getenv("VLLM_BASE", "http://192.168.4.66:8000/v1"),
    "api_key": "EMPTY",
    "generate_cfg": {
        "max_input_tokens": 0,
        "request_timeout": 1800,
    },
}

# Token budget — all models have 256k limit, we target 200k to leave room for generation
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "200000"))
MAX_TOOL_OUTPUT_TOKENS = int(os.getenv("MAX_TOOL_OUTPUT_TOKENS", "8000"))
MAX_CODE_OUTPUT_TOKENS = int(os.getenv("MAX_CODE_OUTPUT_TOKENS", "4000"))
CHARS_PER_TOKEN = 4  # Rough estimate for English/code; conservative for CJK

TOOLS_API_BASE = os.getenv("TOOLS_API_BASE", "http://localhost:8080")
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))  # 30 minutes
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

TOOL_LIST = [
    "web_search", "web_url_fetch", "stock_price",
    "schedule_task", "list_tasks", "complete_task", "cancel_task", "pause_task", "resume_task", "update_task_checkpoint",
    "update_soul", "update_heartbeat",
    "read_memories", "add_memory",
    "code_interpreter",
    "list_chat_logs", "read_chat_log",
    "create_project", "list_projects", "delete_project", "project_write_file",
    "project_read_file", "project_list_files", "project_delete_file",
    "move_file", "delete_file",
    "request_user", "view_requests", "resolve_request",
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
