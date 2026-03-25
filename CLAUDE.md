# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install (minimal)
pip install -e .

# Install with all optional features
pip install -e ".[gui,rag,code_interpreter,mcp]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/agents/test_assistant.py

# Run a single test function
pytest tests/agents/test_assistant.py::test_assistant_system_and_tool

# Lint and format (requires pre-commit installed)
pre-commit run --all-files

# Individual linters
flake8 qwen_agent/ tests/
isort qwen_agent/ tests/ --line-length 120
yapf --style "{based_on_style: google, column_limit: 120}" -i -r qwen_agent/
```

Note: Many tests require `DASHSCOPE_API_KEY` and are skipped without it.

## Architecture

### Core Hierarchy

Three base abstractions with registry-based plugin systems:

- **Agent** (`qwen_agent/agent.py`): Abstract base. Subclass and implement `_run(messages, lang) -> Iterator[List[Message]]`. Public API is `run()` (streaming) and `run_nonstream()`.
- **BaseTool** (`qwen_agent/tools/base.py`): Register with `@register_tool('name')`, implement `call(params)`.
- **BaseChatModel** (`qwen_agent/llm/base.py`): Register with `@register_llm('type')`, implement `_chat_stream()`.

### Agent Inheritance

```
Agent (ABC)
├── BasicAgent          — simple LLM wrapper, no tools
├── FnCallAgent         — function-calling loop (detect tool call → execute → repeat)
│   ├── Assistant       — adds RAG/file support via Memory
│   ├── ReActChat       — ReAct-format tool calling
│   └── TIRMathAgent    — tool-integrated reasoning
├── Memory              — RAG and file management agent
├── GroupChat           — multi-agent orchestration (MultiAgentHub)
└── Router              — routes queries to sub-agents
```

### Message Schema (`qwen_agent/llm/schema.py`)

Pydantic models: `Message` (role, content, function_call, reasoning_content), `ContentItem` (text/image/file/audio/video — mutually exclusive), `FunctionCall` (name, arguments as JSON string). Messages use roles: system, user, assistant, function.

### Key Patterns

- **Streaming generators**: All `_run()` methods yield `List[Message]`. Each yield is the current state (partial during tool execution).
- **Function call loop** (`FnCallAgent`): LLM → detect tool → execute → append function-role message → repeat. Max iterations: `MAX_LLM_CALL_PER_RUN` (default 20, configurable via `QWEN_AGENT_MAX_LLM_CALL_PER_RUN` env var).
- **Registry pattern**: Tools use `TOOL_REGISTRY` + `@register_tool()`, LLMs use `LLM_REGISTRY` + `@register_llm()`. Agents instantiate by name lookup.
- **Dict-based configuration**: Agents accept `llm=dict`, `function_list=list`, `system_message=str`. Tool entries can be strings (registry names), dicts, or BaseTool instances.
- **Input type preservation**: `run()` accepts both plain dicts and Message objects; output type matches input type.

### Settings (`qwen_agent/settings.py`)

All overridable via `QWEN_AGENT_*` environment variables (e.g., `QWEN_AGENT_DEFAULT_MAX_INPUT_TOKENS`).

## Code Style

- **Formatter**: YAPF with Google style, 120-column limit
- **Imports**: isort, 120-character line length
- **Strings**: Double quotes (enforced by pre-commit `double-quote-string-fixer`)
- **Linter**: flake8 with max line length 300
