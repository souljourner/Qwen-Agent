"""Standalone LLM client for subprocesses.

Importable from any Python process (including those launched via the `exec`
tool) without pulling in the rest of the agent. Lets backtest / data-pipeline
scripts do their repetitive LLM loop in Python code — not by burning the
agent's tool-call budget — while still hitting the same model backend the
agent uses.

Environment variables (resolved in this order):
    LLM_CALL_CHAIN — JSON list like '[{"model":"qwen3.6-27b-linux","base":"http://…/v1"}, …]'.
                     Each entry is tried in order until one returns non-empty.
    VLLM_BASE       — OpenAI-compatible base URL. If set and LLM_CALL_CHAIN is
                     not, the default chain is primary (qwen3.6-27b-linux) then
                     backup (qwen3.5 397B) on that base.
    LLM_CALL_BASE + LLM_CALL_MODEL — legacy single-endpoint override. If set,
                     it becomes a single-entry chain.

If none are set, calls raise RuntimeError at invocation time.
"""

import concurrent.futures
import json
import os
from typing import List, Optional, Tuple

import requests

_SESSION = requests.Session()

_DEFAULT_MODELS = ("qwen3.6-27b-linux", "qwen3.5-9b")


def _resolve_chain() -> List[Tuple[str, str]]:
    """Return the ordered [(model, base_url), ...] chain to try."""
    raw = os.environ.get("LLM_CALL_CHAIN", "").strip()
    if raw:
        try:
            items = json.loads(raw)
            chain = [(it["model"], it["base"].rstrip("/")) for it in items if it.get("model") and it.get("base")]
            if chain:
                return chain
        except (ValueError, KeyError, TypeError):
            pass

    single_base = os.environ.get("LLM_CALL_BASE", "").strip()
    single_model = os.environ.get("LLM_CALL_MODEL", "").strip()
    if single_base and single_model:
        return [(single_model, single_base.rstrip("/"))]

    vllm_base = os.environ.get("VLLM_BASE", "").strip()
    if vllm_base:
        base = vllm_base.rstrip("/")
        return [(m, base) for m in _DEFAULT_MODELS]

    raise RuntimeError(
        "No LLM endpoint configured — set LLM_CALL_CHAIN, or VLLM_BASE, "
        "or LLM_CALL_BASE + LLM_CALL_MODEL."
    )


def llm_call(
    prompt: str,
    system: str = "",
    temperature: float = 0.6,
    timeout: int = 300,
    max_tokens: Optional[int] = None,
) -> str:
    """Single LLM completion. Tries each model in the fallback chain in order,
    retrying once per model on an empty completion. Raises RuntimeError if
    every endpoint fails or returns empty."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err: Optional[BaseException] = None
    for model, base in _resolve_chain():
        for attempt in (1, 2):
            try:
                body = {"model": model, "messages": messages, "temperature": temperature}
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens
                resp = _SESSION.post(f"{base}/chat/completions", json=body, timeout=timeout)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content") or ""
                if content.strip():
                    return content
                last_err = RuntimeError(f"{model} @ {base}: empty completion (attempt {attempt})")
            except Exception as e:
                last_err = e
                break  # HTTP/JSON error — skip the retry and move to the next model

    raise RuntimeError(f"All LLM endpoints failed. Last error: {last_err}")


def llm_batch(
    system: str,
    prompts: List[str],
    max_concurrent: int = 4,
    temperature: float = 0.6,
    timeout: int = 300,
    max_tokens: Optional[int] = None,
) -> List[str]:
    """Parallel LLM calls with a shared `system` prefix.

    Why: every call has the same system prompt prefix → vLLM/Ollama KV cache
    hits on every call after the first, giving large speedups on classification
    / labeling workloads.

    Returns a list of completions in the same order as `prompts`. Individual
    failures surface as exceptions on the failing slot.
    """
    def _one(prompt: str) -> str:
        return llm_call(prompt, system=system, temperature=temperature, timeout=timeout, max_tokens=max_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        return list(pool.map(_one, prompts))
