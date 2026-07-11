# Token Efficiency — Deep Dive
> llm_call/llm_batch details, VLLM_BASE runtime code + enable_thinking footgun, batch-URL pattern, directory-sizing protocol.

## Files as scratch space — worked example
Inside code_interpreter, pre-configured path variables `DATA_DIR` and `PROJECTS_DIR` are available.

```python
# Example: fetch URLs, process with llm_call, save results — only print summary
urls = [...]
with open(f'{PROJECTS_DIR}/my-project/data/pages.jsonl', 'w') as f:
    for url in urls:
        html = requests.get(url, timeout=30).text[:4000]
        f.write(json.dumps({'url': url, 'html': html}) + '\n')

results = []
with open(f'{PROJECTS_DIR}/my-project/data/pages.jsonl') as f:
    for line in f:
        page = json.loads(line)
        insight = llm_call(f'Extract key facts:\n{page["html"]}', system='Return 2-3 bullet points.')
        results.append({'url': page['url'], 'insight': insight})

with open(f'{PROJECTS_DIR}/my-project/research/analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Processed {len(results)} URLs. Full results saved.')
for r in results[:3]:
    print(f'  - {r["insight"][:80]}')
```

## web_url_fetch pagination
web_url_fetch paginates long pages: pass `max_chars` to cap how much comes back, and `offset` to continue. If a result ends with a `[web_url_fetch: ... offset=N ...]` note, call it again with that `offset` (same url) to read the rest — but for anything large prefer pulling it inside code_interpreter so it never enters context.

## llm_call() — per-item LLM reasoning inside code_interpreter
Inside code_interpreter, `llm_call(prompt, system='', think=False)` calls a background LLM.
Use this when you need the LLM to reason about, extract from, or classify individual pieces of content.
Each llm_call() runs on the background model — it does NOT add tokens to your main conversation.

## llm_batch() — parallel labeling; never write a sequential loop
For 2+ items sharing the same system prompt (classify, extract, score), use:
```python
from sandbox_agent.tools.llm_client import llm_batch
results = llm_batch(system="...", prompts=[...], max_concurrent=8)
```
Returns a list[str] in input order. Runs in parallel AND hits vLLM's prefix cache on the shared system prompt → roughly an order of magnitude faster than `for x in items: llm_call(...)`. Bridge `llm_call()` is serialized at the HTTP layer, so threading it does NOT parallelize. The vLLM primary has 15 concurrent slots shared with user chat — keep `max_concurrent` ≤ 8 unless you know chat is idle.

## Project code that needs an LLM at runtime — use VLLM_BASE
`llm_call`/`llm_batch` are for YOUR scratch reasoning inside code_interpreter. When you're WRITING code into a project (e.g. `mvp/generation.py`, a backtest script, a paper-trading worker) that needs to call an LLM at the project's own runtime, do NOT hard-code an OpenAI/Anthropic key. Use the local vLLM endpoint already in env. **Always use the `openai` Python SDK (it's installed) — do NOT hand-roll the HTTP request with `requests`/`httpx`.**
```python
import os, openai
client = openai.OpenAI(base_url=os.environ["VLLM_BASE"], api_key="EMPTY")
resp = client.chat.completions.create(
    model="qwen3.6-27b-linux",
    messages=[{"role":"system","content":"..."},{"role":"user","content":"..."}],
    temperature=0.6,
    # Disable reasoning for fast, direct answers (classification, extraction,
    # formatting). Omit this line for complex synthesis where the model
    # benefits from thinking out loud.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
text = resp.choices[0].message.content
```
The `openai` package is already installed in the runtime image. `VLLM_BASE` resolves to the same vLLM the agent itself uses. Models: `qwen3.6-27b-linux` (primary, supports concurrency), `qwen3.5` (397B, slower, prefer for complex reasoning). Both honor `enable_thinking`. Streaming: pass `stream=True` and the same `extra_body` together. `enable_thinking=False` is per-call.

**Hugging Face:** `HF_TOKEN` (and the legacy alias `HUGGINGFACE_HUB_TOKEN`) is set if the operator configured it — `huggingface_hub` / `transformers` / `datasets` pick it up automatically for gated model downloads, no per-call wiring needed. `HF_HOME=/app/data/hf_cache` so downloaded models persist across container rebuilds (it's on the DATA_DIR bind mount, visible on the host). If `HF_TOKEN` isn't set, anonymous downloads still work for public models.

**Footgun — `enable_thinking` and `extra_body`:** what vLLM actually reads is a TOP-LEVEL request body field `chat_template_kwargs`. The OpenAI SDK's `extra_body={...}` works because the SDK *spreads its contents into the top level of the wire body* — that's its whole purpose. If you ignore the rule above and hand-roll with `requests`/`httpx`, putting `"extra_body": {"chat_template_kwargs": {...}}` in the JSON does NOTHING (vLLM sees an unknown `extra_body` key and drops it → thinking stays ON → the model burns your whole `max_tokens` budget on `<think>…` tokens and `content` comes back empty). Hand-rolled, it must be `json={"model":..., "messages":..., "chat_template_kwargs": {"enable_thinking": False}}` — top level, no `extra_body` wrapper. But really: just use the SDK.

## Batch URL processing — MANDATORY pattern
When you need to process multiple URLs (2 or more):
1. Write ONE code_interpreter call containing the entire workflow as a Python script.
2. Fetch all URLs with requests.get() and write raw content to a .jsonl file.
3. Read from the file, process each item with llm_call(), write results to another file.
4. print() ONLY a 3-5 line summary at the end. Save the full report to a file.
5. NEVER print raw HTML, page content, or full results. NEVER make separate code_interpreter calls per URL.

## Check size before listing or dumping a directory
Never run a bare `ls`, `find`, `cat`, `grep -r`, or `project_list_files` on a directory whose contents you haven't bounded. A single dir can hold thousands of files (corpora, raw scrapes, llm_cache), and the full listing ends up in your context window for the rest of the conversation.

**The protocol — two steps, not one:**
1. **Size it first**: `ls <dir> | wc -l` (or `find <dir> -type f | wc -l` for recursive). Also `du -sh <dir>` if the file count isn't the concern but per-file size is. This returns a single number, not a listing.
2. **Then list wisely**:
   - Small (≤ 50 files) → full `ls <dir>` is fine.
   - Medium (50-500) → `ls <dir> | head -30` and `ls <dir> | tail -5` to sample both ends, or `ls <dir> | sort -R | head -20` for a random sample.
   - Large (> 500) → do NOT list. Write a summary step: filter with glob (`ls <dir>/*.json | wc -l`), aggregate with Python (`Counter` of extensions / prefixes), or read a manifest file if one exists (`cat <dir>/manifest.json | jq '.[:5]'`).

Same rule for file contents: check `wc -l <file>` and `du -h <file>` before `cat`. For anything over a few hundred lines, read by ranges (`sed -n '1,50p'`, `head`, `tail`) or parse structurally (`jq`, `python -c`).

**Shared corpora** live at `/app/data/shared/` — these are frequently thousands of files. Always consult the companion manifest (e.g., `/app/data/shared/filings/prem14a_manifest.json`) rather than enumerating the directory directly.
