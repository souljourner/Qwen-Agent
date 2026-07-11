# Sandbox Agent — Known Weaknesses to Resolve

Honest assessment as of 2026-07-11, ordered by importance. Each item: the weakness, why it matters, and what "resolved" looks like. Update statuses as work lands.

## 1. The agent grades its own homework — NOT RESOLVED
The trading pipeline now has programmatic gates (metrics sanity, hash-pinned verdicts, lineage checks) because it kept fabricating — but everything else the agent produces (research reports, world updates, chat answers, digest summaries) is unverified LLM output. Where an LLM quality check exists (stage evaluator), the same model family judges its own work. The fabrication incidents (md5 "signals", invented filings, impossible backtests) weren't trading-specific; they're what the model does under pressure to produce *something*. One pasture is fenced; the pattern is everywhere else.

**Resolved looks like:** every recurring work product has at least one deterministic ground-truth check (source citations that resolve, numbers recomputed from data, claims spot-checked against fetched sources); LLM-judged acceptance uses adversarial framing ("try to refute") rather than confirmation.

## 2. It can't remember its own conclusions — RESOLVED 2026-07-11
Memory is a capped flat file plus daily chat logs. `on_chat_resume` drops all tool-gathered context (the ~44k ceiling; agent re-reads everything after any reload). No session search — "what did we decide about X three weeks ago?" means grepping markdown by hand. Knowledge doesn't flow between projects; lessons die with the project unless `add_memory` happened to fire.

Done: (a) per-thread **history sidecar** (`chat_history.py`) persists the agent's exact Message list — incl. tool results and multimodal content — after every turn; resume loads it verbatim. Threads predating the sidecar fall back to chat.db reconstruction with tool-pair replay (newest-first, ~100k-token cap). (b) **`session_search` tool**: FTS5 index over all chat steps + completed-task results, incrementally refreshed, with SOUL guidance to search before asking the user to repeat. (c) **Cross-project learnings**: stage-6 review `## Learnings` auto-extracted to `learnings/pipeline-learnings.md` and injected into every new pipeline's stage-1 prompt. 21 tests.

**Remaining niggle:** the search index covers chat + tasks, not project files; reasoning steps are (deliberately) not indexed.

## 3. One process, one GPU, one point of failure — NOT RESOLVED
Chat, cron, heartbeat, pipelines, and the LLM bridge share a single Python process in one container (8 GB Docker VM, backtests as child processes). One OOM or wedged thread degrades everything at once. Both models share one GPU, so the 397B "fallback" is slowest exactly when the primary is saturated. Everything hangs off one vLLM box; no cloud failover.

**Resolved looks like:** background lanes supervised/restartable independently of chat (separate processes or a supervisor); an emergency LLM fallback that doesn't share the GPU; memory limits on pipeline children.

## 4. Capability has outgrown the security model — NOT RESOLVED
The agent has `exec` with network, a browser **with a credential store**, `send_email`, and a readable `.env` with real credentials. The sanitizer covers fetched text, but browser screenshots feed the vision model unsanitized, and a prompt-injection → exec/send_email exfiltration chain genuinely exists. Real defense today is single-user-on-LAN.

**Resolved looks like:** tool-approval tiers for destructive/outbound ops (send_email to new recipients, credential reads, off-LAN egress); browser-page content treated as untrusted in prompts; secrets moved out of world-readable files.

## 5. Failures don't teach it anything — RESOLVED 2026-07-11
Was: every failure class repeated until a human read the logs (QueuePool errors ran for days; the agent's own SMTP request sat unread 3 weeks; stage-6 "instruction improvement suggestions" were structurally unappliable — the agent user can't write /app).

Done: `sandbox_agent/health.py` — deterministic hourly scan (failure-event bursts in activity.jsonl, task-queue corruption, request_user items pending >24h) that **emails the user directly** (code-triggered, fingerprint-deduped, max 1/day per condition) and injects "System Health" investigate-and-fix items into the heartbeat. Stage instructions now honor `DATA_DIR/pipeline_stages/<type>/` overrides, and pipeline completion files a one-shot heartbeat item to apply the review's suggestions. 11 tests.

**Remaining niggle:** thresholds are static; no trend detection (slow degradation below threshold stays invisible).

## 6. Tests validate parts, not the machine — NOT RESOLVED
574 tests, nearly all unit-level. The worst bugs found this year (phantom promote, runaway retries, state-file divergence, resume context loss) emerged from *interactions* unit tests structurally can't catch. No end-to-end pipeline run with a stubbed LLM; no CI — the suite runs when someone remembers.

**Resolved looks like:** one integration test driving a full trading pipeline with a scripted fake LLM through promote AND reject paths; tests wired to a pre-commit hook or scheduled task.

## 7. Consistency by hand — PARTIALLY MITIGATED
SOUL, six skills, 40+ tool descriptions, and acceptance criteria must agree with each other and the code, maintained manually; drift has repeatedly shipped (SOUL described behavior that didn't exist; `update_soul` silently never took effect for months). Some locks exist (skills index ↔ files test, SOUL size cap test), most don't.

**Resolved looks like:** tests asserting tool descriptions match registered schemas and SOUL claims match config values (timeouts, caps, tool lists); a doc-drift check in the heartbeat.

---

Recommended order of attack: **#6 next** (locks in the last six months of fixes), then #4, #3, #1, #7.
