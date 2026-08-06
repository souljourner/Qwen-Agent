"""Tests for mid-loop context compaction (weakness: background tasks hit the
model context limit).

Incident: pipeline stage sessions died with vLLM 400 (input 196609 + reserved
output 65536 > 262144) because compaction ran only ONCE before the fncall
loop and skipped short message lists, while tool results accumulated across
up to 20 LLM calls with no re-compaction.

Fixes under test:
- FnCallAgent._run calls an optional `_precall_compact` hook before EVERY
  LLM call, mutating the loop's message list in place.
- compact_midrun pins the head (system + first user message = the task) and
  compacts only the accumulated middle.
- maybe_compact no longer no-ops on short lists (a single oversized message
  is eligible).
- select_tier accounts for tool-schema/system overhead the char-estimator
  can't see.
- LLM cfgs carry a max_input_tokens hard backstop instead of 0 (disabled).
"""

import pytest

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, FunctionCall, Message

import sandbox_agent.compaction as compaction
from sandbox_agent.compaction import compact_midrun, maybe_compact
from sandbox_agent.config import COMPACTION_RESERVE_TOKENS, MAX_CONTEXT_TOKENS
from sandbox_agent.token_budget import estimate_messages_tokens

BUDGET = MAX_CONTEXT_TOKENS - COMPACTION_RESERVE_TOKENS

# Captured BEFORE the autouse identity-summarizer fixture patches it — the
# chunk-sizing test needs the real summarize_history pipeline.
import sandbox_agent.compaction.compactor as _compactor_mod
_REAL_SUMMARIZE_HISTORY = _compactor_mod.summarize_history


@pytest.fixture(autouse=True)
def _quiet_side_effects(monkeypatch):
    """No checkpoint files / status writes / LLM summarization during tests."""
    import sandbox_agent.compaction.checkpoint as checkpoint
    import sandbox_agent.compaction.compactor as compactor
    import sandbox_agent.model_tracker as model_tracker
    monkeypatch.setattr(checkpoint, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(model_tracker, "set_agent_status", lambda *a, **k: None)
    monkeypatch.setattr(model_tracker, "clear_agent_status", lambda *a, **k: None)
    # identity summarizer: deterministic; forces the trim fallback when
    # truncation alone can't fit (never calls the real LLM)
    monkeypatch.setattr(compactor, "summarize_history", lambda msgs: msgs)


class _ScriptedAgent(FnCallAgent):
    """FnCallAgent with a scripted LLM: N tool calls, then a final answer.
    Bypasses __init__ (no real LLM); records the estimated tokens of every
    message list actually sent to the 'model'."""

    def __init__(self, n_tool_calls: int, tool_result_chars: int):
        self.function_map = {}
        self.llm = None
        self.extra_generate_cfg = {}
        self._remaining = n_tool_calls
        self._tool_result_chars = tool_result_chars
        self.sent_tokens = []

    def _call_llm(self, messages, functions=None, stream=True, extra_generate_cfg=None):
        self.sent_tokens.append(estimate_messages_tokens(messages))
        if self._remaining > 0:
            self._remaining -= 1
            yield [Message(role=ASSISTANT, content="",
                           function_call=FunctionCall(name="big_tool", arguments="{}"),
                           extra={})]
        else:
            yield [Message(role=ASSISTANT, content="done", extra={})]

    def _call_tool(self, tool_name, tool_args=None, **kwargs):
        return "R" * self._tool_result_chars


def _run_agent(agent, messages):
    out = []
    for out in agent._run(messages):
        pass
    return out


def test_loop_recompacts_and_preserves_task():
    # 8 tool calls x ~150k chars ≈ 300k tokens of tool results — far over
    # budget without mid-loop compaction.
    agent = _ScriptedAgent(n_tool_calls=8, tool_result_chars=150_000)
    agent._precall_compact = compact_midrun
    system = Message(role=SYSTEM, content="You are the stage runner.")
    task = Message(role=USER, content="STAGE TASK: validate the strategy. " + "detail " * 500)
    _run_agent(agent, [system, task])

    assert len(agent.sent_tokens) == 9  # 8 tool rounds + final answer
    # Every request the "model" saw fit the compaction budget.
    assert max(agent.sent_tokens) <= BUDGET, agent.sent_tokens
    # Without the hook this is impossible: raw accumulation would be huge.
    assert sum(agent.sent_tokens[:1]) < BUDGET  # sanity: base prompt small


def test_head_survives_compaction_verbatim():
    system = Message(role=SYSTEM, content="SYSTEM-ANCHOR")
    task = Message(role=USER, content="TASK-ANCHOR " + "x" * 5000)
    mid = []
    for i in range(40):
        mid.append(Message(role=ASSISTANT, content="",
                           function_call=FunctionCall(name="t", arguments="{}"), extra={}))
        mid.append(Message(role=FUNCTION, name="t", content=f"result-{i} " + "y" * 60_000,
                           extra={}))
    out = compact_midrun([system, task] + mid)
    assert out[0].role == "system" and out[0].content == "SYSTEM-ANCHOR"
    assert out[1].role == "user" and out[1].content.startswith("TASK-ANCHOR")
    assert estimate_messages_tokens(out) <= BUDGET


def test_short_list_over_budget_is_compacted():
    # The old `len(messages) < 4` guard made short lists a no-op even when
    # massively over budget — exactly the shape of a fresh pipeline stage
    # after its first giant tool result.
    msgs = [
        Message(role=SYSTEM, content="sys"),
        Message(role=USER, content="task"),
        Message(role=FUNCTION, name="t", content="z" * 900_000, extra={}),
    ]
    before = estimate_messages_tokens(msgs)
    out = maybe_compact(msgs)
    assert estimate_messages_tokens(out) < before
    assert estimate_messages_tokens(out) <= BUDGET


def test_select_tier_accounts_for_overhead():
    from sandbox_agent.compaction.estimator import select_tier
    from sandbox_agent.config import ESTIMATOR_OVERHEAD_TOKENS
    assert ESTIMATOR_OVERHEAD_TOKENS >= 10_000
    # Estimate just under the budget WITHOUT overhead, over WITH it:
    # tokens*margin < budget  but  (tokens+overhead)*margin > budget.
    from sandbox_agent.config import COMPACTION_SAFETY_MARGIN
    target_tokens = int(BUDGET / COMPACTION_SAFETY_MARGIN) - ESTIMATOR_OVERHEAD_TOKENS // 2
    msgs = [Message(role=USER, content="a" * (target_tokens * 4))]
    tier, overflow = select_tier(msgs)
    assert tier != "fits"
    assert overflow > 0


def test_llm_cfg_backstop():
    from sandbox_agent.config import (BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG,
                                      PRIMARY_CONTEXT_TOKENS,
                                      SPILLABLE_CONTEXT_TOKENS)
    # Both tiers budgeted equally at 200k/160k. laguna's vendor-claimed 1M is
    # UNVALIDATED (128x YaRN off an 8k base, 36/48 sliding-window-512 layers,
    # no published long-context eval); measured clean at 85k, confabulating
    # at 275k-281k live on 2026-08-05. Raising these requires evidence.
    assert PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"] == 160_000
    assert BACKGROUND_LLM_CFG["generate_cfg"]["max_input_tokens"] == 160_000
    assert PRIMARY_LLM_CFG["context_window_tokens"] == 200_000
    assert BACKGROUND_LLM_CFG["context_window_tokens"] == 200_000
    # Size-based pinning must be unreachable: compaction caps every history
    # at budget (170k) < threshold, so no conversation gets trapped on a tier.
    assert SPILLABLE_CONTEXT_TOKENS >= PRIMARY_CONTEXT_TOKENS


class TestPerTierBudgets:
    """Compaction honors a per-tier context budget (passed explicitly here —
    the 900k values exercise the PARAMETER, not the deployed config, which
    now budgets both tiers at 200k) while chunk sizing stays pinned to the
    SUMMARIZER's window."""

    def _big_history(self, n_tokens):
        # Multi-turn shape: segment_messages keeps the last 2 USER exchanges
        # verbatim, so real user turns are needed for a summarizable history.
        msgs = [Message(role=SYSTEM, content="sys")]
        chunk = "w" * 40_000  # 10k tokens per message
        for i in range(n_tokens // 10_000):
            if i % 4 == 0:
                msgs.append(Message(role=USER, content=f"question {i}"))
            msgs.append(Message(role=ASSISTANT, content=""))
            msgs.append(Message(role=FUNCTION, name="t", content=chunk, extra={}))
        return msgs

    def test_context_tokens_honored(self):
        msgs = self._big_history(500_000)  # ~500k tokens
        out = maybe_compact(msgs, context_tokens=900_000)
        assert out is msgs  # fits the laguna budget — untouched
        out2 = maybe_compact(list(msgs))
        assert estimate_messages_tokens(out2) <= BUDGET  # default budget compacts

    def test_chunks_never_exceed_summarizer_window(self, monkeypatch):
        from sandbox_agent.config import COMPACTION_RESERVE_TOKENS as RESERVE
        from sandbox_agent.config import SUMMARIZER_CONTEXT_TOKENS
        import sandbox_agent.compaction.compactor as compactor
        import sandbox_agent.compaction.summarizer as summarizer
        sizes = []

        def fake_chunk(text):
            sizes.append(len(text) // 4)
            return "summary"

        # restore the REAL summarize_history (autouse fixture stubs it) and
        # stub only the LLM boundary
        monkeypatch.setattr(compactor, "summarize_history", _REAL_SUMMARIZE_HISTORY)
        monkeypatch.setattr(summarizer, "summarize_chunk", fake_chunk)
        monkeypatch.setattr(summarizer, "merge_summaries",
                            lambda summaries, identifiers: "merged summary")
        # over the laguna budget → summarization path with the BIG budget
        msgs = self._big_history(1_000_000)
        maybe_compact(msgs, context_tokens=900_000)
        assert sizes, "summarizer was never invoked"
        cap = SUMMARIZER_CONTEXT_TOKENS - RESERVE
        assert max(sizes) <= cap, f"chunk of {max(sizes)} tokens exceeds summarizer window"

    def test_exception_fallback_respects_budget(self, monkeypatch):
        # A compaction exception on a big-budget history must NOT trim to the
        # 200k default (that would amputate 600k tokens of pinned history).
        import sandbox_agent.compaction.estimator as estimator

        calls = {"n": 0}
        real_select = estimator.select_tier

        def exploding_select(messages, context_tokens=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_select(messages, context_tokens)
            raise RuntimeError("boom mid-compaction")

        monkeypatch.setattr(estimator, "select_tier", exploding_select)
        # 1M tokens: over even the laguna budget, so compaction runs and the
        # second select_tier (post-truncation re-check) explodes
        msgs = self._big_history(1_000_000)
        out = maybe_compact(msgs, context_tokens=900_000)
        # trimmed to the LAGUNA budget (fits 870k → basically untouched),
        # not to the 200k default
        assert estimate_messages_tokens(out) > 300_000


class TestRequestTimeoutScaling:

    def test_small_contexts_unchanged(self):
        from sandbox_agent.token_budget import compute_request_timeout
        small = [Message(role=USER, content="x" * 4_000)]  # ~1k tokens
        assert 600 <= compute_request_timeout(small) < 700

    def test_large_context_scales_to_3600(self):
        from sandbox_agent.token_budget import compute_request_timeout
        big = [Message(role=USER, content="x" * 3_600_000)]  # ~900k tokens
        assert compute_request_timeout(big) == 3600


class TestCompactionNotice:
    """The chat UI gets a heads-up when a summarization-tier compaction is
    about to make the user wait (it can run for minutes: cold-prefill of the
    whole history through the summarizer). Truncation-only tiers are fast —
    no notice."""

    def _register(self, monkeypatch):
        events = []
        compaction.register_compaction_notice_hook(lambda payload: events.append(payload))
        return events

    def _big_summarize_history(self):
        msgs = [Message(role=SYSTEM, content="sys")]
        for i in range(30):
            msgs.append(Message(role=USER, content=f"q{i}"))
            msgs.append(Message(role=ASSISTANT, content="a " * 22_000))  # ~11k tokens; no fn msgs → no truncation tier
        return msgs

    def test_notice_fired_for_summarize_tier(self, monkeypatch):
        events = self._register(monkeypatch)
        try:
            maybe_compact(self._big_summarize_history())
        finally:
            compaction.unregister_compaction_notice_hook()
        phases = [e["phase"] for e in events]
        assert "start" in phases and "done" in phases
        start = events[phases.index("start")]
        assert start["est_tokens"] > 0
        assert start["est_minutes"] >= 1

    def test_no_notice_when_fits(self):
        events = self._register(None)
        try:
            maybe_compact([Message(role=USER, content="small")])
        finally:
            compaction.unregister_compaction_notice_hook()
        assert events == []

    def test_no_notice_for_truncation_only(self):
        # one giant tool result → tier 1 handles it fast; no scary banner
        events = self._register(None)
        msgs = [
            Message(role=SYSTEM, content="s"),
            Message(role=USER, content="t"),
            Message(role=FUNCTION, name="t", content="z" * 900_000, extra={}),
        ]
        try:
            maybe_compact(msgs)
        finally:
            compaction.unregister_compaction_notice_hook()
        assert events == []

    def test_hook_is_per_thread(self):
        import threading
        events = []
        compaction.register_compaction_notice_hook(lambda p: events.append(p))
        try:
            seen_in_thread = []

            def other_thread():
                # no hook registered on THIS thread → nothing fires here
                maybe_compact(self._big_summarize_history())
                seen_in_thread.append(len(events))

            before = len(events)
            t = threading.Thread(target=other_thread)
            t.start()
            t.join()
            assert len(events) == before  # other thread's compaction: silent here
        finally:
            compaction.unregister_compaction_notice_hook()


class TestSummaryMessageRole:
    """Regression (found in in-band planning, 2026-08-01): the summary was
    emitted as a SECOND system message — qwen_agent's input validation
    (base.py: 'no more than one system message') rejects the very next
    request, so summarize-tier compaction poisoned the conversation. The
    validator only runs with max_input_tokens > 0, i.e. this was armed by
    the truncation-backstop change."""

    def _reassembled(self):
        from sandbox_agent.compaction.compactor import reassemble
        system = [Message(role=SYSTEM, content="the real system prompt")]
        recent = [Message(role=USER, content="latest q"),
                  Message(role=ASSISTANT, content="latest a")]
        return reassemble(system, "DIGEST-BODY", recent, 42)

    def test_exactly_one_system_message(self):
        out = self._reassembled()
        assert [m.role for m in out].count("system") == 1
        assert out[0].role == "system"
        summary = out[1]
        assert summary.role == "user"
        assert "DIGEST-BODY" in str(summary.content)
        assert "[Context compacted" in str(summary.content)

    def test_survives_qwen_agent_input_validation(self):
        from qwen_agent.llm.base import _truncate_input_messages_roughly
        out = self._reassembled()
        # must NOT raise 'no more than one system message'
        result = _truncate_input_messages_roughly(out, max_tokens=160_000)
        assert result
