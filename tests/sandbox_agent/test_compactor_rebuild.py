"""Tests for the compactor rebuild (2026-08-06).

Incident being fixed: a 281k-token chat compacted to 13.4k ("it knows nothing
about the past") and then re-compacted on the next turn. Root causes found:

  1. _messages_to_text truncated every message before the summarizer saw it
     (tool results 8k chars, user/assistant 4k, function_call args 500) while
     the chunker MEASURED them at full size — a ~50x accounting gap, so the
     summarizer received confetti.
  2. summarize_history only aborted when ALL chunks failed; a partial failure
     silently deleted the un-summarized remainder.
  3. Compaction targeted 170k while the model's own truncation backstop sat
     BELOW it at 160k, and that truncation deletes oldest-first — so a
     persisted digest (position 0) would be the first thing destroyed.
"""

import json

import pytest

from qwen_agent.llm.schema import ASSISTANT, FUNCTION, USER, FunctionCall, Message

import sandbox_agent.compaction.compactor as compactor


def _turn(i, tool_chars=30_000, args_chars=20_000):
    """One realistic turn: user ask, assistant tool call with big args, result."""
    return [
        Message(role=USER, content=f"user question {i} " + f"u{i} " * 200),
        Message(role=ASSISTANT, content=f"reasoning about {i} " + f"a{i} " * 200,
                function_call=FunctionCall(
                    name="project_write_file",
                    arguments=json.dumps({"path": f"n{i}.md", "content": f"C{i}" * (args_chars // 2)}))),
        Message(role=FUNCTION, name="project_write_file",
                content=f"tool output {i} " + f"t{i} " * (tool_chars // 4)),
    ]


class TestRenderPreservesMaterial:
    """The summarizer must receive the conversation, not a stub of it."""

    def test_render_retains_most_characters(self):
        msgs = [m for i in range(4) for m in _turn(i)]
        raw_chars = sum(len(str(m.content or "")) +
                        len(str(getattr(m.function_call, "arguments", "") or ""))
                        for m in msgs)
        rendered = compactor._messages_to_text(msgs)
        # Old behavior kept ~2%. Anything below 80% means we're summarizing
        # confetti and the digest cannot be faithful.
        assert len(rendered) >= 0.8 * raw_chars, (
            f"rendered {len(rendered):,} of {raw_chars:,} chars "
            f"({len(rendered)/raw_chars:.1%}) — summarizer is being starved")

    def test_function_call_arguments_survive_rendering(self):
        msgs = _turn(7)
        rendered = compactor._messages_to_text(msgs)
        # the file body lived only in function_call.arguments (was cut at 500)
        assert rendered.count("C7") > 1_000

    def test_char_budget_is_respected_when_given(self):
        msgs = [m for i in range(4) for m in _turn(i)]
        rendered = compactor._messages_to_text(msgs, char_budget=50_000)
        assert len(rendered) <= 52_000  # budget + truncation marker slack


class TestPartialFailureIsAllOrNothing:
    """A failed chunk must never mean 'delete that part of the history'."""

    def test_partial_chunk_failure_leaves_history_untouched(self, monkeypatch):
        msgs = [m for i in range(12) for m in _turn(i)]
        calls = {"n": 0}

        def flaky(text):
            calls["n"] += 1
            return "" if calls["n"] % 2 == 0 else "## Decisions Made\nsomething"

        import sandbox_agent.compaction.summarizer as summarizer
        monkeypatch.setattr(summarizer, "summarize_chunk", flaky)
        monkeypatch.setattr(summarizer, "merge_summaries", lambda s, i: "merged")
        # force multiple chunks
        monkeypatch.setattr(compactor, "build_chunks",
                            lambda history: [history[i:i + 3] for i in range(0, len(history), 3)])
        out = compactor.summarize_history(msgs)
        assert out is msgs, "a partial summarization must abort, not half-delete"

    def test_all_chunks_succeed_produces_digest(self, monkeypatch):
        msgs = [m for i in range(6) for m in _turn(i)]
        import sandbox_agent.compaction.summarizer as summarizer
        monkeypatch.setattr(summarizer, "summarize_chunk", lambda t: "## Decisions Made\nok")
        monkeypatch.setattr(summarizer, "merge_summaries", lambda s, i: "merged digest")
        out = compactor.summarize_history(msgs)
        assert any("[Context compacted" in str(m.content) for m in out)


class TestSummarizerRequestShape:

    def test_sends_max_tokens_but_never_temperature(self, monkeypatch):
        import sandbox_agent.compaction.summarizer as summarizer
        captured = {}

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json=None, timeout=None, **kw):
            captured.update(json or {})
            return _Resp()

        monkeypatch.setattr(summarizer.requests, "post", fake_post)
        summarizer.summarize_chunk("some conversation text")
        # bounded output so a rambling digest can't blow the budget...
        assert captured.get("max_tokens", 0) > 0
        # ...but temperature stays a HOST default (standing instruction).
        assert "temperature" not in captured


class TestBudgetOrdering:
    """target < trigger < hard <= max_input_tokens — so the model's own
    truncation backstop never fires on healthy compacted output, and can
    never delete the digest (which sits at position 0)."""

    def test_ordering_invariant(self):
        from sandbox_agent.compaction.budget import derive_budgets
        b = derive_budgets(200_000, 160_000)
        assert b.target < b.trigger <= b.hard <= 160_000

    def test_hard_stays_under_model_input_cap(self):
        from sandbox_agent.compaction.budget import derive_budgets
        from sandbox_agent.config import BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG
        for cfg in (PRIMARY_LLM_CFG, BACKGROUND_LLM_CFG):
            b = derive_budgets(cfg["context_window_tokens"],
                               cfg["generate_cfg"]["max_input_tokens"])
            assert b.hard <= cfg["generate_cfg"]["max_input_tokens"]

    def test_compacted_output_survives_qwen_agent_truncation(self):
        """The F2 guard: a history at TARGET must pass through qwen_agent's
        real truncation with the digest intact. Before this ordering fix a
        170k-target history hit a 160k backstop that deletes oldest-first —
        and the digest is oldest."""
        from qwen_agent.llm.base import _truncate_input_messages_roughly
        from sandbox_agent.compaction.budget import derive_budgets
        from sandbox_agent.config import PRIMARY_LLM_CFG
        from sandbox_agent.token_budget import estimate_messages_tokens

        b = derive_budgets(PRIMARY_LLM_CFG["context_window_tokens"],
                           PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"])
        digest = Message(role=USER, content="[Context compacted: 400 messages]\n\nDURABLE facts")
        msgs = [digest]
        while estimate_messages_tokens(msgs) < b.target:
            msgs.append(Message(role=USER, content="q " + "x" * 8_000))
            msgs.append(Message(role=ASSISTANT, content="a " + "y" * 8_000))

        out = _truncate_input_messages_roughly(
            msgs, max_tokens=PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"])
        assert any("[Context compacted" in str(m.content) for m in out), \
            "digest was deleted by the model's own truncation"
        assert len(out) == len(msgs), "truncation dropped messages at TARGET"


class TestAggregateToolBudget:

    def test_many_results_share_one_budget(self):
        msgs = []
        for i in range(60):
            msgs.append(Message(role=ASSISTANT, content="",
                                function_call=FunctionCall(name="t", arguments="{}")))
            msgs.append(Message(role=FUNCTION, name="t", content=f"r{i} " + "z" * 40_000))
        out = compactor.truncate_tool_results(msgs, total_budget_chars=200_000)
        total = sum(len(str(m.content)) for m in out if m.role == FUNCTION)
        # old per-result cap let 60x40k = 2.4M chars all pass "within cap"
        assert total < 600_000, f"aggregate budget not enforced: {total:,} chars"

    def test_newest_results_keep_most(self):
        msgs = []
        for i in range(30):
            msgs.append(Message(role=FUNCTION, name="t", content=f"r{i} " + "z" * 30_000))
        out = [m for m in compactor.truncate_tool_results(msgs, total_budget_chars=150_000)
               if m.role == FUNCTION]
        assert len(str(out[-1].content)) > len(str(out[0].content))

    def test_truncation_is_idempotent(self):
        msgs = [Message(role=FUNCTION, name="t", content="z" * 300_000)]
        once = compactor.truncate_tool_results(msgs, total_budget_chars=50_000)
        twice = compactor.truncate_tool_results(once, total_budget_chars=50_000)
        assert str(once[0].content) == str(twice[0].content)
