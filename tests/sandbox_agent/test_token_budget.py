"""Tests for token budget management."""

import json

from qwen_agent.llm.schema import Message

from sandbox_agent.token_budget import (
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    trim_to_budget,
    truncate_output,
)


class TestEstimateTokens:

    def test_simple(self):
        # Real BPE now, not chars/4 — assert plausibility, not a fixed ratio
        # (repeated chars merge into few tokens; prose lands near chars/4).
        assert 1 <= estimate_tokens("a" * 20) <= 20
        prose = "The quick brown fox jumps over the lazy dog. " * 20
        assert 0.5 <= estimate_tokens(prose) / (len(prose) / 4) <= 2.0

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_monotonic(self):
        assert estimate_tokens("hello world " * 100) > estimate_tokens("hello world " * 10)


class TestTrimToBudget:

    def test_under_budget_unchanged(self):
        msgs = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        result = trim_to_budget(msgs, max_tokens=10000)
        assert len(result) == 3

    def test_drops_oldest_non_system(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="a" * 4000),  # ~1000 tokens
            Message(role="assistant", content="b" * 4000),
            Message(role="user", content="c" * 4000),
            Message(role="assistant", content="d" * 4000),
        ]
        # Budget of 2500 tokens — should keep system + last ~2 messages + trim note
        result = trim_to_budget(msgs, max_tokens=2500)
        assert len(result) < len(msgs)
        # System message preserved
        assert result[0].role == "system"
        assert result[0].content == "sys"

    def test_preserves_system_message(self):
        msgs = [
            Message(role="system", content="important system prompt " * 100),
            Message(role="user", content="hello"),
        ]
        result = trim_to_budget(msgs, max_tokens=100000)
        assert result[0].role == "system"
        assert "important" in result[0].content

    def test_adds_trim_note(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="a" * 4000),
            Message(role="assistant", content="b" * 4000),
            Message(role="user", content="c" * 400),
        ]
        result = trim_to_budget(msgs, max_tokens=500)
        # Should have a trim note
        # note is role="user" now: a second system message makes qwen_agent
        # reject the whole request (2cb762e), and this path can be persisted.
        notes = [m for m in result if m.role == "user" and "trimmed" in m.content]
        assert len(notes) == 1
        assert "earlier messages were trimmed" in notes[0].content

    def test_no_trim_note_when_under_budget(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
        ]
        result = trim_to_budget(msgs, max_tokens=100000)
        notes = [m for m in result if "trimmed" in (m.content if isinstance(m.content, str) else "")]
        assert len(notes) == 0


class TestTruncateOutput:

    def test_short_unchanged(self):
        assert truncate_output("hello", 100) == "hello"

    def test_truncates_long(self):
        text = "a" * 10000
        result = truncate_output(text, 100)  # 100 tokens = 400 chars
        assert len(result) < 600
        assert "TRUNCATED" in result

    def test_exact_boundary(self):
        # Boundary expressed via the calibrated divisor, not a hardcoded 4.0
        from sandbox_agent.config import CHARS_PER_TOKEN
        text = "a" * int(100 * CHARS_PER_TOKEN)
        result = truncate_output(text, 100)
        assert result == text  # exactly at budget — must not truncate


class TestEstimatorAccuracy:
    """2026-08-05: a live chat reached 281,152 real tokens while the estimator
    reported 183,684 (1.53x undercount) — the conversation blew past every
    budget because the ruler was wrong. Two root causes, both fixed here:
      1. function_call arguments were NEVER counted (51,179 tokens across 131
         tool calls in that thread — project_write_file passes whole files).
      2. chars/4 undercounts real content (~207k real vs ~180k estimated)."""

    def test_function_call_arguments_are_counted(self):
        from qwen_agent.llm.schema import FunctionCall, Message
        from sandbox_agent.token_budget import estimate_message_tokens
        big_args = json.dumps({"path": "x.py", "content": "print('hi')\n" * 4000})
        with_call = Message(role="assistant", content="",
                            function_call=FunctionCall(name="project_write_file",
                                                       arguments=big_args))
        bare = Message(role="assistant", content="")
        # the arguments must dominate — not be charged as ~10 tokens of overhead
        assert estimate_message_tokens(with_call) > estimate_message_tokens(bare) + 5_000

    def test_reasoning_content_is_counted(self):
        from qwen_agent.llm.schema import Message
        from sandbox_agent.token_budget import estimate_message_tokens
        m = Message(role="assistant", content="short")
        m.reasoning_content = "deliberation " * 2000
        assert estimate_message_tokens(m) > 2_000

    def test_divisor_is_calibrated_not_four(self):
        # 4.0 undercounted real agent traffic by 1.53x; 3.46 was measured on a
        # real 419-message thread. Real tokenization is NOT used here: tiktoken
        # takes 2.2s on 100k repetitive chars and crashes above ~1M.
        from sandbox_agent.config import CHARS_PER_TOKEN
        from sandbox_agent.token_budget import estimate_tokens
        assert 3.0 <= CHARS_PER_TOKEN <= 3.6
        assert estimate_tokens("x" * 35_000) > 35_000 // 4

    def test_never_raises_on_pathological_input(self):
        from sandbox_agent.token_budget import estimate_tokens
        assert estimate_tokens("x" * 2_000_000) > 0  # tokenizers choke here; we must not
