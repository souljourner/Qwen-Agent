"""Tests for token budget management."""

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
        # 20 chars / 4 = 5 tokens
        assert estimate_tokens("a" * 20) == 5

    def test_empty(self):
        assert estimate_tokens("") == 0


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
        notes = [m for m in result if m.role == "system" and "trimmed" in m.content]
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
        text = "a" * 400  # exactly 100 tokens
        result = truncate_output(text, 100)
        assert result == text  # Should not truncate
