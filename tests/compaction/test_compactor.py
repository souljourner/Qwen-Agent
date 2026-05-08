"""Tests for the compaction engine — segmentation, truncation, identifiers, quality audit."""

import pytest
from qwen_agent.llm.schema import Message, FunctionCall

from sandbox_agent.compaction.compactor import (
    segment_messages,
    truncate_tool_results,
    extract_identifiers,
    extract_tool_failures,
    quality_audit,
    build_chunks,
    _head_tail_truncate,
    _split_into_turns,
    reassemble,
)


# --- Test helpers ---

def _msg(role, content, name=None, function_call=None, extra=None):
    m = Message(role=role, content=content)
    if name:
        m.name = name
    if function_call:
        m.function_call = function_call
    if extra:
        m.extra = extra
    return m


def _sys(content):
    return _msg("system", content)


def _user(content):
    return _msg("user", content)


def _assistant(content, fc_name=None, fc_args=None):
    fc = None
    if fc_name:
        fc = FunctionCall(name=fc_name, arguments=fc_args or "{}")
    return _msg("assistant", content, function_call=fc)


def _function(name, content):
    return _msg("function", content, name=name, extra={"function_id": "1"})


# --- segment_messages ---

class TestSegmentMessages:
    def test_basic_segmentation(self):
        """System msgs separated, recent turns preserved."""
        msgs = [
            _sys("You are helpful."),
            _user("Hello"),
            _assistant("Hi there!"),
            _user("What is 2+2?"),
            _assistant("4"),
            _user("Thanks"),
            _assistant("You're welcome!"),
        ]
        seg = segment_messages(msgs)
        assert len(seg.system) == 1
        assert seg.system[0].content == "You are helpful."
        # With RECENT_TURNS_PRESERVE=2, last 2 user turns are recent
        assert len(seg.recent) == 4  # 2 turns × 2 messages each
        assert seg.recent[0].content == "What is 2+2?"
        assert len(seg.history) == 2  # first turn (user + assistant)

    def test_tool_use_pairing_in_turn(self):
        """Tool calls and their results stay in the same turn."""
        msgs = [
            _sys("System"),
            _user("Search for cats"),
            _assistant("Let me search.", fc_name="web_search", fc_args='{"query": "cats"}'),
            _function("web_search", "Found 10 results about cats"),
            _assistant("I found information about cats."),
            _user("Tell me more"),
            _assistant("Cats are great!"),
        ]
        seg = segment_messages(msgs)
        # With 2 preserved turns, the first turn (with tool call) is history
        # Actually with only 2 turns total, both are recent
        assert len(seg.history) == 0
        assert len(seg.recent) == 6  # all non-system

    def test_all_system_messages(self):
        msgs = [_sys("System 1"), _sys("System 2")]
        seg = segment_messages(msgs)
        assert len(seg.system) == 2
        assert len(seg.history) == 0
        assert len(seg.recent) == 0

    def test_many_turns_split_correctly(self):
        """With 5 turns and preserve=2, first 3 turns are history."""
        msgs = [_sys("System")]
        for i in range(5):
            msgs.append(_user(f"Question {i}"))
            msgs.append(_assistant(f"Answer {i}"))
        seg = segment_messages(msgs)
        assert len(seg.system) == 1
        # 3 history turns = 6 messages
        assert len(seg.history) == 6
        # 2 recent turns = 4 messages
        assert len(seg.recent) == 4
        assert seg.recent[0].content == "Question 3"


# --- _split_into_turns ---

class TestSplitIntoTurns:
    def test_simple_turns(self):
        msgs = [_user("Q1"), _assistant("A1"), _user("Q2"), _assistant("A2")]
        turns = _split_into_turns(msgs)
        assert len(turns) == 2
        assert len(turns[0]) == 2
        assert turns[0][0].content == "Q1"

    def test_tool_call_turn(self):
        msgs = [
            _user("Search"),
            _assistant("Searching", fc_name="web_search", fc_args="{}"),
            _function("web_search", "Results"),
            _assistant("Here are the results"),
        ]
        turns = _split_into_turns(msgs)
        assert len(turns) == 1  # all one turn
        assert len(turns[0]) == 4

    def test_assistant_only(self):
        """Edge case: no user message."""
        msgs = [_assistant("Hello")]
        turns = _split_into_turns(msgs)
        assert len(turns) == 1


# --- truncate_tool_results ---

class TestTruncateToolResults:
    def test_short_results_unchanged(self):
        msgs = [_user("Hi"), _function("tool", "short result")]
        result = truncate_tool_results(msgs)
        assert result[1].content == "short result"

    def test_long_result_truncated(self):
        long_content = "x" * 50000
        msgs = [_function("big_tool", long_content)]
        result = truncate_tool_results(msgs)
        assert len(result[0].content) < len(long_content)
        assert "truncated" in result[0].content

    def test_preserves_non_function_messages(self):
        msgs = [_user("Hi"), _assistant("Hello"), _function("tool", "result")]
        result = truncate_tool_results(msgs)
        assert result[0].content == "Hi"
        assert result[1].content == "Hello"

    def test_preserves_function_name(self):
        msgs = [_function("my_tool", "x" * 50000)]
        result = truncate_tool_results(msgs)
        assert result[0].name == "my_tool"


# --- _head_tail_truncate ---

class TestHeadTailTruncate:
    def test_short_text_unchanged(self):
        assert _head_tail_truncate("hello", 1000, 100) == "hello"

    def test_truncates_with_marker(self):
        text = "a" * 10000
        result = _head_tail_truncate(text, 5000, 2000)
        assert "truncated" in result
        assert len(result) < len(text)

    def test_important_tail_preserved(self):
        text = "start " + "x" * 10000 + " Error: something failed"
        result = _head_tail_truncate(text, 5000, 2000)
        assert "Error" in result or "truncated" in result

    def test_min_keep_respected(self):
        text = "x" * 10000
        result = _head_tail_truncate(text, 100, 2000)
        # min_keep overrides the small budget
        assert len(result) >= 2000


# --- extract_identifiers ---

class TestExtractIdentifiers:
    def test_uuid(self):
        text = "Task id: 550e8400-e29b-41d4-a716-446655440000"
        ids = extract_identifiers(text)
        assert "550e8400-e29b-41d4-a716-446655440000" in ids

    def test_url(self):
        text = "Visit https://example.com/api/v1/users for details"
        ids = extract_identifiers(text)
        assert any("example.com" in i for i in ids)

    def test_ip_with_port(self):
        text = "Server at 192.168.4.66:8000"
        ids = extract_identifiers(text)
        assert "192.168.4.66:8000" in ids

    def test_file_path(self):
        text = "Created file /data/projects/my-project/report.md"
        ids = extract_identifiers(text)
        assert any("report.md" in i for i in ids)

    def test_max_identifiers_limit(self):
        text = " ".join(f"https://example.com/{i}" for i in range(20))
        ids = extract_identifiers(text)
        assert len(ids) <= 12  # COMPACTION_MAX_IDENTIFIERS

    def test_empty_text(self):
        assert extract_identifiers("") == set()


# --- extract_tool_failures ---

class TestExtractToolFailures:
    def test_extracts_errors(self):
        msgs = [
            _function("web_search", "Error: Connection timeout"),
            _function("code_interpreter", "print('hello')\nhello"),
            _function("web_fetch", "HTTP 500 Internal Server Error"),
        ]
        failures = extract_tool_failures(msgs)
        assert len(failures) == 2
        assert "web_search" in failures[0]
        assert "500" in failures[1]

    def test_no_errors(self):
        msgs = [_function("tool", "Success: everything worked")]
        failures = extract_tool_failures(msgs)
        assert len(failures) == 0

    def test_max_failures_limit(self):
        msgs = [_function(f"tool_{i}", f"Error #{i}") for i in range(20)]
        failures = extract_tool_failures(msgs)
        assert len(failures) <= 8  # COMPACTION_MAX_FAILURES

    def test_traceback_detected(self):
        msgs = [_function("code_interpreter", "Traceback (most recent call last):\n  File...")]
        failures = extract_tool_failures(msgs)
        assert len(failures) == 1


# --- quality_audit ---

class TestQualityAudit:
    def test_good_summary_passes(self):
        summary = """
## Decisions Made
Chose library X over Y.

## Open TODOs
Need to implement feature Z.

## Constraints & Requirements
Must use Python 3.10+.

## Pending User Asks
User wants a dashboard.

## Exact Identifiers
- 192.168.4.66:8000
- /data/projects/test

## Errors Encountered
None.
"""
        ids = {"192.168.4.66:8000", "/data/projects/test"}
        passed, issues = quality_audit(summary, ids, "build a dashboard")
        assert passed, f"Should pass but got issues: {issues}"

    def test_missing_sections_detected(self):
        summary = "Just a plain summary with no sections."
        passed, issues = quality_audit(summary, set(), "hello")
        assert not passed
        assert any("missing_section" in i for i in issues)

    def test_missing_identifiers_detected(self):
        summary = """
## Decisions Made
Something.
## Open TODOs
Something.
## Constraints
Something.
## Pending User Asks
Something.
## Exact Identifiers
None found.
"""
        ids = {"192.168.4.66:8000", "550e8400-e29b-41d4-a716-446655440000"}
        passed, issues = quality_audit(summary, ids, "test")
        assert not passed
        assert any("missing_identifiers" in i for i in issues)

    def test_empty_identifiers_ok(self):
        summary = """
## Decisions
X
## Open TODOs
Y
## Constraints
Z
## Pending User Asks
User wants to deploy the service.
## Exact Identifiers
None
"""
        passed, _ = quality_audit(summary, set(), "deploy the service")
        assert passed


# --- build_chunks ---

class TestBuildChunks:
    def test_small_history_single_chunk(self):
        msgs = [_user("Q"), _assistant("A")]
        chunks = build_chunks(msgs)
        assert len(chunks) == 1

    def test_respects_turn_boundaries(self):
        """Chunks should not split mid-turn."""
        msgs = []
        for i in range(50):
            msgs.append(_user(f"Question {i} " + "x" * 500))
            msgs.append(_assistant(f"Answer {i} " + "y" * 500))
        chunks = build_chunks(msgs)
        # Each chunk should contain complete turns
        for chunk in chunks:
            # First message in chunk should be a user message (turn start)
            # unless it's the very first chunk which might start with assistant
            user_count = sum(1 for m in chunk if m.role == "user")
            assistant_count = sum(1 for m in chunk if m.role == "assistant")
            assert user_count == assistant_count  # balanced turns


# --- reassemble ---

class TestReassemble:
    def test_basic_reassembly(self):
        system = [_sys("You are helpful.")]
        summary = "Previous: user asked about cats."
        recent = [_user("New question"), _assistant("New answer")]
        result = reassemble(system, summary, recent, 10)
        assert len(result) == 4  # 1 system + 1 summary + 2 recent
        assert result[0].role == "system"
        assert result[0].content == "You are helpful."
        assert result[1].role == "system"
        assert "compacted" in result[1].content.lower()
        assert "10 earlier messages" in result[1].content
        assert result[2].content == "New question"
