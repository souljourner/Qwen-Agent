"""Tests for sandbox_agent.chat_logger.

Targets the bug where today's chat_logs/{date}.md only captured user lines —
agent responses, tool calls, and tool results never landed. Verifies the
rewritten log_turn writes every channel of a realistic streamed response.
"""

import os
from unittest.mock import patch

import pytest

from qwen_agent.llm.schema import ContentItem, FunctionCall, Message

# Patch DATA_DIR via env BEFORE importing chat_logger (it caches CHAT_LOGS_DIR
# at import time from sandbox_agent.config.DATA_DIR).
@pytest.fixture
def chat_logger_with_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("sandbox_agent.chat_logger.CHAT_LOGS_DIR", str(tmp_path / "chat_logs"))
    import sandbox_agent.chat_logger as cl_mod
    return cl_mod, tmp_path / "chat_logs"


def _read_log(log_dir):
    files = list(log_dir.glob("*.md"))
    assert len(files) == 1, f"expected one log file, got {[f.name for f in files]}"
    return files[0].read_text()


# ---------------------------------------------------------------------------
# _format_content
# ---------------------------------------------------------------------------

class TestFormatContent:

    def test_string_content(self, chat_logger_with_tmp_dir):
        cl_mod, _ = chat_logger_with_tmp_dir
        msg = Message(role="user", content="hello world")
        assert cl_mod._format_content(msg) == "hello world"

    def test_list_of_content_items(self, chat_logger_with_tmp_dir):
        cl_mod, _ = chat_logger_with_tmp_dir
        msg = Message(role="user", content=[
            ContentItem(text="line one"),
            ContentItem(text="line two"),
        ])
        assert cl_mod._format_content(msg) == "line one\nline two"

    def test_dict_input(self, chat_logger_with_tmp_dir):
        cl_mod, _ = chat_logger_with_tmp_dir
        assert cl_mod._format_content({"role": "user", "content": "hi"}) == "hi"

    def test_dict_list_content(self, chat_logger_with_tmp_dir):
        cl_mod, _ = chat_logger_with_tmp_dir
        assert cl_mod._format_content({"content": [{"text": "a"}, {"text": "b"}]}) == "a\nb"

    def test_none_content(self, chat_logger_with_tmp_dir):
        cl_mod, _ = chat_logger_with_tmp_dir
        msg = Message(role="assistant", content=None)
        assert cl_mod._format_content(msg) == ""

    def test_empty_string_content(self, chat_logger_with_tmp_dir):
        cl_mod, _ = chat_logger_with_tmp_dir
        msg = Message(role="assistant", content="")
        assert cl_mod._format_content(msg) == ""


# ---------------------------------------------------------------------------
# log_turn — real-world streaming response shapes
# ---------------------------------------------------------------------------

class TestLogTurn:

    def test_text_only_reply(self, chat_logger_with_tmp_dir):
        cl_mod, log_dir = chat_logger_with_tmp_dir
        cl_mod.log_turn(
            Message(role="user", content="what is 2+2?"),
            [Message(role="assistant", content="4")],
        )
        text = _read_log(log_dir)
        assert "User" in text
        assert "what is 2+2?" in text
        assert "Agent" in text
        assert "4" in text

    def test_tool_call_then_final_text(self, chat_logger_with_tmp_dir):
        """The exact shape FnCallAgent yields: an assistant tool-call
        message, the function result, and a final assistant text message."""
        cl_mod, log_dir = chat_logger_with_tmp_dir
        cl_mod.log_turn(
            Message(role="user", content="list the projects"),
            [
                Message(
                    role="assistant",
                    content="",
                    function_call=FunctionCall(name="list_projects", arguments="{}"),
                ),
                Message(role="function", name="list_projects", content="Project A\nProject B"),
                Message(role="assistant", content="You have two projects: A and B."),
            ],
        )
        text = _read_log(log_dir)
        assert "list the projects" in text
        assert "Tool call: list_projects" in text
        assert "Tool: list_projects result" in text
        assert "Project A" in text
        assert "You have two projects: A and B." in text

    def test_reasoning_plus_text(self, chat_logger_with_tmp_dir):
        """Qwen-agent's openai client splits reasoning_content and content
        into TWO separate assistant messages within a single yield."""
        cl_mod, log_dir = chat_logger_with_tmp_dir
        cl_mod.log_turn(
            Message(role="user", content="hard question"),
            [
                Message(role="assistant", content="", reasoning_content="step by step..."),
                Message(role="assistant", content="The answer is 42."),
            ],
        )
        text = _read_log(log_dir)
        assert "Agent thinking" in text
        assert "step by step..." in text
        assert "The answer is 42." in text

    def test_assistant_message_with_only_function_call_is_logged(self, chat_logger_with_tmp_dir):
        """Regression: previously, assistant messages with empty content but a
        function_call were silently skipped. They must now log a Tool call line."""
        cl_mod, log_dir = chat_logger_with_tmp_dir
        cl_mod.log_turn(
            Message(role="user", content="run the thing"),
            [
                Message(
                    role="assistant",
                    content="",
                    function_call=FunctionCall(name="exec", arguments='{"cmd":"ls"}'),
                ),
            ],
        )
        text = _read_log(log_dir)
        assert "Tool call: exec" in text
        assert "ls" in text

    def test_long_tool_result_is_truncated(self, chat_logger_with_tmp_dir):
        cl_mod, log_dir = chat_logger_with_tmp_dir
        big = "x" * 5000
        cl_mod.log_turn(
            Message(role="user", content="dump it"),
            [
                Message(role="function", name="bigtool", content=big),
            ],
        )
        text = _read_log(log_dir)
        assert "truncated in log" in text
        assert text.count("x") < 5000  # got cut down

    def test_dict_messages_work_too(self, chat_logger_with_tmp_dir):
        """Resumed history may pass dicts instead of Message objects."""
        cl_mod, log_dir = chat_logger_with_tmp_dir
        cl_mod.log_turn(
            {"role": "user", "content": "hi"},
            [{"role": "assistant", "content": "hello back"}],
        )
        text = _read_log(log_dir)
        assert "hi" in text
        assert "hello back" in text

    def test_appends_to_existing_file(self, chat_logger_with_tmp_dir):
        cl_mod, log_dir = chat_logger_with_tmp_dir
        cl_mod.log_turn(Message(role="user", content="first"), [Message(role="assistant", content="A")])
        cl_mod.log_turn(Message(role="user", content="second"), [Message(role="assistant", content="B")])
        text = _read_log(log_dir)
        # Only one header for the day, but both turns present
        assert text.count("# Chat Log") == 1
        assert "first" in text and "second" in text
        assert "A" in text and "B" in text

    def test_empty_response_messages(self, chat_logger_with_tmp_dir):
        """If the agent run produced nothing, still log the user message."""
        cl_mod, log_dir = chat_logger_with_tmp_dir
        cl_mod.log_turn(Message(role="user", content="nothing happened"), [])
        text = _read_log(log_dir)
        assert "nothing happened" in text
