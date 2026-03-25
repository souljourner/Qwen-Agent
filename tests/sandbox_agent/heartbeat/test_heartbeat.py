"""Tests for the heartbeat runner."""

import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from qwen_agent.llm.schema import Message

from sandbox_agent.heartbeat.heartbeat_runner import (
    HeartbeatRunner,
    _extract_response_text,
    _is_heartbeat_ok,
    _load_heartbeat_md,
    _parse_pending_items,
)
from sandbox_agent.scheduler.task_queue import TaskQueue


class TestParseItems:

    def test_parses_unchecked(self):
        md = "# Checklist\n- [ ] Do thing A\n- [x] Done B\n- [ ] Do thing C"
        items = _parse_pending_items(md)
        assert items == ["Do thing A", "Do thing C"]

    def test_empty_checklist(self):
        md = "# Empty\nNothing here"
        assert _parse_pending_items(md) == []

    def test_all_checked(self):
        md = "- [x] Done A\n- [x] Done B"
        assert _parse_pending_items(md) == []


class TestHeartbeatOk:

    def test_ok_signal(self):
        assert _is_heartbeat_ok("HEARTBEAT_OK") is True

    def test_ok_with_whitespace(self):
        assert _is_heartbeat_ok("  HEARTBEAT_OK  \n") is True

    def test_not_ok_long_response(self):
        text = "HEARTBEAT_OK " + "x" * 400
        assert _is_heartbeat_ok(text) is False

    def test_not_ok_no_signal(self):
        assert _is_heartbeat_ok("I found an issue with task X") is False


class TestExtractResponseText:

    def test_string_content(self):
        msgs = [Message(role="assistant", content="hello")]
        assert _extract_response_text(msgs) == "hello"

    def test_skips_non_assistant(self):
        msgs = [
            Message(role="function", content="tool result", name="web_search"),
            Message(role="assistant", content="the answer"),
        ]
        assert _extract_response_text(msgs) == "the answer"


class TestLoadHeartbeatMd:

    def test_loads_bundled_default(self):
        md = _load_heartbeat_md("/nonexistent")
        assert "Heartbeat Checklist" in md

    def test_loads_custom(self):
        tmp = tempfile.mkdtemp()
        try:
            with open(f"{tmp}/HEARTBEAT.md", "w") as f:
                f.write("# Custom\n- [ ] Custom task")
            md = _load_heartbeat_md(tmp)
            assert "Custom task" in md
        finally:
            shutil.rmtree(tmp)


class TestHeartbeatRunner:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def task_queue(self, tmp_dir):
        return TaskQueue(data_dir=tmp_dir)

    def test_nothing_to_check_skips_llm(self, tmp_dir, task_queue):
        """When there are no pending items and no due tasks, skip the LLM call."""
        # Write a checklist with all items checked
        with open(f"{tmp_dir}/HEARTBEAT.md", "w") as f:
            f.write("# Checklist\n- [x] All done")

        mock_factory = MagicMock()
        runner = HeartbeatRunner(
            agent_factory=mock_factory,
            task_queue=task_queue,
            data_dir=tmp_dir,
        )
        result = runner.run_once()
        assert result is None
        mock_factory.assert_not_called()  # Should not create an agent

    def test_heartbeat_ok_suppressed(self, tmp_dir, task_queue):
        """When agent responds HEARTBEAT_OK, result should be None (suppressed)."""
        with open(f"{tmp_dir}/HEARTBEAT.md", "w") as f:
            f.write("- [ ] Check something")

        mock_agent = MagicMock()
        mock_agent.run.return_value = iter([
            [Message(role="assistant", content="HEARTBEAT_OK")]
        ])

        runner = HeartbeatRunner(
            agent_factory=lambda: mock_agent,
            task_queue=task_queue,
            data_dir=tmp_dir,
        )
        result = runner.run_once()
        assert result is None

    def test_alert_returned(self, tmp_dir, task_queue):
        """When agent finds something, the alert text is returned."""
        with open(f"{tmp_dir}/HEARTBEAT.md", "w") as f:
            f.write("- [ ] Check disk space")

        mock_agent = MagicMock()
        mock_agent.run.return_value = iter([
            [Message(role="assistant", content="WARNING: Disk space is at 95% capacity. Immediate action required.")]
        ])

        runner = HeartbeatRunner(
            agent_factory=lambda: mock_agent,
            task_queue=task_queue,
            data_dir=tmp_dir,
        )
        result = runner.run_once()
        assert result is not None
        assert "Disk space" in result

    def test_due_tasks_included_in_message(self, tmp_dir, task_queue):
        """Due tasks from TaskQueue should be included in the heartbeat check."""
        from datetime import datetime, timedelta

        with open(f"{tmp_dir}/HEARTBEAT.md", "w") as f:
            f.write("- [ ] Check tasks")

        # Add a due task
        task_queue.add_task(
            name="check stocks", description="Check AAPL price",
            schedule_type="at", run_at=datetime.now() - timedelta(seconds=1),
        )

        mock_agent = MagicMock()
        mock_agent.run.return_value = iter([
            [Message(role="assistant", content="HEARTBEAT_OK")]
        ])

        runner = HeartbeatRunner(
            agent_factory=lambda: mock_agent,
            task_queue=task_queue,
            data_dir=tmp_dir,
        )
        runner.run_once()

        # Verify the agent was called with a message containing the due task
        call_args = mock_agent.run.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        user_msg = messages[0].content
        assert "check stocks" in user_msg
