"""Tests for compaction tier selection."""

from qwen_agent.llm.schema import Message

from sandbox_agent.compaction.estimator import select_tier


def _msg(role, content):
    return Message(role=role, content=content)


class TestSelectTier:
    def test_small_messages_fit(self):
        msgs = [_msg("user", "Hello"), _msg("assistant", "Hi")]
        tier, overflow = select_tier(msgs)
        assert tier == "fits"
        assert overflow == 0

    def test_large_messages_trigger_compaction(self):
        # ~200k tokens = ~800k chars. Create messages exceeding the budget.
        big_content = "x" * 900000  # ~225k tokens, well over budget with safety margin
        msgs = [_msg("user", big_content)]
        tier, overflow = select_tier(msgs)
        assert tier != "fits"
        assert overflow > 0

    def test_tool_heavy_messages_suggest_truncation(self):
        # Many large function results that could be truncated
        msgs = [_msg("user", "query")]
        for i in range(20):
            m = _msg("function", "x" * 50000)
            m.name = f"tool_{i}"
            msgs.append(m)
        tier, overflow = select_tier(msgs)
        if tier != "fits":
            # Should suggest tool truncation since there's lots of reducible content
            assert tier in ("truncate_tools", "compact_and_truncate")

    def test_no_tool_results_suggest_compact(self):
        # Large user/assistant messages with no function results
        msgs = []
        for i in range(100):
            msgs.append(_msg("user", f"Question {i} " + "detail " * 500))
            msgs.append(_msg("assistant", f"Answer {i} " + "explanation " * 500))
        tier, overflow = select_tier(msgs)
        if tier != "fits":
            assert tier == "compact"
