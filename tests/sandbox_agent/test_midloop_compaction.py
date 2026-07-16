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
    from sandbox_agent.config import BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG
    # 160k, not 190k: the char-heuristics undercounted by ~16% in the real
    # incident, so the backstop needs real margin below the 196,608 ceiling.
    assert PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"] == 160_000
    assert BACKGROUND_LLM_CFG["generate_cfg"]["max_input_tokens"] == 160_000
