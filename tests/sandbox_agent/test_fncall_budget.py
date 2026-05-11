"""Tests for the LOCAL MOD in qwen_agent/agents/fncall_agent.py: graceful
tool-call budget exhaustion (budget-pressure note on tool results + wrap-up
summary call when MAX_LLM_CALL_PER_RUN is spent mid-chain).
"""

import pytest

import qwen_agent.agents.fncall_agent as fa
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from qwen_agent.tools.base import BaseTool


class _NoopTool(BaseTool):
    name = "noop"
    description = "no-op tool for tests"
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params, **kwargs):
        return "noop result"


class _StubLLM:
    """First `tool_calls_before_answer` LLM calls (those made WITH tools)
    return a `noop` tool call; after that, a plain text answer. A call made
    with `functions=None` (the wrap-up call) always returns the summary text."""

    def __init__(self, tool_calls_before_answer: int):
        self.generate_cfg = {}
        self.model = "test-stub-model"  # FnCallAgent.__init__ inspects self.llm.model
        self._n_tool_calls = 0
        self._budget = tool_calls_before_answer
        self.wrap_up_called = False

    def chat(self, messages, functions=None, stream=True, extra_generate_cfg=None):
        if functions is None:
            self.wrap_up_called = True
            yield [Message(role=ASSISTANT, content="did some stuff; X still remains; here's the wrap-up")]
            return
        # tools present
        if self._n_tool_calls < self._budget:
            self._n_tool_calls += 1
            yield [Message(role=ASSISTANT, content="",
                           function_call=FunctionCall(name="noop", arguments="{}"),
                           extra={"function_id": f"fc{self._n_tool_calls}"})]
        else:
            yield [Message(role=ASSISTANT, content="All done — here is the final answer.")]


def _make_agent(stub: _StubLLM) -> FnCallAgent:
    return FnCallAgent(function_list=[_NoopTool()], llm=stub, name="t")


def _drain(agent, messages):
    last = []
    for last in agent._run(messages):
        pass
    return last


def _assistant_text(msgs):
    return "\n".join(
        m.content for m in msgs
        if getattr(m, "role", None) == ASSISTANT and isinstance(getattr(m, "content", None), str) and m.content
    )


@pytest.fixture
def low_budget(monkeypatch):
    # `_run` reads the bare name imported into fncall_agent's namespace.
    monkeypatch.setattr(fa, "MAX_LLM_CALL_PER_RUN", 4)
    monkeypatch.setattr(fa, "_BUDGET_PRESSURE_AT", 3)
    yield


@pytest.fixture
def high_budget(monkeypatch):
    monkeypatch.setattr(fa, "MAX_LLM_CALL_PER_RUN", 50)
    yield


def test_budget_exhaustion_emits_wrapup(low_budget):
    stub = _StubLLM(tool_calls_before_answer=10)  # never reaches a plain answer within 4 calls
    agent = _make_agent(stub)
    final = _drain(agent, [Message(role="user", content="do a long thing")])

    # The wrap-up (tool-less) call was made
    assert stub.wrap_up_called

    # Last message is an assistant message starting with the deterministic marker
    last = final[-1]
    assert getattr(last, "role", None) == ASSISTANT
    assert isinstance(last.content, str)
    assert last.content.startswith(fa._BUDGET_EXHAUSTED_MARKER)

    # The marker carries both phrases stage_runner._detect_part_completion greps for
    lowered = last.content.lower()
    assert "part-completion" in lowered
    assert "ran out of tool calls" in lowered


def test_budget_pressure_note_on_tool_results(low_budget):
    stub = _StubLLM(tool_calls_before_answer=10)
    agent = _make_agent(stub)
    final = _drain(agent, [Message(role="user", content="do a long thing")])

    fn_msgs = [m for m in final if getattr(m, "role", None) == FUNCTION]
    assert fn_msgs, "expected at least one tool-result message"
    # With cap=4 and pressure threshold=3, the budget note appears on the
    # tool result(s) produced once 3 or fewer LLM calls remain.
    assert any(isinstance(m.content, str) and m.content.startswith("[budget:") for m in fn_msgs)


def test_detect_part_completion_recognizes_wrapup(low_budget):
    from sandbox_agent.pipeline.stage_runner import _detect_part_completion
    from sandbox_agent.pipeline.models import PipelineState

    stub = _StubLLM(tool_calls_before_answer=10)
    agent = _make_agent(stub)
    final = _drain(agent, [Message(role="user", content="do a long thing")])
    text = _assistant_text(final)

    # _detect_part_completion(state, stage_number, result_text) — only result_text matters here.
    state = PipelineState(project_name="x", description="d", pipeline_type="startup", current_stage=2)
    assert _detect_part_completion(state, 2, text) is True


def test_normal_completion_no_wrapup(high_budget):
    """Under a comfortable budget, the model gives a plain answer — no budget
    note, no wrap-up call. The unchanged path."""
    stub = _StubLLM(tool_calls_before_answer=2)  # 2 tool calls then a plain answer on call 3
    agent = _make_agent(stub)
    final = _drain(agent, [Message(role="user", content="quick thing")])

    assert not stub.wrap_up_called
    last = final[-1]
    assert getattr(last, "role", None) == ASSISTANT
    assert last.content == "All done — here is the final answer."
    assert not any(
        isinstance(m.content, str) and m.content.startswith("[budget:")
        for m in final if getattr(m, "role", None) == FUNCTION
    )
