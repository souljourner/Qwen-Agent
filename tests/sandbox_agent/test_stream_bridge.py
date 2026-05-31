"""Tests for _StreamBridge's per-chunk work.

Root cause of the "streaming gets progressively slower within a multi-tool
turn, then a fresh message is fast again" lag: the worker pushes the FULL
cumulative List[Message] on every chunk, and `consume()` re-scans all of them.
`_close_tool_with_result` had no "already applied" guard, so every finished
tool result was re-`update()`d (socket emit + DB persist of the whole step) on
EVERY subsequent chunk. With N accumulated tool results that's O(N) redundant
persists per chunk — the event loop saturates and token streaming crawls.
"""

# Neutralize chat_app's import-time bootstrap (it starts cron/heartbeat/bridge
# threads). Must run before importing chat_app.
import sandbox_agent.main as _main  # noqa: E402
_main.bootstrap_background = lambda *a, **k: None

import asyncio  # noqa: E402

import chainlit as cl  # noqa: E402

from sandbox_agent.chat_app import _StreamBridge  # noqa: E402
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message  # noqa: E402


class _FakeStep:
    def __init__(self, **kw):
        self.name = kw.get("name")
        self.input = kw.get("input")
        self.output = None
        self.start = None
        self.end = None
        self.id = f"step-{id(self)}"
        self.update_calls = 0
        self.send_calls = 0

    async def send(self):
        self.send_calls += 1
        return self

    async def update(self):
        self.update_calls += 1
        return self


class _FakeMessage:
    def __init__(self, **kw):
        self.content = kw.get("content", "")
        self.id = f"msg-{id(self)}"
        self.update_calls = 0
        self.send_calls = 0
        self.tokens = []

    async def send(self):
        self.send_calls += 1
        return self

    async def update(self):
        self.update_calls += 1
        return self

    async def stream_token(self, token, is_sequence=False):
        # Mirror Chainlit's Message.stream_token: appends to content.
        self.tokens.append(token)
        if is_sequence:
            self.content = token
        else:
            self.content += token
        return None


def test_finished_tool_result_not_reupdated_each_chunk(monkeypatch):
    monkeypatch.setattr(cl, "Step", _FakeStep)
    monkeypatch.setattr(cl, "Message", _FakeMessage)

    bridge = _StreamBridge()
    fc = FunctionCall(name="search", arguments='{"q":"x"}')
    a_call = Message(role=ASSISTANT, content="", function_call=fc)
    result = Message(role=FUNCTION, name="search", content="R" * 500)

    asyncio.run(bridge.consume([a_call]))            # tool call appears
    asyncio.run(bridge.consume([a_call, result]))    # result appears → applied once

    tool_steps = [s["obj"] for s in bridge._by_index.values() if s.get("kind") == "tool"]
    assert len(tool_steps) == 1
    step = tool_steps[0]
    assert step.output == "R" * 500
    assert step.end is not None
    baseline = step.update_calls

    # The answer text now streams in over several chunks; the tool result is
    # unchanged in every one of them.
    for partial in ("Hel", "Hello", "Hello wo", "Hello world", "Hello world."):
        text = Message(role=ASSISTANT, content=partial)
        asyncio.run(bridge.consume([a_call, result, text]))

    extra = step.update_calls - baseline
    assert extra == 0, f"finished tool step re-update()d {extra} extra times across 5 chunks"


def test_pending_tail_flushes_before_footer(monkeypatch):
    """The per-turn stats footer must land at the END of the message, not spliced
    into the middle. The last streamed tokens sit in the batch buffer when a turn
    ends; flush_pending_text() must move them into the message BEFORE the footer
    is appended, else content becomes [text-minus-tail] + [footer] + [tail]."""
    import sandbox_agent.chat_app as ca
    monkeypatch.setattr(cl, "Step", _FakeStep)
    monkeypatch.setattr(cl, "Message", _FakeMessage)

    clock = {"t": 100.0}
    monkeypatch.setattr(ca.time, "monotonic", lambda: clock["t"])

    bridge = _StreamBridge()
    # First delta emits immediately (last_emit was 0).
    asyncio.run(bridge.consume([Message(role=ASSISTANT, content="Want me to monitor or")]))
    msg = bridge.last_text_message()
    assert msg.content == "Want me to monitor or"

    # Second delta arrives within the batch interval → held back in _text_pending.
    clock["t"] = 100.01
    asyncio.run(bridge.consume([Message(role=ASSISTANT, content="Want me to monitor or adjust anything?")]))
    assert msg.content == "Want me to monitor or", "tail should still be pending (held back)"

    # End-of-turn sequence as _execute_agent_turn does it: flush pending FIRST,
    # then append the footer, then finalize.
    asyncio.run(bridge.flush_pending_text())
    assert msg.content == "Want me to monitor or adjust anything?", "flush must move the tail into content"

    footer = "\n\n---\n📊 _last turn: ..._"
    msg.content = (msg.content or "") + footer
    asyncio.run(bridge.finalize())

    # The model's sentence is intact and contiguous; the footer follows it.
    assert "Want me to monitor or adjust anything?" in msg.content
    assert msg.content.index("adjust anything?") < msg.content.index("📊")
