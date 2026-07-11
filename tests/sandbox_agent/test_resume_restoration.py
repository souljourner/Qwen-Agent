"""Weakness #2A: resume must restore the agent's gathered context.

Sidecar: the agent-side history (incl. tool-call/result pairs) is persisted
per thread after every turn and loaded verbatim on resume. Fallback: threads
predating the sidecar are reconstructed from chat.db steps, replaying tool
pairs newest-first up to a char budget (~100k tokens)."""

# Neutralize chat_app's import-time bootstrap.
import sandbox_agent.main as _main  # noqa: E402
_main.bootstrap_background = lambda *a, **k: None

import pytest  # noqa: E402

from qwen_agent.llm.schema import ContentItem, FunctionCall, Message  # noqa: E402

import sandbox_agent.chat_history as ch  # noqa: E402
from sandbox_agent.chat_app import _rebuild_history_from_thread  # noqa: E402


# --- sidecar round-trip -------------------------------------------------------

@pytest.fixture
def hist_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
    return tmp_path


def test_sidecar_roundtrip_with_tools_and_multimodal(hist_env):
    history = [
        Message(role="user", content="[2026-07-11 09:00am PDT] check SOXS"),
        Message(role="assistant", content="", function_call=FunctionCall(
            name="web_search", arguments='{"query": "SOXS decay"}')),
        Message(role="function", name="web_search", content="[TOOL_OUTPUT] results..."),
        Message(role="assistant", content="SOXS decays ~7%/mo."),
        Message(role="user", content=[ContentItem(text="and this chart?"),
                                      ContentItem(image="data:image/png;base64,AAAA")]),
    ]
    ch.save_history("thread-1", history)
    loaded = ch.load_history("thread-1")
    assert loaded is not None and len(loaded) == 5
    assert loaded[1].function_call.name == "web_search"
    assert loaded[2].role == "function" and "TOOL_OUTPUT" in loaded[2].content
    assert isinstance(loaded[4].content, list)
    assert loaded[4].content[1].image.startswith("data:image/png")


def test_sidecar_missing_returns_none(hist_env):
    assert ch.load_history("no-such-thread") is None


def test_sidecar_corrupt_returns_none(hist_env, tmp_path):
    d = tmp_path / "chat_history"
    d.mkdir()
    (d / "bad.json").write_text("{not json")
    assert ch.load_history("bad") is None


def test_sidecar_save_is_atomic_overwrite(hist_env):
    ch.save_history("t", [Message(role="user", content="v1")])
    ch.save_history("t", [Message(role="user", content="v2")])
    assert ch.load_history("t")[0].content == "v2"


# --- reconstruction fallback ---------------------------------------------------

def _step(stype, output, name=None, input_=None, id_=None, metadata=None):
    return {"type": stype, "output": output, "name": name, "input": input_,
            "id": id_ or f"s-{id(output)}", "metadata": metadata or {}}


def _thread(steps):
    return {"id": "th", "steps": steps, "elements": []}


def test_rebuild_replays_tool_pairs_in_order():
    thread = _thread([
        _step("user_message", "find SOXS decay rate"),
        _step("tool", "search results here", name="web_search", input_='{"query": "SOXS"}'),
        _step("assistant_message", "It decays ~7%/mo."),
    ])
    hist = _rebuild_history_from_thread(thread)
    roles = [(m.role, getattr(m, "name", None)) for m in hist]
    assert roles == [("user", None), ("assistant", None), ("function", "web_search"),
                     ("assistant", None)]
    assert hist[1].function_call.name == "web_search"
    assert hist[1].function_call.arguments == '{"query": "SOXS"}'
    assert hist[2].content == "search results here"


def test_rebuild_budget_drops_oldest_tools_first():
    big = "x" * 600
    thread = _thread([
        _step("user_message", "q"),
        _step("tool", big, name="old_tool", input_="{}"),
        _step("tool", big, name="new_tool", input_="{}"),
        _step("assistant_message", "done"),
    ])
    import sandbox_agent.chat_app as ca
    hist = _rebuild_history_from_thread(thread, tool_budget_chars=800)
    names = [m.name for m in hist if m.role == "function"]
    assert names == ["new_tool"]                       # newest kept, oldest dropped
    assert any(m.role == "assistant" and m.content == "done" for m in hist)


def test_rebuild_still_skips_document_and_tags_background(monkeypatch):
    thread = _thread([
        _step("assistant_message", "full doc body", name="document"),
        _step("assistant_message", "✅ Background task **x** finished.", name="background task"),
        _step("assistant_message", "normal answer"),
    ])
    hist = _rebuild_history_from_thread(thread)
    texts = [str(m.content) for m in hist]
    assert not any("full doc body" in t for t in texts)
    assert any(t.startswith("[system event]") for t in texts)
    assert any("normal answer" in t for t in texts)


def test_rebuild_skips_thought_and_run_steps():
    thread = _thread([
        _step("run", "on_message"),
        _step("thought", "internal reasoning"),
        _step("user_message", "hi"),
        _step("assistant_message", "hello"),
    ])
    hist = _rebuild_history_from_thread(thread)
    assert len(hist) == 2
    assert not any("internal reasoning" in str(m.content) for m in hist)
