"""Tests for sandbox_agent.activity_log — focused on the **extra-kwargs passthrough
that lets oai.py's _capture_usage record per-call llm_usage events."""

import json

from sandbox_agent import activity_log


def setup_function():
    # Clean slate for each test.
    activity_log._recent_events.clear()


def test_named_kwargs_still_map_to_known_fields():
    activity_log.log_event(
        "tool_call",
        detail="ran something",
        tool_name="web_search",
        tool_args='{"q":"x"}',
        model="qwen3.6-27b-linux",
    )
    evt = activity_log.get_recent_events(1)[0]
    assert evt["type"] == "tool_call"
    assert evt["detail"] == "ran something"
    assert evt["tool"] == "web_search"
    assert evt["args"] == '{"q":"x"}'
    assert evt["model"] == "qwen3.6-27b-linux"


def test_extra_kwargs_are_merged_into_event():
    """The reason for the change: oai.py wants to log_event('llm_usage',
    prompt_tokens=…, completion_tokens=…, total_tokens=…, model=…) — those
    fields aren't in the named signature, so before the fix the call
    TypeError'd and got swallowed by oai.py's try/except."""
    activity_log.log_event(
        "llm_usage",
        model="qwen3.6-27b-linux",
        prompt_tokens=1234,
        completion_tokens=567,
        total_tokens=1801,
    )
    evt = activity_log.get_recent_events(1)[0]
    assert evt["type"] == "llm_usage"
    assert evt["model"] == "qwen3.6-27b-linux"
    assert evt["prompt_tokens"] == 1234
    assert evt["completion_tokens"] == 567
    assert evt["total_tokens"] == 1801


def test_extra_kwargs_with_none_are_dropped():
    activity_log.log_event("xyz", prompt_tokens=10, completion_tokens=None, extra_field="hi")
    evt = activity_log.get_recent_events(1)[0]
    assert evt.get("prompt_tokens") == 10
    assert "completion_tokens" not in evt   # None gets dropped
    assert evt.get("extra_field") == "hi"


def test_model_can_arrive_via_extras():
    """If the caller didn't use the named `model=` arg, an extras `model` key
    still lands on the event (so older callers that route through `**extra`
    keep working)."""
    activity_log.log_event("y", **{"model": "from_extras", "prompt_tokens": 7})
    evt = activity_log.get_recent_events(1)[0]
    assert evt["model"] == "from_extras"
    assert evt["prompt_tokens"] == 7


def test_event_persists_to_file():
    import os
    from sandbox_agent.activity_log import _activity_path
    path = _activity_path()
    before = os.path.getsize(path) if os.path.exists(path) else 0
    activity_log.log_event("llm_usage", prompt_tokens=99, completion_tokens=1)
    after = os.path.getsize(path)
    assert after > before
    # last line should be our event
    with open(path, "rb") as f:
        f.seek(max(0, after - 500))
        tail = f.read().decode("utf-8", errors="replace")
    last_line = tail.strip().splitlines()[-1]
    evt = json.loads(last_line)
    assert evt["type"] == "llm_usage"
    assert evt["prompt_tokens"] == 99
    assert evt["completion_tokens"] == 1
