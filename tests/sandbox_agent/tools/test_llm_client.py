"""Tests for sandbox_agent.tools.llm_client — standalone subprocess LLM helper.

These tests mock `llm_client._SESSION.post` so they never hit a real endpoint.
"""

from unittest.mock import MagicMock

import pytest

from sandbox_agent.tools import llm_client


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Default single-entry chain via legacy vars so most tests stay simple."""
    monkeypatch.delenv("LLM_CALL_CHAIN", raising=False)
    monkeypatch.delenv("VLLM_BASE", raising=False)
    monkeypatch.setenv("LLM_CALL_BASE", "http://test.local/v1")
    monkeypatch.setenv("LLM_CALL_MODEL", "test-model")


def _mock_response(text: str):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
    }
    return resp


class TestLlmCallShape:

    def test_llm_call_posts_to_endpoint_with_model_and_messages(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _mock_response("hello")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        out = llm_client.llm_call("say hi", system="be brief", temperature=0.0, timeout=30)
        assert out == "hello"
        assert captured["url"] == "http://test.local/v1/chat/completions"
        assert captured["json"]["model"] == "test-model"
        assert captured["json"]["temperature"] == 0.0
        assert captured["json"]["messages"][0] == {"role": "system", "content": "be brief"}
        assert captured["json"]["messages"][1] == {"role": "user", "content": "say hi"}
        assert captured["timeout"] == 30

    def test_llm_call_without_system(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["messages"] = json["messages"]
            return _mock_response("ok")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        llm_client.llm_call("just user")
        assert len(captured["messages"]) == 1
        assert captured["messages"][0] == {"role": "user", "content": "just user"}

    def test_llm_call_passes_max_tokens_when_set(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["body"] = json
            return _mock_response("ok")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        llm_client.llm_call("hi", max_tokens=512)
        assert captured["body"]["max_tokens"] == 512

    def test_missing_all_env_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_CALL_BASE", raising=False)
        monkeypatch.delenv("LLM_CALL_MODEL", raising=False)
        monkeypatch.delenv("LLM_CALL_CHAIN", raising=False)
        monkeypatch.delenv("VLLM_BASE", raising=False)
        with pytest.raises(RuntimeError, match="No LLM endpoint"):
            llm_client.llm_call("hi")


class TestChainResolution:

    def test_chain_from_json_env(self, monkeypatch):
        monkeypatch.delenv("LLM_CALL_BASE", raising=False)
        monkeypatch.delenv("LLM_CALL_MODEL", raising=False)
        monkeypatch.setenv(
            "LLM_CALL_CHAIN",
            '[{"model":"a","base":"http://a.local/v1"},{"model":"b","base":"http://b.local/v1"}]',
        )
        assert llm_client._resolve_chain() == [
            ("a", "http://a.local/v1"),
            ("b", "http://b.local/v1"),
        ]

    def test_chain_defaults_from_vllm_base(self, monkeypatch):
        monkeypatch.delenv("LLM_CALL_BASE", raising=False)
        monkeypatch.delenv("LLM_CALL_MODEL", raising=False)
        monkeypatch.delenv("LLM_CALL_CHAIN", raising=False)
        monkeypatch.setenv("VLLM_BASE", "http://vllm.local/v1")
        chain = llm_client._resolve_chain()
        assert chain == [("qwen3.6-27b-linux", "http://vllm.local/v1"), ("laguna-s-2.1", "http://vllm.local/v1")]

    def test_legacy_single_pair(self, monkeypatch):
        monkeypatch.delenv("LLM_CALL_CHAIN", raising=False)
        monkeypatch.delenv("VLLM_BASE", raising=False)
        monkeypatch.setenv("LLM_CALL_BASE", "http://legacy.local/v1")
        monkeypatch.setenv("LLM_CALL_MODEL", "legacy-model")
        assert llm_client._resolve_chain() == [("legacy-model", "http://legacy.local/v1")]

    def test_json_chain_takes_precedence_over_legacy(self, monkeypatch):
        monkeypatch.setenv(
            "LLM_CALL_CHAIN",
            '[{"model":"x","base":"http://x/v1"}]',
        )
        monkeypatch.setenv("LLM_CALL_BASE", "http://legacy.local/v1")
        monkeypatch.setenv("LLM_CALL_MODEL", "legacy")
        assert llm_client._resolve_chain() == [("x", "http://x/v1")]


class TestFallbackBehavior:

    def test_empty_content_retries_once_then_falls_through(self, monkeypatch):
        monkeypatch.setenv(
            "LLM_CALL_CHAIN",
            '[{"model":"primary","base":"http://p/v1"},{"model":"backup","base":"http://b/v1"}]',
        )
        calls = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            calls.append(json["model"])
            if json["model"] == "primary":
                return _mock_response("")  # both attempts empty
            return _mock_response("ok")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        assert llm_client.llm_call("hi") == "ok"
        # primary tried twice (retry on empty), then backup once
        assert calls == ["primary", "primary", "backup"]

    def test_http_error_skips_retry_goes_to_next_model(self, monkeypatch):
        monkeypatch.setenv(
            "LLM_CALL_CHAIN",
            '[{"model":"primary","base":"http://p/v1"},{"model":"backup","base":"http://b/v1"}]',
        )
        calls = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            calls.append(json["model"])
            if json["model"] == "primary":
                err = MagicMock()
                err.raise_for_status.side_effect = Exception("http 500")
                return err
            return _mock_response("ok")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        assert llm_client.llm_call("hi") == "ok"
        # primary tried once (HTTP error skips retry), then backup once
        assert calls == ["primary", "backup"]

    def test_all_empty_raises(self, monkeypatch):
        monkeypatch.setenv(
            "LLM_CALL_CHAIN",
            '[{"model":"a","base":"http://a/v1"},{"model":"b","base":"http://b/v1"}]',
        )

        def fake_post(url, json=None, timeout=None, **kwargs):
            return _mock_response("")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)
        with pytest.raises(RuntimeError, match="All LLM endpoints failed"):
            llm_client.llm_call("hi")


class TestLlmBatchSharedSystem:

    def test_batch_issues_one_call_per_prompt(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            calls.append(json)
            return _mock_response(f"resp:{json['messages'][-1]['content']}")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        results = llm_client.llm_batch(
            system="STATIC SYSTEM",
            prompts=["item 1", "item 2", "item 3"],
            max_concurrent=2,
        )
        assert len(results) == 3
        assert results == ["resp:item 1", "resp:item 2", "resp:item 3"]
        assert len(calls) == 3

    def test_batch_system_prompt_identical_across_calls(self, monkeypatch):
        """KV-cache contract: every call's messages[0] (system) must be byte-
        identical, else the backend won't cache the prefix."""
        seen_systems = []

        def fake_post(url, json=None, timeout=None, **kwargs):
            seen_systems.append(json["messages"][0])
            return _mock_response("ok")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        system = "You classify articles. Output JSON only."
        llm_client.llm_batch(system=system, prompts=["a", "b", "c", "d"])

        assert len(seen_systems) == 4
        for sys_msg in seen_systems:
            assert sys_msg == {"role": "system", "content": system}

    def test_batch_preserves_prompt_order(self, monkeypatch):
        def fake_post(url, json=None, timeout=None, **kwargs):
            # Return the prompt verbatim so we can check ordering
            return _mock_response(json["messages"][-1]["content"])

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)

        prompts = [f"p{i}" for i in range(10)]
        results = llm_client.llm_batch(system="S", prompts=prompts, max_concurrent=4)
        assert results == prompts


class TestModuleShape:

    def test_exports(self):
        assert callable(llm_client.llm_call)
        assert callable(llm_client.llm_batch)

    def test_session_is_requests_session(self):
        import requests
        assert isinstance(llm_client._SESSION, requests.Session)


class TestNoDefaultTemperature:
    """Host defaults rule: nothing sends temperature unless a caller
    explicitly passes one (each backend knows its model's recommended
    sampling; we were blindly sending qwen's 0.6 to laguna)."""

    def test_llm_call_omits_temperature_by_default(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["json"] = json
            return _mock_response("hi")

        monkeypatch.setattr(llm_client._SESSION, "post", fake_post)
        llm_client.llm_call("say hi", timeout=30)
        assert "temperature" not in captured["json"]

    def test_agent_cfgs_have_no_temperature(self):
        from sandbox_agent.config import BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG
        assert "temperature" not in PRIMARY_LLM_CFG["generate_cfg"]
        assert "temperature" not in BACKGROUND_LLM_CFG["generate_cfg"]

    def test_bridge_body_has_no_temperature(self):
        import inspect
        from sandbox_agent.tools import llm_bridge
        assert '"temperature"' not in inspect.getsource(llm_bridge)
