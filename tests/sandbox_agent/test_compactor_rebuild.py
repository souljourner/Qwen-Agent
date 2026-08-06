"""Tests for the compactor rebuild (2026-08-06).

Incident being fixed: a 281k-token chat compacted to 13.4k ("it knows nothing
about the past") and then re-compacted on the next turn. Root causes found:

  1. _messages_to_text truncated every message before the summarizer saw it
     (tool results 8k chars, user/assistant 4k, function_call args 500) while
     the chunker MEASURED them at full size — a ~50x accounting gap, so the
     summarizer received confetti.
  2. summarize_history only aborted when ALL chunks failed; a partial failure
     silently deleted the un-summarized remainder.
  3. Compaction targeted 170k while the model's own truncation backstop sat
     BELOW it at 160k, and that truncation deletes oldest-first — so a
     persisted digest (position 0) would be the first thing destroyed.
"""

import json

import pytest

from qwen_agent.llm.schema import ASSISTANT, FUNCTION, USER, FunctionCall, Message

import sandbox_agent.compaction.compactor as compactor


def _turn(i, tool_chars=30_000, args_chars=20_000):
    """One realistic turn: user ask, assistant tool call with big args, result."""
    return [
        Message(role=USER, content=f"user question {i} " + f"u{i} " * 200),
        Message(role=ASSISTANT, content=f"reasoning about {i} " + f"a{i} " * 200,
                function_call=FunctionCall(
                    name="project_write_file",
                    arguments=json.dumps({"path": f"n{i}.md", "content": f"C{i}" * (args_chars // 2)}))),
        Message(role=FUNCTION, name="project_write_file",
                content=f"tool output {i} " + f"t{i} " * (tool_chars // 4)),
    ]


class TestRenderPreservesMaterial:
    """The summarizer must receive the conversation, not a stub of it."""

    def test_render_retains_most_characters(self):
        msgs = [m for i in range(4) for m in _turn(i)]
        raw_chars = sum(len(str(m.content or "")) +
                        len(str(getattr(m.function_call, "arguments", "") or ""))
                        for m in msgs)
        rendered = compactor._messages_to_text(msgs)
        # Old behavior kept ~2%. Anything below 80% means we're summarizing
        # confetti and the digest cannot be faithful.
        assert len(rendered) >= 0.8 * raw_chars, (
            f"rendered {len(rendered):,} of {raw_chars:,} chars "
            f"({len(rendered)/raw_chars:.1%}) — summarizer is being starved")

    def test_function_call_arguments_survive_rendering(self):
        msgs = _turn(7)
        rendered = compactor._messages_to_text(msgs)
        # the file body lived only in function_call.arguments (was cut at 500)
        assert rendered.count("C7") > 1_000

    def test_char_budget_is_respected_when_given(self):
        msgs = [m for i in range(4) for m in _turn(i)]
        rendered = compactor._messages_to_text(msgs, char_budget=50_000)
        assert len(rendered) <= 52_000  # budget + truncation marker slack


class TestPartialFailureIsAllOrNothing:
    """A failed chunk must never mean 'delete that part of the history'."""

    def test_partial_chunk_failure_leaves_history_untouched(self, monkeypatch):
        msgs = [m for i in range(12) for m in _turn(i)]
        calls = {"n": 0}

        def flaky(text):
            calls["n"] += 1
            return "" if calls["n"] % 2 == 0 else "## Decisions Made\nsomething"

        import sandbox_agent.compaction.summarizer as summarizer
        monkeypatch.setattr(summarizer, "summarize_chunk", flaky)
        monkeypatch.setattr(summarizer, "merge_summaries", lambda s, i: "merged")
        # force multiple chunks
        monkeypatch.setattr(compactor, "build_chunks",
                            lambda history: [history[i:i + 3] for i in range(0, len(history), 3)])
        out = compactor.summarize_history(msgs)
        assert out is msgs, "a partial summarization must abort, not half-delete"

    def test_all_chunks_succeed_produces_digest(self, monkeypatch):
        msgs = [m for i in range(6) for m in _turn(i)]
        import sandbox_agent.compaction.summarizer as summarizer
        monkeypatch.setattr(summarizer, "summarize_chunk", lambda t: "## Decisions Made\nok")
        monkeypatch.setattr(summarizer, "merge_summaries", lambda s, i: "merged digest")
        out = compactor.summarize_history(msgs)
        assert any("[Context compacted" in str(m.content) for m in out)


class TestSummarizerRequestShape:

    def test_sends_max_tokens_but_never_temperature(self, monkeypatch):
        import sandbox_agent.compaction.summarizer as summarizer
        captured = {}

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return {"choices": [{"message": {"content": "ok"}}]}

        def fake_post(url, json=None, timeout=None, **kw):
            captured.update(json or {})
            return _Resp()

        monkeypatch.setattr(summarizer.requests, "post", fake_post)
        summarizer.summarize_chunk("some conversation text")
        # bounded output so a rambling digest can't blow the budget...
        assert captured.get("max_tokens", 0) > 0
        # ...but temperature stays a HOST default (standing instruction).
        assert "temperature" not in captured


class TestBudgetOrdering:
    """target < trigger < hard <= max_input_tokens — so the model's own
    truncation backstop never fires on healthy compacted output, and can
    never delete the digest (which sits at position 0)."""

    def test_ordering_invariant(self):
        from sandbox_agent.compaction.budget import derive_budgets
        b = derive_budgets(200_000, 160_000)
        assert b.target < b.trigger <= b.hard <= 160_000

    def test_hard_stays_under_model_input_cap(self):
        from sandbox_agent.compaction.budget import derive_budgets
        from sandbox_agent.config import BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG
        for cfg in (PRIMARY_LLM_CFG, BACKGROUND_LLM_CFG):
            b = derive_budgets(cfg["context_window_tokens"],
                               cfg["generate_cfg"]["max_input_tokens"])
            assert b.hard <= cfg["generate_cfg"]["max_input_tokens"]

    def test_compacted_output_survives_qwen_agent_truncation(self):
        """The F2 guard: a history at TARGET must pass through qwen_agent's
        real truncation with the digest intact. Before this ordering fix a
        170k-target history hit a 160k backstop that deletes oldest-first —
        and the digest is oldest."""
        from qwen_agent.llm.base import _truncate_input_messages_roughly
        from sandbox_agent.compaction.budget import derive_budgets
        from sandbox_agent.config import PRIMARY_LLM_CFG
        from sandbox_agent.token_budget import estimate_messages_tokens

        b = derive_budgets(PRIMARY_LLM_CFG["context_window_tokens"],
                           PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"])
        digest = Message(role=USER, content="[Context compacted: 400 messages]\n\nDURABLE facts")
        msgs = [digest]
        while estimate_messages_tokens(msgs) < b.target:
            msgs.append(Message(role=USER, content="q " + "x" * 8_000))
            msgs.append(Message(role=ASSISTANT, content="a " + "y" * 8_000))

        out = _truncate_input_messages_roughly(
            msgs, max_tokens=PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"])
        assert any("[Context compacted" in str(m.content) for m in out), \
            "digest was deleted by the model's own truncation"
        assert len(out) == len(msgs), "truncation dropped messages at TARGET"


class TestAggregateToolBudget:

    def test_many_results_share_one_budget(self):
        msgs = []
        for i in range(60):
            msgs.append(Message(role=ASSISTANT, content="",
                                function_call=FunctionCall(name="t", arguments="{}")))
            msgs.append(Message(role=FUNCTION, name="t", content=f"r{i} " + "z" * 40_000))
        out = compactor.truncate_tool_results(msgs, total_budget_chars=200_000)
        total = sum(len(str(m.content)) for m in out if m.role == FUNCTION)
        # old per-result cap let 60x40k = 2.4M chars all pass "within cap"
        assert total < 600_000, f"aggregate budget not enforced: {total:,} chars"

    def test_newest_results_keep_most(self):
        msgs = []
        for i in range(30):
            msgs.append(Message(role=FUNCTION, name="t", content=f"r{i} " + "z" * 30_000))
        out = [m for m in compactor.truncate_tool_results(msgs, total_budget_chars=150_000)
               if m.role == FUNCTION]
        assert len(str(out[-1].content)) > len(str(out[0].content))

    def test_truncation_is_idempotent(self):
        msgs = [Message(role=FUNCTION, name="t", content="z" * 300_000)]
        once = compactor.truncate_tool_results(msgs, total_budget_chars=50_000)
        twice = compactor.truncate_tool_results(once, total_budget_chars=50_000)
        assert str(once[0].content) == str(twice[0].content)


class TestPointers:
    """L0: replace recoverable bulk with pointers — allow-list only."""

    def _write(self, path="notes.md", mode=None, n=20_000, name="project_write_file"):
        args = {"project": "p", "path": path, "content": "X" * n}
        if mode:
            args["mode"] = mode
        return Message(role=ASSISTANT, content="",
                       function_call=FunctionCall(name=name, arguments=json.dumps(args)))

    def test_large_write_becomes_pointer_with_valid_json(self):
        from sandbox_agent.compaction.pointers import pointerize
        msgs = [self._write("a.md"), self._write("b.md")]  # b is newest for b.md
        out, saved = pointerize(msgs)
        args = json.loads(out[0].function_call.arguments)
        assert saved > 10_000
        assert "elided" in args["content"] and "sha256:" in args["content"]
        assert set(args) == {"project", "path", "content"}  # shape preserved

    def test_tail_is_the_only_inline_protection(self):
        # Uniform rule: everything eligible outside the protected tail is
        # pointer-ized. The tail is where "just wrote it, may edit next"
        # lives, so no extra per-path carve-out is needed.
        from sandbox_agent.compaction.pointers import pointerize
        msgs = [self._write("same.md"), self._write("same.md")]
        out, _ = pointerize(msgs, protect_indices={1})
        assert "elided" in json.loads(out[0].function_call.arguments)["content"]
        assert "X" * 100 in json.loads(out[1].function_call.arguments)["content"]

    def test_edit_mode_never_pointerized(self):
        from sandbox_agent.compaction.pointers import pointerize
        msgs = [self._write("a.md", mode="edit"), self._write("z.md")]
        out, _ = pointerize(msgs)
        assert "X" * 100 in json.loads(out[0].function_call.arguments)["content"]

    def test_forbidden_tools_untouched(self):
        from sandbox_agent.compaction.pointers import pointerize
        for tool in ("project_apply_patch", "code_interpreter", "exec", "send_email"):
            msgs = [self._write("a.md", name=tool), self._write("z.md")]
            out, _ = pointerize(msgs)
            assert "X" * 100 in out[0].function_call.arguments, tool

    def test_protected_indices_untouched(self):
        from sandbox_agent.compaction.pointers import pointerize
        msgs = [self._write("a.md"), self._write("z.md")]
        out, _ = pointerize(msgs, protect_indices={0})
        assert "X" * 100 in out[0].function_call.arguments

    def test_small_writes_and_malformed_args_skipped(self):
        from sandbox_agent.compaction.pointers import pointerize
        small = self._write("s.md", n=50)
        bad = Message(role=ASSISTANT, content="",
                      function_call=FunctionCall(name="project_write_file", arguments="{not json"))
        out, saved = pointerize([small, bad])
        assert saved == 0, "neither a small write nor malformed args may be touched"
        assert out[1].function_call.arguments == "{not json"

    def test_ships_disabled_by_default(self):
        from sandbox_agent.config import COMPACTION_POINTERS_ENABLED
        assert COMPACTION_POINTERS_ENABLED is False


def _stub_summarizer(monkeypatch, text="## Decisions Made\nchose SQLite"):
    import sandbox_agent.compaction.summarizer as s
    monkeypatch.setattr(s, "summarize_chunk", lambda t: text)
    monkeypatch.setattr(s, "merge_summaries", lambda su, i: text)


class TestBudgetFilledTail:
    """The 281k->13.4k incident: a fixed 2-turn tail discarded ~12x more
    than the budget required."""

    def test_tail_fills_budget_not_two_turns(self):
        msgs = [m for i in range(60) for m in _turn(i, tool_chars=800, args_chars=200)]
        seg = compactor.segment_messages(msgs, target_tokens=60_000)
        kept_turns = sum(1 for m in seg.recent if m.role == USER)
        assert kept_turns > 2, f"budget-filled tail kept only {kept_turns} turns"

    def test_newest_turn_always_kept_even_if_oversized(self):
        msgs = _turn(0, tool_chars=4_000) + _turn(1, tool_chars=900_000)
        seg = compactor.segment_messages(msgs, target_tokens=5_000)
        assert seg.recent, "the newest turn must survive even when oversized"
        assert seg.recent[-1].content.startswith("tool output 1")

    def test_tail_starts_with_user_message(self):
        msgs = [Message(role=ASSISTANT, content="orphan"),
                Message(role=FUNCTION, name="t", content="r")] + \
               [m for i in range(6) for m in _turn(i, tool_chars=500, args_chars=100)]
        seg = compactor.segment_messages(msgs, target_tokens=30_000)
        if seg.recent:
            assert seg.recent[0].role == USER


class TestDigestAccretion:

    def test_durable_survives_a_hostile_summarizer(self):
        from sandbox_agent.compaction.digest import make_digest, merge_into, parse_sections
        d1 = make_digest(["- USER CORRECTION: never use tabs"], "doing things", 10)
        d2 = merge_into(d1, [], "TOTALLY DIFFERENT", 5)
        d3 = merge_into(d2, [], "DIFFERENT AGAIN", 5)
        durable, _ = parse_sections(str(d3.content))
        assert any("never use tabs" in x for x in durable)

    def test_digest_is_user_role_and_marked(self):
        from sandbox_agent.compaction.digest import is_digest, make_digest
        d = make_digest(["- a decision"], "state", 3)
        assert d.role == "user" and is_digest(d)

    def test_legacy_unmarked_digest_still_detected(self):
        from sandbox_agent.compaction.digest import is_digest
        legacy = Message(role=USER, content="[Context compacted: 12 earlier messages...]\n\nstuff")
        assert is_digest(legacy), "existing threads' digests must be recognized"

    def test_digest_never_re_summarized(self):
        from sandbox_agent.compaction.digest import make_digest
        d = make_digest(["- keep me"], "w", 5)
        msgs = [d] + [m for i in range(6) for m in _turn(i, tool_chars=500, args_chars=100)]
        seg = compactor.segment_messages(msgs, target_tokens=20_000)
        assert seg.digest is d
        assert all(m is not d for m in seg.history)

    def test_digests_merge_not_stack(self, monkeypatch):
        from sandbox_agent.compaction.digest import is_digest, make_digest
        _stub_summarizer(monkeypatch)
        d = make_digest(["- prior fact"], "prior", 5)
        msgs = [d] + [m for i in range(10) for m in _turn(i, tool_chars=2_000, args_chars=500)]
        out = compactor.summarize_history(msgs, target_tokens=8_000)
        assert sum(1 for m in out if is_digest(m)) == 1
        assert "prior fact" in str(out[0].content)


class TestLadder:

    def test_stops_early_without_calling_the_llm(self, monkeypatch):
        from sandbox_agent.compaction import policy
        import sandbox_agent.compaction.summarizer as s
        called = {"n": 0}
        monkeypatch.setattr(s, "summarize_chunk",
                            lambda t: called.__setitem__("n", called["n"] + 1) or "x")
        # Sized so the aggregate tool-result budget CAN reach target on its
        # own (the newest few results are exempt from the shared budget, so
        # they set a floor the target must clear).
        msgs = []
        for i in range(20):
            msgs.append(Message(role=USER, content=f"q{i}"))
            msgs.append(Message(role=FUNCTION, name="t", content="z" * 10_000))
        res = policy.compact_for_persistence(msgs, target_tokens=45_000)
        assert res.changed, "ladder should have compacted"
        assert called["n"] == 0, f"LLM used despite cheap tiers sufficing: {res.levels}"
        assert res.levels == ["tool_results"]

    def test_no_llm_mode_never_summarizes(self, monkeypatch):
        from sandbox_agent.compaction import policy
        import sandbox_agent.compaction.summarizer as s
        monkeypatch.setattr(s, "summarize_chunk",
                            lambda t: pytest.fail("LLM used in allow_llm=False mode"))
        msgs = [m for i in range(30) for m in _turn(i)]
        policy.compact_for_persistence(msgs, target_tokens=5_000, allow_llm=False)

    def test_under_target_is_a_noop(self):
        from sandbox_agent.compaction import policy
        msgs = [Message(role=USER, content="tiny")]
        res = policy.compact_for_persistence(msgs, target_tokens=100_000)
        assert not res.changed and res.messages == msgs

    def test_input_never_mutated(self, monkeypatch):
        from sandbox_agent.compaction import policy
        _stub_summarizer(monkeypatch)
        msgs = [m for i in range(20) for m in _turn(i)]
        snapshot = [str(m.content) for m in msgs]
        policy.compact_for_persistence(msgs, target_tokens=10_000)
        assert [str(m.content) for m in msgs] == snapshot


class TestArchive:

    def test_roundtrip_and_delta_only(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_history as ch
        from sandbox_agent.compaction import archive
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        a = [Message(role=USER, content="first"), Message(role=ASSISTANT, content="second")]
        b = [Message(role=USER, content="third")]
        assert archive.archive_append("t1", a)
        assert archive.archive_append("t1", b)
        loaded = archive.load_archive("t1")
        assert [str(m.content) for m in loaded] == ["first", "second", "third"]

    def test_empty_delta_is_success(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_history as ch
        from sandbox_agent.compaction import archive
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        assert archive.archive_append("t2", []) is True


class TestPersistedCompaction:
    """The wiring: compaction must COMMIT into the stored history, exactly
    once per overflow, and never destroy anything that isn't archived."""

    def _history(self, n=40):
        return [m for i in range(n) for m in _turn(i, tool_chars=20_000, args_chars=8_000)]

    def test_compacts_in_place_and_archives_delta(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_app as ca
        import sandbox_agent.chat_history as ch
        from sandbox_agent.compaction import archive
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        _stub_summarizer(monkeypatch)

        history = self._history()
        before = len(history)
        ca._persist_compaction("thr", history)
        assert len(history) < before, "compaction did not commit into the list"
        assert archive.load_archive("thr"), "destroyed messages were not archived"

    def test_second_call_is_a_noop(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_app as ca
        import sandbox_agent.chat_history as ch
        import sandbox_agent.compaction.summarizer as s
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        _stub_summarizer(monkeypatch)

        history = self._history()
        ca._persist_compaction("thr", history)
        after_first = [str(m.content) for m in history]

        calls = {"n": 0}
        monkeypatch.setattr(s, "summarize_chunk",
                            lambda t: calls.__setitem__("n", calls["n"] + 1) or "x")
        ca._persist_compaction("thr", history)
        assert calls["n"] == 0, "re-compacted an already-compacted history"
        assert [str(m.content) for m in history] == after_first

    def test_nothing_destroyed_when_archive_fails(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_app as ca
        import sandbox_agent.chat_history as ch
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        _stub_summarizer(monkeypatch)
        monkeypatch.setattr("sandbox_agent.compaction.archive.archive_append",
                            lambda *a, **k: False)
        history = self._history()
        snapshot = [str(m.content) for m in history]
        ca._persist_compaction("thr", history)
        assert [str(m.content) for m in history] == snapshot

    def test_notifier_append_during_compaction_survives(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_app as ca
        import sandbox_agent.chat_history as ch
        monkeypatch.setattr(ch, "DATA_DIR", str(tmp_path))
        history = self._history()

        # simulate the notifier appending mid-summarization
        def racing_compact(snapshot, *, target_tokens, allow_llm=True):
            from sandbox_agent.compaction.policy import CompactionResult
            history.append(Message(role=USER, content="LATE ARRIVAL",
                                   extra={"synthetic": "notifier"}))
            return CompactionResult(messages=snapshot[:5], changed=True,
                                    archived=snapshot[5:], before_tokens=9, after_tokens=1)

        monkeypatch.setattr("sandbox_agent.compaction.policy.compact_for_persistence",
                            racing_compact)
        ca._persist_compaction("thr", history)
        assert any(str(m.content) == "LATE ARRIVAL" for m in history), \
            "a message appended during compaction was lost"

    def test_under_trigger_does_nothing(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_app as ca
        import sandbox_agent.compaction.summarizer as s
        monkeypatch.setattr(s, "summarize_chunk",
                            lambda t: pytest.fail("compacted a small history"))
        history = [Message(role=USER, content="hi"), Message(role=ASSISTANT, content="hello")]
        ca._persist_compaction("thr", history)
        assert len(history) == 2

    def test_budgets_use_the_tighter_tier(self):
        import sandbox_agent.chat_app as ca
        from sandbox_agent.config import BACKGROUND_LLM_CFG, PRIMARY_LLM_CFG
        b = ca._compaction_budgets()
        cap = min(PRIMARY_LLM_CFG["generate_cfg"]["max_input_tokens"],
                  BACKGROUND_LLM_CFG["generate_cfg"]["max_input_tokens"])
        assert b.target < b.trigger <= b.hard <= cap


class TestSummarizerResilience:
    """2026-08-06 live regression: full-fidelity chunks made each summarizer
    request take minutes, but the timeout was still 120s — so every chunk
    timed out, all-or-nothing refused to compact, and the whole thing
    repeated every turn ('it seems to continuously compact')."""

    def test_timeout_fits_a_full_fidelity_chunk(self):
        from sandbox_agent.config import COMPACTION_TIMEOUT
        assert COMPACTION_TIMEOUT >= 300, "too short for a full-content chunk"

    def test_retries_once_before_giving_up(self, monkeypatch):
        import sandbox_agent.compaction.summarizer as s
        calls = {"n": 0}

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return {"choices": [{"message": {"content": "recovered"}}]}

        def flaky(url, json=None, timeout=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise s.requests.exceptions.ReadTimeout("cold model load")
            return _Resp()

        monkeypatch.setattr(s.requests, "post", flaky)
        assert s.summarize_chunk("text") == "recovered"
        assert calls["n"] == 2

    def test_still_returns_empty_after_all_attempts(self, monkeypatch):
        import sandbox_agent.compaction.summarizer as s

        def always_fail(url, json=None, timeout=None, **kw):
            raise s.requests.exceptions.ReadTimeout("down")

        monkeypatch.setattr(s.requests, "post", always_fail)
        assert s.summarize_chunk("text") == ""  # never raises; caller aborts safely
