"""Workflow-level tests: MULTIPLE consecutive turns, mocking only HTTP.

Why this file exists
--------------------
The 2026-08-06 timeout incident shipped past ~800 passing tests. Both of
its symptoms were structurally invisible to every one of them:

  1. Every existing test stubs `summarize_chunk` (or `_call_ollama`) to
     return instantly. `COMPACTION_TIMEOUT` was therefore never on any
     test's execution path — the value could have been 1s or 1 day and the
     suite would not have noticed.
  2. Not one existing test ran more than a SINGLE turn. The actual user
     complaint — "it seems to continuously compact" — is by definition a
     statement about turn N+1, so no single-turn test could express it.

So these tests mock at the `requests.post` boundary (the real timeout is
exercised, the real retry loop runs) and drive several consecutive turns
through `compact_for_persistence`, which is what a chat turn actually
calls. They assert on things the user can feel: does the turn finish
promptly, does the history survive, and does work repeat.
"""

import json

import pytest
import requests

from qwen_agent.llm.schema import ASSISTANT, FUNCTION, USER, FunctionCall, Message

from sandbox_agent.compaction import breaker
from sandbox_agent.compaction.policy import compact_for_persistence
from sandbox_agent.token_budget import estimate_messages_tokens


@pytest.fixture(autouse=True)
def _clean_breaker():
    breaker.reset()
    yield
    breaker.reset()


def _history(turns=60):
    out = [Message(role=USER, content="system-ish opener")]
    for i in range(turns):
        out += [
            Message(role=USER, content=f"question {i} " + f"q{i} " * 300),
            Message(role=ASSISTANT, content=f"answer {i} " + f"a{i} " * 300,
                    function_call=FunctionCall(
                        name="project_write_file",
                        arguments=json.dumps({"path": f"f{i}.md", "content": "X" * 8000}))),
            Message(role=FUNCTION, name="project_write_file",
                    content=f"wrote f{i}.md " + f"t{i} " * 2000),
        ]
    return out


class _Transport:
    """Records every HTTP call the summarizer makes."""

    def __init__(self, behavior):
        self.behavior = behavior   # callable(call_index) -> str, or raises
        self.calls = []

    def post(self, url, json=None, timeout=None, **kw):
        self.calls.append({"timeout": timeout, "chars": len(str(json))})
        text = self.behavior(len(self.calls) - 1)

        class _R:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"choices": [{"message": {"content": text}}]}

        return _R()

    def install(self, monkeypatch):
        monkeypatch.setattr(requests, "post", self.post)
        return self


def _always_times_out(_i):
    raise requests.exceptions.ReadTimeout("read timeout")


def _always_works(i):
    return f"## Decisions Made\n- summary of segment {i}\n\n## Open TODOs\n- none"


TARGET = 40_000


class TestTheReportedSymptom:
    """'it seems to continuously compact' — a statement about turn N+1."""

    def test_dead_summarizer_does_not_repeat_full_cost_every_turn(self, monkeypatch):
        t = _Transport(_always_times_out).install(monkeypatch)
        history = _history()

        for _ in range(5):
            result = compact_for_persistence(history, target_tokens=TARGET)
            history = result.messages

        # Without a breaker this is 5 turns x chunks x attempts of full-timeout
        # stalls. The user experiences that as a chat that never responds.
        assert len(t.calls) <= 2, (
            f"summarizer called {len(t.calls)} times across 5 failing turns — "
            "the expensive call is repeating every turn")

    def test_history_survives_a_dead_summarizer_intact(self, monkeypatch):
        _Transport(_always_times_out).install(monkeypatch)
        history = _history()
        original_user_text = [m.content for m in history if m.role == USER]

        for _ in range(3):
            history = compact_for_persistence(history, target_tokens=TARGET).messages

        # Degraded (over target) is acceptable. Half-destroyed is not.
        surviving = [m.content for m in history if m.role == USER]
        assert surviving == original_user_text, "user turns lost while summarizer was down"

    def test_a_successful_compaction_is_not_redone_next_turn(self, monkeypatch):
        t = _Transport(_always_works).install(monkeypatch)
        history = _history()

        first = compact_for_persistence(history, target_tokens=TARGET)
        assert first.changed
        calls_after_first = len(t.calls)
        assert calls_after_first > 0, "first turn should have summarized"

        second = compact_for_persistence(first.messages, target_tokens=TARGET)
        assert len(t.calls) == calls_after_first, (
            "second turn re-summarized an already-compacted history")
        assert not second.changed


class TestTheTimeoutIsRealAndReachesTheWire:
    """The bug was a config value that no test could observe."""

    def test_configured_timeout_is_passed_to_the_request(self, monkeypatch):
        from sandbox_agent.config import COMPACTION_TIMEOUT
        t = _Transport(_always_works).install(monkeypatch)
        compact_for_persistence(_history(), target_tokens=TARGET)
        assert t.calls, "no request was made"
        assert all(c["timeout"] == COMPACTION_TIMEOUT for c in t.calls)

    def test_timeout_is_plausible_for_a_full_fidelity_chunk(self, monkeypatch):
        """Chunks now carry full content, so requests take minutes. A timeout
        sized for the old truncating renderer silently fails everything."""
        from sandbox_agent.config import COMPACTION_TIMEOUT
        t = _Transport(_always_works).install(monkeypatch)
        compact_for_persistence(_history(), target_tokens=TARGET)
        biggest = max(c["chars"] for c in t.calls)
        assert biggest > 50_000, "chunk looks truncated — the confetti bug is back"
        # ~1k chars/s is a conservative floor for prefill+generation here.
        assert COMPACTION_TIMEOUT >= biggest / 1000, (
            f"timeout {COMPACTION_TIMEOUT}s cannot deliver a {biggest:,}-char chunk")


class TestTransientVersusSustainedFailure:
    def test_one_bad_call_still_compacts(self, monkeypatch):
        def flaky(i):
            if i == 0:
                raise requests.exceptions.ConnectionError("model still spawning")
            return _always_works(i)

        _Transport(flaky).install(monkeypatch)
        result = compact_for_persistence(_history(), target_tokens=TARGET)
        assert result.changed, "a single transient failure aborted the whole compaction"
        assert result.error is None

    def test_breaker_reopens_after_cooldown(self, monkeypatch):
        import sandbox_agent.compaction.breaker as b
        clock = {"t": 1000.0}
        monkeypatch.setattr(b.time, "monotonic", lambda: clock["t"])

        t = _Transport(_always_times_out).install(monkeypatch)
        history = _history()
        compact_for_persistence(history, target_tokens=TARGET)
        assert b.is_open(), "breaker should trip after sustained failure"

        from sandbox_agent.config import COMPACTION_BREAKER_COOLDOWN
        clock["t"] += COMPACTION_BREAKER_COOLDOWN + 1
        assert not b.is_open(), "breaker never recovers"

        t.behavior = _always_works
        result = compact_for_persistence(history, target_tokens=TARGET)
        assert result.changed, "did not resume compacting after recovery"

    def test_success_clears_prior_failures(self):
        import sandbox_agent.compaction.breaker as b
        b.record_failure()
        assert b.status()["consecutive_failures"] == 1
        b.record_success()
        assert b.status() == {"open": False, "consecutive_failures": 0,
                              "cooldown_remaining_s": 0}

    def test_open_breaker_leaves_history_intact_rather_than_partial(self, monkeypatch):
        """Failing fast must not become a licence to half-compact."""
        import sandbox_agent.compaction.breaker as b
        b.record_failure()
        assert b.is_open()
        t = _Transport(_always_works).install(monkeypatch)
        history = _history()
        result = compact_for_persistence(history, target_tokens=TARGET)
        assert not t.calls, "called the summarizer with the breaker open"
        assert [m.content for m in result.messages if m.role == USER] == \
               [m.content for m in history if m.role == USER]


class TestSteadyStateOverManyTurns:
    def test_ten_turns_stay_under_budget_without_runaway_summarizing(self, monkeypatch):
        t = _Transport(_always_works).install(monkeypatch)
        history = _history(turns=30)

        for turn in range(10):
            history = compact_for_persistence(history, target_tokens=TARGET).messages
            history += [
                Message(role=USER, content=f"new ask {turn} " + f"n{turn} " * 400),
                Message(role=ASSISTANT, content=f"new answer {turn} " + f"r{turn} " * 400),
            ]
            assert estimate_messages_tokens(history) < TARGET * 3, (
                f"history diverged at turn {turn}")

        assert len(t.calls) < 25, (
            f"{len(t.calls)} summarizer calls across 10 turns — compaction is "
            "re-summarizing material it already folded into the digest")


class TestRetryPolicy:
    """A retry must not double a stall the user is already sitting through."""

    def test_timeout_is_not_retried(self, monkeypatch):
        t = _Transport(_always_times_out).install(monkeypatch)
        compact_for_persistence(_history(), target_tokens=TARGET)
        assert len(t.calls) == 1, (
            "retried after a timeout — that doubles the worst-case turn stall "
            "for a failure mode the retry cannot fix")

    def test_fast_failure_is_retried(self, monkeypatch):
        def cold_then_warm(i):
            if i == 0:
                raise requests.exceptions.ConnectionError("connection refused")
            return _always_works(i)

        t = _Transport(cold_then_warm).install(monkeypatch)
        result = compact_for_persistence(_history(), target_tokens=TARGET)
        assert len(t.calls) >= 2 and result.changed


class TestMergeFailureLosesNothing:
    """The chunk loop is all-or-nothing, but the MERGE step had the same
    'partial failure drops the remainder' bug: it fell back to summaries[0]
    while still deleting every message chunks 2..N covered."""

    def _merge_fails(self, i):
        # Chunk calls succeed; the merge (the call carrying "### Summary")
        # fails. Distinguished by payload below.
        raise AssertionError("unused")

    def test_all_chunk_summaries_survive_a_failed_merge(self, monkeypatch):
        seen = []

        def behavior(i):
            raise AssertionError("unused")

        class T(_Transport):
            def post(self, url, json=None, timeout=None, **kw):
                payload = str(json)
                seen.append(payload)
                is_merge = "Merge these conversation summaries" in payload
                if is_merge:
                    raise requests.exceptions.ConnectionError("merge endpoint down")

                class _R:
                    def raise_for_status(s): pass

                    def json(s):
                        n = len([p for p in seen if "Merge these" not in p])
                        return {"choices": [{"message": {
                            "content": f"## Decisions Made\n- UNIQUE_MARKER_{n}"}}]}

                return _R()

        t = T(behavior).install(monkeypatch)
        result = compact_for_persistence(_history(turns=90), target_tokens=TARGET)

        chunk_calls = len([p for p in seen if "Merge these" not in p])
        assert chunk_calls >= 2, "need multiple chunks to exercise the merge"

        digest = "\n".join(str(m.content) for m in result.messages)
        for n in range(1, chunk_calls + 1):
            assert f"UNIQUE_MARKER_{n}" in digest, (
                f"chunk {n}'s summary was dropped when the merge failed — "
                "its messages were deleted but its content did not survive")

    def test_failed_merge_does_not_open_the_breaker(self, monkeypatch):
        seen = []

        class T(_Transport):
            def post(self, url, json=None, timeout=None, **kw):
                payload = str(json)
                seen.append(payload)
                if "Merge these conversation summaries" in payload:
                    raise requests.exceptions.ConnectionError("merge down")

                class _R:
                    def raise_for_status(s): pass

                    def json(s):
                        return {"choices": [{"message": {"content": "## Decisions Made\n- ok"}}]}

                return _R()

        T(lambda i: "").install(monkeypatch)
        compact_for_persistence(_history(turns=90), target_tokens=TARGET)
        assert not breaker.is_open(), (
            "a failed merge opened the breaker, blocking the chunk "
            "summarization that actually matters for 900s")


class TestResume:
    """Resuming a large thread: the digest must survive the round-trip, and
    the turn must summarize ONCE, not once transiently + once persisted."""

    def _big(self, tokens=134_000):
        from sandbox_agent.compaction.digest import make_digest
        h = [Message(role="system", content="you are an agent")]
        h.append(make_digest(["- user said: always use HTML email"], "prior work", 174))
        i = 0
        while estimate_messages_tokens(h) < tokens:
            h += [Message(role=USER, content=f"q{i} " + "x " * 900),
                  Message(role=ASSISTANT, content=f"a{i} " + "y " * 900)]
            i += 1
        return h

    def test_digest_survives_the_sidecar_round_trip(self, tmp_path, monkeypatch):
        import sandbox_agent.chat_history as CH
        from sandbox_agent.compaction.digest import make_digest, is_digest, parse_sections
        monkeypatch.setattr(CH, "DATA_DIR", str(tmp_path))

        d = make_digest(["- user said: always use HTML email"], "state", 174)
        CH.save_history("t", [d, Message(role=USER, content="hi")])
        back = CH.load_history("t")

        assert back and is_digest(back[0])
        # marker, not just the content sentinel — counters must survive too
        assert back[0].extra["compaction"]["kind"] == "digest"
        assert back[0].extra["compaction"]["summarized"] == 174
        assert parse_sections(str(back[0].content))[0] == ["- user said: always use HTML email"]

    def test_trim_never_drops_the_digest(self):
        """trim_to_budget protected only role=="system"; the digest is
        role=="user" at position 0, so it was the FIRST thing dropped —
        the agent kept recent chatter and lost its entire summarized past."""
        from sandbox_agent.token_budget import trim_to_budget
        from sandbox_agent.compaction.digest import is_digest

        out = trim_to_budget(self._big(60_000), max_tokens=15_000)
        assert any(is_digest(m) for m in out), "digest dropped by trim"
        assert any(m.role == "system" for m in out)
        assert estimate_messages_tokens(out) <= 15_000

    def test_start_of_turn_does_not_summarize(self, monkeypatch):
        """Asserts the real CALL SITE in main.create_agent, not just that
        maybe_compact honours the flag — passing allow_llm=True there must
        fail this test."""
        import sandbox_agent.compaction as comp
        import sandbox_agent.main as m
        from sandbox_agent.config import PRIMARY_LLM_CFG

        seen = []                      # a list: the mid-run call follows the
        real = comp.maybe_compact      # entry call and would clobber a dict

        def spy(messages, **kw):
            before = len(t.calls)
            try:
                return real(messages, **kw)
            finally:
                seen.append({**kw, "http_calls": len(t.calls) - before})

        t = _Transport(_always_works).install(monkeypatch)
        monkeypatch.setattr(comp, "maybe_compact", spy)

        agent = m.create_agent("sys", llm_cfg=PRIMARY_LLM_CFG)
        gen = agent._run(self._big())
        try:
            next(gen)          # entry compaction runs before the first yield
        except Exception:
            pass               # the LLM call itself is not under test
        finally:
            gen.close()

        assert seen, "entry compaction never invoked maybe_compact"
        entry = seen[0]                # first call is the start-of-turn one
        assert entry.get("pinned_head", 0) == 0, "not the entry call"
        assert entry.get("allow_llm") is False, (
            f"start-of-turn call site passed allow_llm={entry.get('allow_llm')} — "
            "it summarizes a throwaway copy, duplicating end-of-turn work")
        # Scoped to the ENTRY call: mid-run compaction summarizing later in
        # the same turn is correct (test_midrun_may_still_summarize).
        assert entry["http_calls"] == 0, (
            f"start-of-turn compaction made {entry['http_calls']} summarizer "
            "calls on a copy that is discarded at the end of the turn")

    def test_midrun_may_still_summarize(self, monkeypatch):
        """Inside the tool loop there is no end-of-turn backstop."""
        from sandbox_agent.compaction import compact_midrun
        t = _Transport(_always_works).install(monkeypatch)
        compact_midrun(self._big(), context_tokens=200_000)
        assert t.calls, "mid-run compaction can no longer summarize"

    def test_resumed_turn_summarizes_once_not_twice(self, monkeypatch):
        """The whole point: entry + end-of-turn used to each pay ~2 minutes."""
        from sandbox_agent.compaction import maybe_compact
        from sandbox_agent.compaction.budget import derive_budgets
        t = _Transport(_always_works).install(monkeypatch)
        history = self._big()

        maybe_compact(history, context_tokens=200_000, allow_llm=False)
        entry_calls = len(t.calls)

        b = derive_budgets(200_000, 160_000)
        compact_for_persistence(history, target_tokens=b.target)
        persist_calls = len(t.calls) - entry_calls

        assert entry_calls == 0, f"entry path made {entry_calls} summarizer calls"
        assert persist_calls > 0, "end-of-turn path did not summarize"


class TestDigestPointsAtRetrieval:
    """Compaction shrinks what the agent CARRIES, not what it can REACH:
    chat.db keeps every step and session_search indexes it. But the model
    only knows that if the digest says so — otherwise it confabulates
    details it could simply have looked up."""

    def test_digest_tells_the_agent_to_search(self):
        from sandbox_agent.compaction.digest import make_digest
        body = str(make_digest(["- a decision"], "state", 174).content)
        assert "session_search" in body, "digest gives the model no way to recover detail"
        assert "not deleted" in body.lower() or "not delete" in body.lower()

    def test_guidance_survives_accretion(self):
        """merge_into re-renders the digest; the instruction must not be a
        one-time header that later generations drop."""
        from sandbox_agent.compaction.digest import make_digest, merge_into
        first = make_digest(["- one"], "state one", 100)
        second = merge_into(first, ["- two"], "state two", 50)
        assert "session_search" in str(second.content)
        assert "- one" in str(second.content) and "- two" in str(second.content)

    def test_guidance_is_not_mistaken_for_durable_content(self):
        from sandbox_agent.compaction.digest import make_digest, parse_sections
        d = make_digest(["- real durable item"], "state", 10)
        durable, _ = parse_sections(str(d.content))
        assert durable == ["- real durable item"], (
            f"retrieval guidance leaked into DURABLE: {durable}")
