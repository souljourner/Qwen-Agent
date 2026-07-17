"""Phase-2 tests: pilot-history quarantine, insane-metrics decision, stage-3
stamping + hash pin, stage-4 skew → RESET_STAGE:3, evaluator-written
pipeline/metrics.json."""

import json
import os
import shutil
import tempfile

import pytest

import sandbox_agent.pipeline.evaluator as ev
import sandbox_agent.pipeline.orchestrator as orch
from sandbox_agent.pipeline.metrics_sanity import canonical_hash


@pytest.fixture
def proj(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE", os.path.join(d, "pipeline.lock"))
    monkeypatch.setattr("sandbox_agent.pipeline.evaluator.DATA_DIR", d)
    pdir = os.path.join(d, "projects", "proj")
    os.makedirs(os.path.join(pdir, "strategy"))
    os.makedirs(os.path.join(pdir, "backtest", "full"))
    os.makedirs(os.path.join(pdir, "backtest", "pilot"))
    os.makedirs(os.path.join(pdir, "pipeline"))
    yield d, pdir
    shutil.rmtree(d)


def _write_loop_state(pdir, **kwargs):
    ls = {"oos_cutoff_date": "2025-06-01", "pilot_history": [], "hypothesis_count": 1}
    ls.update(kwargs)
    path = os.path.join(pdir, "strategy", "loop_state.json")
    with open(path, "w") as f:
        json.dump(ls, f)
    return path


GOOD_ROW = {"version": 1, "return_pct": 12.0, "dd_pct": -8.0, "trades": 40, "sharpe": 1.1}
INSANE_ROW = {"version": 2, "return_pct": 1_399_591.9, "dd_pct": -235.3, "trades": 7}


# --- quarantine ------------------------------------------------------------------

def test_sanitizer_quarantines_insane_rows(proj):
    _, pdir = proj
    path = _write_loop_state(pdir, pilot_history=[GOOD_ROW, INSANE_ROW])
    changed = ev.sanitize_loop_state_file(path)
    assert changed
    ls = json.load(open(path))
    assert [r["version"] for r in ls["pilot_history"]] == [1]
    assert len(ls["pilot_history_rejected"]) == 1
    assert ls["pilot_history_rejected"][0]["_violations"]


def test_sanitizer_keeps_legit_short_blowup_row(proj):
    _, pdir = proj
    short_row = {"version": 3, "return_pct": 80.0, "dd_pct": -150.0, "trades": 25}
    path = _write_loop_state(pdir, pilot_history=[short_row])
    ev.sanitize_loop_state_file(path)
    ls = json.load(open(path))
    assert len(ls["pilot_history"]) == 1          # NOT quarantined


# --- stage-3 stamping + pin --------------------------------------------------------

GOOD_FULL_METRICS = {
    "pilot_sharpe": 1.2, "oos_sharpe": 1.0, "walk_forward_win_rate": 0.75,
    "pilot_sortino": 1.6, "oos_sortino": 1.3,
    "pilot_annualized_return_pct": 14.0, "oos_annualized_return_pct": 11.0,
    "total_trades": 250, "t_stat_daily_returns": 3.1, "deflated_sharpe": 0.4,
    "turnover": 10.0, "declared_turnover": 9.0,
    "return_pct": 40.0, "dd_pct": -18.0,
}


def _init_state(project="proj"):
    state = orch.init_pipeline(project, "test", pipeline_type="trading")
    orch.save_state(state)
    return state


def test_stage3_pass_stamps_and_pins(proj):
    _, pdir = proj
    _init_state()
    full = os.path.join(pdir, "backtest", "full", "metrics.json")
    json.dump(GOOD_FULL_METRICS, open(full, "w"))

    passed, msg = ev._stage_specific_checks(pdir, "trading", 3)
    assert passed, msg
    on_disk = json.load(open(full))
    assert on_disk["schema_version"] == 1
    assert on_disk["content_hash"] == canonical_hash(on_disk)
    state = orch.load_state("proj")
    assert state.pinned_full_metrics_hash == on_disk["content_hash"]


def test_stage3_rejects_insane_full_metrics(proj):
    _, pdir = proj
    _init_state()
    bad = dict(GOOD_FULL_METRICS, dd_pct=-5000.0)
    json.dump(bad, open(os.path.join(pdir, "backtest", "full", "metrics.json"), "w"))
    passed, msg = ev._check_full_validation_metrics(pdir)
    assert not passed
    assert "sanity" in msg.lower()


# --- stage-4 pin gate ----------------------------------------------------------------

def _pass_stage3(pdir):
    full = os.path.join(pdir, "backtest", "full", "metrics.json")
    json.dump(GOOD_FULL_METRICS, open(full, "w"))
    passed, msg = ev._stage_specific_checks(pdir, "trading", 3)
    assert passed, msg
    return json.load(open(full))["content_hash"]


def _write_verdict(pdir, rec="promote", hash_line=None):
    text = f"## Final Recommendation\n{rec}\n\n## Rationale\nnumbers good\n"
    if hash_line:
        text += f"\nMetrics Hash: {hash_line}\n"
    open(os.path.join(pdir, "pipeline", "verdict.md"), "w").write(text)


def test_stage4_passes_with_matching_pin(proj):
    _, pdir = proj
    _init_state()
    pin = _pass_stage3(pdir)
    _write_verdict(pdir, "promote", hash_line=pin)
    passed, msg = ev._stage_specific_checks(pdir, "trading", 4)
    assert passed, msg
    # evaluator wrote the summary
    summary = json.load(open(os.path.join(pdir, "pipeline", "metrics.json")))
    assert summary["verdict"] == "promote"
    assert summary["gates_passed"] is True
    assert summary["written_by"] == "evaluator"


def test_stage4_rejects_missing_hash_line(proj):
    _, pdir = proj
    _init_state()
    _pass_stage3(pdir)
    _write_verdict(pdir, "promote", hash_line=None)
    passed, msg = ev._stage_specific_checks(pdir, "trading", 4)
    assert not passed
    assert "metrics hash" in msg.lower()


def test_stage4_skew_requests_stage3_reset(proj):
    _, pdir = proj
    _init_state()
    pin = _pass_stage3(pdir)
    # Metrics change AFTER stage 3 passed (the phantom-promote scenario).
    full = os.path.join(pdir, "backtest", "full", "metrics.json")
    m = json.load(open(full))
    m["oos_sharpe"] = 9.9
    json.dump(m, open(full, "w"))
    _write_verdict(pdir, "promote", hash_line=pin)
    passed, msg = ev._check_verdict_matches_metrics(pdir)
    assert not passed
    assert "RESET_STAGE:3" in msg


def test_advance_pipeline_handles_reset_sentinel(proj, monkeypatch):
    _, pdir = proj
    state = _init_state()
    state.pinned_full_metrics_hash = "deadbeef"
    state.stages[3].status = "completed"
    state.stages[4].status = "running"
    orch.save_state(state)
    scheduled = []
    monkeypatch.setattr(orch, "_schedule_stage", lambda st, n: scheduled.append(n))

    orch.advance_pipeline("proj", 4, passed=False,
                          feedback="Metrics changed... RESET_STAGE:3")

    reloaded = orch.load_state("proj")
    assert reloaded.stages[3].status == "scheduled"
    assert reloaded.stages[4].status == "scheduled"
    assert reloaded.pinned_full_metrics_hash is None
    assert scheduled == [3]                        # stage 3 requeued, stage 4 waits


def test_legacy_pipeline_without_pin_skips_hash_gate(proj):
    _, pdir = proj
    _init_state()                                   # pin stays None
    json.dump(GOOD_FULL_METRICS, open(os.path.join(pdir, "backtest", "full", "metrics.json"), "w"))
    _write_verdict(pdir, "promote", hash_line=None)
    passed, msg = ev._check_verdict_matches_metrics(pdir)
    assert passed, msg


# --- next_step deadlock recovery (2026-07-16 SOXS: 28 part-completions) ----------


def test_sanitizer_restores_evaluator_next_step(proj):
    # The agent clobbered the evaluator's next_step with narrative; the
    # sanitizer used to discard it to None — a DEADLOCK, because with the doc
    # telling the agent to end every run in part-completion, no evaluation
    # ever ran again to restore it. Now the evaluator's shadow copy wins.
    d, pdir = proj
    path = _write_loop_state(
        pdir, next_step="verify_convergence_again",
        _evaluator_next_step="revise_hypothesis")
    changed = ev.sanitize_loop_state_file(path)
    assert changed
    ls = json.load(open(path))
    assert ls["next_step"] == "revise_hypothesis"


def test_sanitizer_nulls_when_no_shadow(proj):
    d, pdir = proj
    path = _write_loop_state(pdir, next_step="made_up_step")
    ev.sanitize_loop_state_file(path)
    ls = json.load(open(path))
    assert ls["next_step"] is None


def test_sanitizer_ignores_corrupt_shadow(proj):
    d, pdir = proj
    path = _write_loop_state(
        pdir, next_step="made_up", _evaluator_next_step="also_made_up")
    ev.sanitize_loop_state_file(path)
    ls = json.load(open(path))
    assert ls["next_step"] is None


def test_write_decision_persists_shadow(proj):
    d, pdir = proj
    path = _write_loop_state(pdir)
    ls = json.load(open(path))
    ev._write_decision(path, ls, ev.StageDecision(
        type="infeasible_data", passed=False, feedback="f",
        next_step="revise_hypothesis", next_phase="revise_hypothesis"))
    saved = json.load(open(path))
    assert saved["_evaluator_next_step"] == "revise_hypothesis"
