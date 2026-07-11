"""Legacy pipeline-state compatibility + startup quarantine of unparseable
state files (the boot-time 'Corrupt pipeline state' warnings).

Live evidence: shapiro-cot-contrarian / neumann-puzzle-v2 hold REAL history
with statuses older code wrote ('skipped', 'skipped-rejected', 'complete');
soxs-perpetual-short-compound's state.json is agent-fabricated shadow content
(no project_name/description) from the phantom-promote era."""

import json
import os

import pytest

from sandbox_agent.pipeline.models import PipelineState, StageState


def test_legacy_statuses_normalize():
    assert StageState(stage_number=1, stage_name="x", status="complete").status == "completed"
    assert StageState(stage_number=5, stage_name="x", status="skipped").status == "completed"
    assert StageState(stage_number=6, stage_name="x", status="skipped-rejected").status == "completed"


def test_canonical_statuses_untouched():
    for s in ("scheduled", "running", "part-completion", "completed",
              "completed-no-more-attempts", "failed", "failed-no-more-attempts"):
        assert StageState(stage_number=1, stage_name="x", status=s).status == s


def test_legacy_state_file_shape_loads():
    # The neumann/shapiro shape: real orchestrator file + legacy stage statuses.
    data = {
        "project_name": "neumann-puzzle-v2", "description": "d",
        "pipeline_type": "trading", "current_stage": 6, "status": "completed",
        "stages": {
            "5": {"stage_number": 5, "stage_name": "paper_trading", "status": "skipped"},
            "6": {"stage_number": 6, "stage_name": "review", "status": "completed"},
        },
    }
    state = PipelineState(**data)
    assert state.stages[5].status == "completed"


def test_quarantine_unparseable_state(tmp_path, monkeypatch):
    import sandbox_agent.migrations as mig
    monkeypatch.setattr(mig, "DATA_DIR", str(tmp_path))
    proj = tmp_path / "projects" / "junk-proj" / "pipeline"
    proj.mkdir(parents=True)
    # Agent-fabricated shadow content: missing required fields entirely.
    (proj / "state.json").write_text(json.dumps(
        {"stages": {"1": {"status": "complete"}}, "verdict": "PROMOTE"}))

    mig.quarantine_unparseable_pipeline_state()
    assert not (proj / "state.json").exists()
    quarantined = list(proj.glob("state.json.unparseable-*"))
    assert len(quarantined) == 1
    assert "PROMOTE" in quarantined[0].read_text()      # preserved, not deleted

    mig.quarantine_unparseable_pipeline_state()          # idempotent
    assert len(list(proj.glob("state.json.unparseable-*"))) == 1


def test_quarantine_leaves_valid_and_legacy_states(tmp_path, monkeypatch):
    import sandbox_agent.migrations as mig
    monkeypatch.setattr(mig, "DATA_DIR", str(tmp_path))
    proj = tmp_path / "projects" / "ok-proj" / "pipeline"
    proj.mkdir(parents=True)
    (proj / "state.json").write_text(json.dumps({
        "project_name": "ok-proj", "description": "d", "pipeline_type": "trading",
        "stages": {"5": {"stage_number": 5, "stage_name": "paper", "status": "skipped"}},
    }))
    mig.quarantine_unparseable_pipeline_state()
    assert (proj / "state.json").exists()               # normalizes, not quarantined
