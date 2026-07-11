"""Stage-6 self-improvement loop: review suggestions must become applicable.

Before: stage instructions were bundled-only (the agent user can't write /app,
so 'instruction improvement suggestions' from stage-6 reviews were dead
letters). Now DATA_DIR/pipeline_stages/<type>/ overrides win, and completing
trading stage 6 files a heartbeat follow-up pointing the agent at the review.
"""

import os
import shutil
import tempfile

import pytest

import sandbox_agent.pipeline.orchestrator as orch


@pytest.fixture
def env(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE", os.path.join(d, "pipeline.lock"))
    monkeypatch.setattr("sandbox_agent.tools.self_edit_tools.DATA_DIR", d)
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "autocommit", lambda *a, **k: None)
    yield d
    shutil.rmtree(d)


def test_data_dir_stage_instructions_override(env):
    bundled = orch.load_stage_instructions(2, "trading")
    assert "research loop" in bundled.lower() or len(bundled) > 200   # sanity: real file

    override_dir = os.path.join(env, "pipeline_stages", "trading")
    os.makedirs(override_dir)
    with open(os.path.join(override_dir, "stage_2_research_loop.md"), "w") as f:
        f.write("# Custom stage 2\nimproved instructions here\n")
    assert "improved instructions here" in orch.load_stage_instructions(2, "trading")


def test_bundled_used_when_no_override(env):
    out = orch.load_stage_instructions(1, "trading")
    assert "Data Landscape" in out


def test_stage6_completion_files_heartbeat_followup(env, monkeypatch):
    state = orch.init_pipeline("proj", "test", pipeline_type="trading")
    for n in range(1, 6):
        state.stages[n].status = "completed"
    state.stages[6].status = "running"
    orch.save_state(state)
    monkeypatch.setattr(orch, "_schedule_stage", lambda *a: None)

    orch.advance_pipeline("proj", 6, passed=True, feedback="ok")

    hb = open(os.path.join(env, "HEARTBEAT.md")).read()
    assert "proj" in hb and "review.md" in hb
    assert "pipeline_stages" in hb          # tells the agent WHERE overrides go

    # Idempotent: completing again doesn't duplicate the item.
    state = orch.load_state("proj")
    state.stages[6].status = "running"
    orch.save_state(state)
    orch.advance_pipeline("proj", 6, passed=True, feedback="ok")
    hb2 = open(os.path.join(env, "HEARTBEAT.md")).read()
    assert hb2.count("review.md") == hb.count("review.md")
