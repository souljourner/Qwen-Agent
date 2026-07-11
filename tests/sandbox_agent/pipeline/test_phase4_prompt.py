"""Phase-4 tests: prefix-cache-friendly prompt ordering, notes capping,
loop-state artifact slimming."""

import json
import os
import shutil
import tempfile

import pytest

import sandbox_agent.pipeline.orchestrator as orch
import sandbox_agent.pipeline.stage_runner as sr


@pytest.fixture
def tmp_env(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE", os.path.join(d, "pipeline.lock"))
    monkeypatch.setattr("sandbox_agent.pipeline.stage_runner.DATA_DIR", d)
    os.makedirs(os.path.join(d, "projects", "proj", "strategy"), exist_ok=True)
    yield d
    shutil.rmtree(d)


def _shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def test_volatile_state_only_changes_prompt_tail(tmp_env):
    state = orch.init_pipeline("proj", "test", pipeline_type="trading")
    stage = state.stages[1]
    instructions = "Do the data landscape.\n" * 50      # static block
    artifacts = "### Artifact: research/x.md\n\n" + "data " * 200

    p1 = sr._build_prompt(state, stage, instructions, artifacts)

    stage.run_count = 3                                  # volatile changes
    stage.notes = "### Attempt 1 feedback:\nfix A\n### Attempt 2 feedback:\nfix B"
    p2 = sr._build_prompt(state, stage, instructions, artifacts)

    # The stable prefix must cover header + instructions + artifacts + output
    # requirements — i.e. everything before "## Current Run State".
    cut = p1.index("## Current Run State")
    assert p2[:cut] == p1[:cut], "volatile state leaked into the stable prompt prefix"
    # And the divergence point is deep (past instructions AND artifacts).
    assert _shared_prefix_len(p1, p2) >= cut
    assert cut > len(instructions) + len(artifacts)


def test_notes_capped_to_last_three_attempts():
    notes = "\n".join(f"### Attempt {i} feedback:\nfeedback {i}" for i in range(1, 8))
    out = sr._last_attempt_notes(notes, keep=3)
    assert "feedback 7" in out and "feedback 5" in out
    assert "feedback 1" not in out and "feedback 4" not in out
    assert "omitted" in out


def test_notes_under_cap_unchanged():
    notes = "### Attempt 1 feedback:\nonly one"
    assert sr._last_attempt_notes(notes, keep=3) == notes


def test_loop_state_artifact_drops_run_notes_for_stage3(tmp_env):
    state = orch.init_pipeline("proj", "test", pipeline_type="trading")
    ls = {"pilot_history": [{"version": 1}], "run_notes": ["long"] * 50,
          "oos_cutoff_date": "2025-06-01"}
    with open(os.path.join(tmp_env, "projects", "proj", "strategy", "loop_state.json"), "w") as f:
        json.dump(ls, f)
    out = sr._load_artifacts(state, 3)
    assert "loop_state.json" in out
    assert '"run_notes"' not in out
    assert "run_notes_omitted" in out
    assert "pilot_history" in out


def test_stage2_loop_state_comes_from_dedicated_block_not_artifacts():
    # Stage 2 receives loop_state via _build_loop_state_block (which renders
    # the last 3 run_notes), NOT via the artifacts list — so the artifact-side
    # run_notes filter (`stage_number != 2`) never suppresses stage-2's view.
    from sandbox_agent.pipeline.orchestrator import get_stage_inputs
    assert "strategy/loop_state.json" not in get_stage_inputs(2, "trading")
