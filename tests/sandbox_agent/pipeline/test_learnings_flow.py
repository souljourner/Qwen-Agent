"""Weakness #2C: pipeline learnings must flow across projects.

On pipeline completion the review's `## Learnings` section is extracted into
a global DATA_DIR/learnings/pipeline-learnings.md (deterministic, idempotent);
stage-1 prompts inject the newest learnings so a new pipeline starts informed."""

import os
import shutil
import tempfile

import pytest

import sandbox_agent.pipeline.orchestrator as orch
import sandbox_agent.pipeline.stage_runner as sr


@pytest.fixture
def env(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE", os.path.join(d, "pipeline.lock"))
    monkeypatch.setattr("sandbox_agent.pipeline.stage_runner.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.tools.self_edit_tools.DATA_DIR", d)
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "autocommit", lambda *a, **k: None)
    yield d
    shutil.rmtree(d)


REVIEW = """# Review

## Performance Summary
Fine.

## Learnings
- Signal decay matters more than entry timing for leveraged shorts.
- Walk-forward windows under 3 months overfit.

## Deployment Readiness
Ready.
"""


def _complete_pipeline(env, project="proj-a"):
    os.makedirs(os.path.join(env, "projects", project, "pipeline"), exist_ok=True)
    with open(os.path.join(env, "projects", project, "pipeline", "review.md"), "w") as f:
        f.write(REVIEW)
    state = orch.init_pipeline(project, "test", pipeline_type="trading")
    for n in range(1, 6):
        state.stages[n].status = "completed"
    state.stages[6].status = "running"
    orch.save_state(state)
    orch.advance_pipeline(project, 6, passed=True, feedback="ok")


def test_learnings_extracted_on_completion(env):
    _complete_pipeline(env)
    path = os.path.join(env, "learnings", "pipeline-learnings.md")
    body = open(path).read()
    assert "Signal decay matters" in body
    assert "proj-a" in body                         # attributed to the project


def test_extraction_idempotent(env):
    _complete_pipeline(env)
    state = orch.load_state("proj-a")
    state.stages[6].status = "running"
    orch.save_state(state)
    orch.advance_pipeline("proj-a", 6, passed=True, feedback="ok")
    body = open(os.path.join(env, "learnings", "pipeline-learnings.md")).read()
    assert body.count("Signal decay matters") == 1


def test_multiple_projects_accumulate(env):
    _complete_pipeline(env, "proj-a")
    _complete_pipeline(env, "proj-b")
    body = open(os.path.join(env, "learnings", "pipeline-learnings.md")).read()
    assert "proj-a" in body and "proj-b" in body


def test_stage1_prompt_injects_learnings(env):
    _complete_pipeline(env)
    state = orch.init_pipeline("proj-new", "another idea", pipeline_type="trading")
    prompt = sr._build_prompt(state, state.stages[1],
                              instructions="do stage 1", artifact_contents="")
    assert "Prior Pipeline Learnings" in prompt
    assert "Signal decay matters" in prompt


def test_stage2_prompt_not_injected(env):
    _complete_pipeline(env)
    state = orch.init_pipeline("proj-new", "another idea", pipeline_type="trading")
    prompt = sr._build_prompt(state, state.stages[2],
                              instructions="do stage 2", artifact_contents="")
    assert "Prior Pipeline Learnings" not in prompt


def test_no_learnings_file_no_injection(env):
    state = orch.init_pipeline("proj-new", "idea", pipeline_type="trading")
    prompt = sr._build_prompt(state, state.stages[1],
                              instructions="do stage 1", artifact_contents="")
    assert "Prior Pipeline Learnings" not in prompt
