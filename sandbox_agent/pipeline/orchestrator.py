"""Pipeline orchestrator — state machine controlling multi-stage project builds.

Supports multiple pipeline types (startup, trading) via STAGE_REGISTRY. Every
caller looks up stages by pipeline_type; there is no global STAGES constant.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from sandbox_agent.config import DATA_DIR
from sandbox_agent.pipeline.models import PipelineState, StageState

logger = logging.getLogger(__name__)

# Global lock — only one pipeline of any type can run at a time.
LOCK_FILE = os.path.join(DATA_DIR, "pipeline.lock")
LOCK_STALE_SECONDS = 7200  # 2 hours

_PIPELINE_ROOT = Path(__file__).parent

# Startup Builder: 6-stage business/product/MVP pipeline.
STARTUP_STAGES = {
    1: {
        "name": "market_research",
        "inputs": [],
        "outputs": ["research/market-research.md"],
        "required_sections": ["Market Size", "Competitors", "Target Customers", "Timing"],
    },
    2: {
        "name": "brd",
        "inputs": ["research/market-research.md"],
        "outputs": ["business/brd.md"],
        "required_sections": ["Branding", "Legal", "Scalability", "Operations", "Finance"],
    },
    3: {
        "name": "prd",
        "inputs": ["research/market-research.md", "business/brd.md"],
        "outputs": ["product/prd.md"],
        "required_sections": ["User Stories", "MVP Scope", "Technical Requirements"],
    },
    4: {
        "name": "vc_pitch",
        "inputs": ["research/market-research.md", "business/brd.md", "product/prd.md"],
        "outputs": ["business/vc-pitch.md"],
        "required_sections": ["Elevator Pitch", "Full Pitch"],
    },
    5: {
        "name": "mvp",
        "inputs": ["product/prd.md"],
        "outputs": ["mvp/README.md"],
        "required_sections": [],
    },
    6: {
        "name": "review",
        "inputs": [
            "research/market-research.md", "business/brd.md",
            "product/prd.md", "business/vc-pitch.md", "mvp/README.md",
        ],
        "outputs": ["pipeline/review.md", "status.md"],
        "required_sections": ["Summary", "Stage Results", "Learnings"],
    },
}

# Trading Strategy Builder: 6-stage research→backtest→paper trading pipeline.
TRADING_STAGES = {
    1: {
        "name": "strategy_research",
        "inputs": [],
        "outputs": ["research/strategy-research.md"],
        "required_sections": ["Alpha Hypothesis", "Market Regime", "Prior Work", "Universe"],
    },
    2: {
        "name": "strategy_spec",
        "inputs": ["research/strategy-research.md"],
        "outputs": ["strategy/spec.md"],
        "required_sections": [
            "Entry Rules", "Exit Rules", "Position Sizing",
            "Risk Limits", "Expected Performance",
        ],
    },
    3: {
        "name": "data_pipeline",
        "inputs": ["strategy/spec.md"],
        "outputs": ["data/pipeline.py", "data/README.md", "data/requirements.txt"],
        "required_sections": ["Data Sources", "Features", "Quality Checks"],
    },
    4: {
        "name": "backtest",
        "inputs": ["strategy/spec.md", "data/README.md"],
        "outputs": ["backtest/strategy.py", "backtest/results.md"],
        "required_sections": ["Sharpe", "Max Drawdown", "CAGR", "Win Rate"],
    },
    5: {
        "name": "paper_trading",
        "inputs": ["strategy/spec.md", "backtest/strategy.py"],
        "outputs": ["paper/deploy.py", "paper/README.md"],
        "required_sections": ["Broker Integration", "Monitoring", "Kill Switch"],
    },
    6: {
        "name": "review",
        "inputs": [
            "research/strategy-research.md", "strategy/spec.md",
            "data/README.md", "backtest/results.md", "paper/README.md",
        ],
        "outputs": ["pipeline/review.md", "status.md"],
        "required_sections": ["Performance Summary", "Robustness", "Deployment Readiness"],
    },
}

STAGE_REGISTRY = {
    "startup": {
        "stages": STARTUP_STAGES,
        "instructions_dir": _PIPELINE_ROOT / "stages",
        "acceptance_path": _PIPELINE_ROOT / "stages" / "acceptance_criteria.md",
    },
    "trading": {
        "stages": TRADING_STAGES,
        "instructions_dir": _PIPELINE_ROOT / "stages_trading",
        "acceptance_path": _PIPELINE_ROOT / "stages_trading" / "acceptance_criteria.md",
    },
}


def get_stages(pipeline_type: str) -> dict:
    """Return the stages dict for a given pipeline type."""
    if pipeline_type not in STAGE_REGISTRY:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")
    return STAGE_REGISTRY[pipeline_type]["stages"]


def get_num_stages(pipeline_type: str) -> int:
    """Return the number of stages for a given pipeline type."""
    return len(get_stages(pipeline_type))


def get_instructions_dir(pipeline_type: str) -> Path:
    """Return the stage instruction directory for a given pipeline type."""
    return STAGE_REGISTRY[pipeline_type]["instructions_dir"]


def get_acceptance_path(pipeline_type: str) -> Path:
    """Return the acceptance criteria path for a given pipeline type."""
    return STAGE_REGISTRY[pipeline_type]["acceptance_path"]


def _projects_dir() -> str:
    return os.path.join(DATA_DIR, "projects")


def _state_path(project_name: str) -> str:
    return os.path.join(_projects_dir(), project_name, "pipeline", "state.json")


def load_state(project_name: str) -> Optional[PipelineState]:
    """Load pipeline state for a project."""
    path = _state_path(project_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return PipelineState(**json.load(f))
    except Exception as e:
        logger.warning(f"Corrupt pipeline state for {project_name}: {e}")
        return None


def save_state(state: PipelineState) -> None:
    """Save pipeline state."""
    path = _state_path(state.project_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state.updated_at = datetime.now()
    with open(path, "w") as f:
        json.dump(state.model_dump(mode="json"), f, indent=2, default=str)


def init_pipeline(project_name: str, description: str, pipeline_type: str = "startup") -> PipelineState:
    """Initialize a new pipeline or reset a completed one for rerun."""
    if pipeline_type not in STAGE_REGISTRY:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")

    existing = load_state(project_name)

    if existing and existing.status not in ("completed", "failed"):
        # Pipeline is still active — don't reset
        return existing

    stages_defn = get_stages(pipeline_type)
    stages = {}
    for num, defn in stages_defn.items():
        stages[num] = StageState(
            stage_number=num,
            stage_name=defn["name"],
            artifacts=defn["outputs"],
        )

    state = PipelineState(
        project_name=project_name,
        description=description,
        pipeline_type=pipeline_type,
        current_stage=1,
        status="running",
        stages=stages,
    )
    save_state(state)
    return state


def advance_pipeline(project_name: str, stage_number: int, passed: bool, feedback: str) -> None:
    """Advance the pipeline after a stage completes or fails acceptance."""
    state = load_state(project_name)
    if not state:
        logger.error(f"Pipeline state not found for {project_name}")
        return

    num_stages = get_num_stages(state.pipeline_type)
    stage = state.stages[stage_number]

    if passed:
        stage.status = "completed"
        stage.completed_at = datetime.now()
        stage.acceptance_result = feedback

        if stage_number < num_stages:
            _schedule_next_stage(state, stage_number + 1)
        else:
            state.status = "completed"
            logger.info(f"Pipeline {project_name} completed all stages")
    else:
        stage.run_count += 1
        stage.acceptance_result = feedback

        if stage.run_count >= stage.max_attempts:
            has_artifacts = _check_artifacts_exist(project_name, stage)
            if has_artifacts:
                stage.status = "completed-no-more-attempts"
                logger.warning(f"Pipeline {project_name} stage {stage_number}: max attempts, proceeding with best effort")
            else:
                stage.status = "failed-no-more-attempts"
                logger.error(f"Pipeline {project_name} stage {stage_number}: failed after {stage.max_attempts} attempts")
                try:
                    from sandbox_agent.tools.notification_tools import RequestUser
                    tool = RequestUser()
                    tool.call(json.dumps({
                        "subject": f"Pipeline {project_name} stage {stage_number} failed",
                        "detail": f"Stage {stage.stage_name} failed after {stage.max_attempts} attempts. Last feedback: {feedback[:300]}",
                        "urgency": "high",
                        "project": project_name,
                    }))
                except Exception:
                    pass

            if stage_number < num_stages:
                _schedule_next_stage(state, stage_number + 1)
            else:
                state.status = "completed"
        else:
            stage.notes += f"\n\n### Attempt {stage.run_count} feedback:\n{feedback}"
            stage.status = "scheduled"
            _schedule_stage(state, stage_number)

    state.current_stage = stage_number
    save_state(state)
    _write_status_md(state)


def mark_stage_running(project_name: str, stage_number: int, task_id: str) -> None:
    """Mark a stage as running."""
    state = load_state(project_name)
    if not state:
        return
    stage = state.stages[stage_number]
    stage.status = "running"
    stage.started_at = datetime.now()
    state.lock_holder = task_id
    save_state(state)


def mark_stage_part_completion(project_name: str, stage_number: int, notes: str) -> None:
    """Mark a stage as partially complete (ran out of tokens/tool calls)."""
    state = load_state(project_name)
    if not state:
        return
    stage = state.stages[stage_number]
    stage.status = "part-completion"
    stage.notes += f"\n\n### Partial completion notes:\n{notes}"
    save_state(state)
    _schedule_stage(state, stage_number)


def mark_stage_failed(project_name: str, stage_number: int, error: str) -> None:
    """Mark a stage as failed (exception/timeout)."""
    state = load_state(project_name)
    if not state:
        return
    stage = state.stages[stage_number]
    stage.status = "failed"
    stage.last_error = error[:500]
    stage.run_count += 1

    if stage.run_count >= stage.max_attempts:
        stage.status = "failed-no-more-attempts"
    else:
        stage.status = "scheduled"

    save_state(state)
    _schedule_stage(state, stage_number)


def _schedule_stage(state: PipelineState, stage_number: int) -> None:
    """Schedule a stage as a TaskQueue task."""
    from sandbox_agent.scheduler.scheduler_tools import get_task_queue

    tq = get_task_queue()
    task_name = f"pipeline:{state.project_name}:stage_{stage_number}"
    stage_defn = get_stages(state.pipeline_type)[stage_number]

    task = tq.add_task(
        name=task_name,
        description=f"Pipeline stage {stage_number} ({stage_defn['name']}) for {state.project_name}",
        schedule_type="at",
        run_at=datetime.now(),
        project=state.project_name,
    )
    state.stages[stage_number].task_id = task.id
    save_state(state)


def _schedule_next_stage(state: PipelineState, next_stage_number: int) -> None:
    """Schedule the next stage."""
    state.current_stage = next_stage_number
    _schedule_stage(state, next_stage_number)


def _check_artifacts_exist(project_name: str, stage: StageState) -> bool:
    """Check if any expected artifacts exist for a stage."""
    project_dir = os.path.join(_projects_dir(), project_name)
    for artifact in stage.artifacts:
        if os.path.exists(os.path.join(project_dir, artifact)):
            return True
    return False


def _write_status_md(state: PipelineState) -> None:
    """Write a human-readable status file."""
    project_dir = os.path.join(_projects_dir(), state.project_name)
    os.makedirs(project_dir, exist_ok=True)

    num_stages = get_num_stages(state.pipeline_type)

    lines = [
        f"# Pipeline Status: {state.project_name}",
        f"",
        f"**Type**: {state.pipeline_type}",
        f"**Status**: {state.status}",
        f"**Current Stage**: {state.current_stage} of {num_stages}",
        f"**Created**: {state.created_at}",
        f"**Updated**: {state.updated_at}",
        f"",
        f"| # | Stage | Status | Attempts | Last Run |",
        f"|---|-------|--------|----------|----------|",
    ]

    for num in range(1, num_stages + 1):
        stage = state.stages.get(num)
        if stage:
            last_run = str(stage.started_at)[:16] if stage.started_at else ""
            lines.append(
                f"| {num} | {stage.stage_name} | {stage.status} | {stage.run_count}/{stage.max_attempts} | {last_run} |"
            )

    for num in range(1, num_stages + 1):
        stage = state.stages.get(num)
        if stage and stage.notes.strip():
            lines.append(f"\n### Stage {num} Notes\n{stage.notes[:500]}")

    with open(os.path.join(project_dir, "status.md"), "w") as f:
        f.write("\n".join(lines))


# --- Lock management ---

def acquire_lock(task_id: str) -> bool:
    """Acquire the pipeline lock. Returns True if acquired."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            lock_age = time.time() - os.path.getmtime(LOCK_FILE)
            if lock_age < LOCK_STALE_SECONDS:
                return False
            logger.warning(f"Breaking stale pipeline lock (age: {lock_age:.0f}s)")
        except Exception:
            pass

    with open(LOCK_FILE, "w") as f:
        json.dump({"task_id": task_id, "acquired_at": datetime.now().isoformat()}, f)
    return True


def release_lock() -> None:
    """Release the pipeline lock."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


def clear_lock_on_startup() -> None:
    """Clear stale lock on startup and reset running pipeline stages."""
    if os.path.exists(LOCK_FILE):
        logger.warning("Clearing stale pipeline lock from previous run")
        os.remove(LOCK_FILE)

    projects_dir = _projects_dir()
    if not os.path.isdir(projects_dir):
        return
    for project_name in os.listdir(projects_dir):
        state = load_state(project_name)
        if not state:
            continue
        changed = False
        for stage in state.stages.values():
            if stage.status == "running":
                logger.warning(f"Resetting stuck pipeline stage: {project_name} stage {stage.stage_number}")
                stage.status = "scheduled"
                changed = True
        if changed:
            save_state(state)


def load_stage_instructions(stage_number: int, pipeline_type: str) -> str:
    """Load the markdown instruction file for a stage."""
    stage_name = get_stages(pipeline_type)[stage_number]["name"]
    md_path = get_instructions_dir(pipeline_type) / f"stage_{stage_number}_{stage_name}.md"
    if md_path.exists():
        return md_path.read_text()
    return f"Execute stage {stage_number}: {stage_name}. Save output to the expected artifact files."


def get_stage_inputs(stage_number: int, pipeline_type: str) -> list:
    """Get the input artifact paths for a stage."""
    return get_stages(pipeline_type)[stage_number]["inputs"]


def list_all_pipelines() -> list:
    """List all projects with pipeline state."""
    projects_dir = _projects_dir()
    pipelines = []
    if not os.path.isdir(projects_dir):
        return pipelines
    for project_name in sorted(os.listdir(projects_dir)):
        state = load_state(project_name)
        if state:
            pipelines.append(state)
    return pipelines
