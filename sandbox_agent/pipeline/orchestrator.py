"""Pipeline orchestrator — state machine controlling multi-stage project builds."""

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

LOCK_FILE = os.path.join(DATA_DIR, "pipeline.lock")
LOCK_STALE_SECONDS = 7200  # 2 hours

# Stage definitions: number → (name, input artifacts, output artifacts, required sections)
STAGES = {
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
        "required_sections": [],  # Checked differently — file existence
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

STAGE_INSTRUCTION_DIR = Path(__file__).parent / "stages"
NUM_STAGES = 6


def _projects_dir() -> str:
    return os.path.join(DATA_DIR, "projects")


def _state_path(project_name: str) -> str:
    return os.path.join(_projects_dir(), project_name, "pipeline", "state.json")


def load_state(project_name: str) -> Optional[PipelineState]:
    """Load pipeline state for a project."""
    path = _state_path(project_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return PipelineState(**json.load(f))


def save_state(state: PipelineState) -> None:
    """Save pipeline state."""
    path = _state_path(state.project_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state.updated_at = datetime.now()
    with open(path, "w") as f:
        json.dump(state.model_dump(mode="json"), f, indent=2, default=str)


def init_pipeline(project_name: str, description: str) -> PipelineState:
    """Initialize a new pipeline or reset a completed one for rerun."""
    existing = load_state(project_name)

    if existing and existing.status not in ("completed", "failed"):
        # Pipeline is still active — don't reset
        return existing

    stages = {}
    for num, defn in STAGES.items():
        stages[num] = StageState(
            stage_number=num,
            stage_name=defn["name"],
            artifacts=defn["outputs"],
        )

    state = PipelineState(
        project_name=project_name,
        description=description,
        current_stage=1,
        status="running",
        stages=stages,
    )
    save_state(state)
    return state


def advance_pipeline(project_name: str, stage_number: int, passed: bool, feedback: str) -> None:
    """Advance the pipeline after a stage completes or fails acceptance."""
    from sandbox_agent.scheduler.scheduler_tools import get_task_queue

    state = load_state(project_name)
    if not state:
        logger.error(f"Pipeline state not found for {project_name}")
        return

    stage = state.stages[stage_number]

    if passed:
        stage.status = "completed"
        stage.completed_at = datetime.now()
        stage.acceptance_result = feedback

        if stage_number < NUM_STAGES:
            _schedule_next_stage(state, stage_number + 1)
        else:
            state.status = "completed"
            logger.info(f"Pipeline {project_name} completed all stages")
    else:
        stage.run_count += 1
        stage.acceptance_result = feedback

        if stage.run_count >= stage.max_attempts:
            # Out of attempts
            has_artifacts = _check_artifacts_exist(project_name, stage)
            if has_artifacts:
                stage.status = "completed-no-more-attempts"
                logger.warning(f"Pipeline {project_name} stage {stage_number}: max attempts, proceeding with best effort")
            else:
                stage.status = "failed-no-more-attempts"
                logger.error(f"Pipeline {project_name} stage {stage_number}: failed after {stage.max_attempts} attempts")
                # Notify user
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

            # Proceed to next stage regardless
            if stage_number < NUM_STAGES:
                _schedule_next_stage(state, stage_number + 1)
            else:
                state.status = "completed"
        else:
            # Retry — add feedback to notes
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
    # Reschedule for continuation
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
        stage.status = "scheduled"  # Will be retried

    save_state(state)
    _schedule_stage(state, stage_number)


def _schedule_stage(state: PipelineState, stage_number: int) -> None:
    """Schedule a stage as a TaskQueue task."""
    from sandbox_agent.scheduler.scheduler_tools import get_task_queue

    tq = get_task_queue()
    task_name = f"pipeline:{state.project_name}:stage_{stage_number}"
    stage_defn = STAGES[stage_number]

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

    lines = [
        f"# Pipeline Status: {state.project_name}",
        f"",
        f"**Status**: {state.status}",
        f"**Current Stage**: {state.current_stage} of {NUM_STAGES}",
        f"**Created**: {state.created_at}",
        f"**Updated**: {state.updated_at}",
        f"",
        f"| # | Stage | Status | Attempts | Last Run |",
        f"|---|-------|--------|----------|----------|",
    ]

    for num in range(1, NUM_STAGES + 1):
        stage = state.stages.get(num)
        if stage:
            last_run = str(stage.started_at)[:16] if stage.started_at else ""
            lines.append(
                f"| {num} | {stage.stage_name} | {stage.status} | {stage.run_count}/{stage.max_attempts} | {last_run} |"
            )

    # Add notes for stages with feedback
    for num in range(1, NUM_STAGES + 1):
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

    # Reset any pipeline stages stuck in "running"
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


def load_stage_instructions(stage_number: int) -> str:
    """Load the markdown instruction file for a stage."""
    stage_name = STAGES[stage_number]["name"]
    md_path = STAGE_INSTRUCTION_DIR / f"stage_{stage_number}_{stage_name}.md"
    if md_path.exists():
        return md_path.read_text()
    return f"Execute stage {stage_number}: {stage_name}. Save output to the expected artifact files."


def get_stage_inputs(stage_number: int) -> list:
    """Get the input artifact paths for a stage."""
    return STAGES[stage_number]["inputs"]


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
