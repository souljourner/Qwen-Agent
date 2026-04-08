"""Stage runner — builds prompts, loads artifacts, runs agent for each pipeline stage."""

import logging
import os
from typing import List, Optional

from qwen_agent.llm.schema import Message

from sandbox_agent.config import DATA_DIR
from sandbox_agent.pipeline.evaluator import evaluate_stage
from sandbox_agent.pipeline.models import PipelineState
from sandbox_agent.pipeline.orchestrator import (
    STAGES,
    advance_pipeline,
    acquire_lock,
    get_stage_inputs,
    load_stage_instructions,
    load_state,
    mark_stage_failed,
    mark_stage_part_completion,
    mark_stage_running,
    release_lock,
    save_state,
)

logger = logging.getLogger(__name__)


def run_pipeline_stage(task, system_message: str) -> str:
    """Run a single pipeline stage. Called by cron_loop when task name starts with 'pipeline:'.

    Args:
        task: The Task object from TaskQueue
        system_message: The base system message (SOUL.md + MEMORIES.md)

    Returns:
        Result text from the agent
    """
    # Parse task name: "pipeline:{project}:stage_{n}"
    parts = task.name.split(":")
    if len(parts) < 3:
        logger.error(f"Invalid pipeline task name: {task.name}")
        return "Invalid pipeline task name"

    project_name = parts[1]
    stage_str = parts[2]  # "stage_1", "stage_2", etc.
    try:
        stage_number = int(stage_str.replace("stage_", ""))
    except ValueError:
        logger.error(f"Cannot parse stage number from: {stage_str}")
        return f"Invalid stage: {stage_str}"

    if stage_number not in STAGES:
        logger.error(f"Unknown stage number: {stage_number}")
        return f"Unknown stage: {stage_number}"

    # Acquire pipeline lock
    if not acquire_lock(task.id):
        logger.info(f"Pipeline lock held, skipping {task.name}")
        return "Pipeline lock held — will retry next cycle"

    try:
        return _execute_stage(project_name, stage_number, task.id, system_message)
    except Exception as e:
        logger.exception(f"Pipeline stage {stage_number} failed for {project_name}")
        mark_stage_failed(project_name, stage_number, str(e))
        return f"Stage {stage_number} failed: {e}"
    finally:
        release_lock()


def _execute_stage(project_name: str, stage_number: int, task_id: str, system_message: str) -> str:
    """Execute a single stage: load context, run agent, evaluate results."""
    from sandbox_agent.main import run_on_best_available

    state = load_state(project_name)
    if not state:
        return f"Pipeline state not found for {project_name}"

    stage = state.stages.get(stage_number)
    if not stage:
        return f"Stage {stage_number} not found in pipeline state"

    # Mark as running
    mark_stage_running(project_name, stage_number, task_id)

    # Load instruction markdown
    instructions = load_stage_instructions(stage_number)

    # Load artifacts from previous stages
    artifact_contents = _load_artifacts(project_name, stage_number)

    # Build the user message
    user_message = _build_prompt(state, stage, instructions, artifact_contents)

    # Run the agent
    logger.info(f"Pipeline: running {project_name} stage {stage_number} ({STAGES[stage_number]['name']}), attempt {stage.run_count + 1}/{stage.max_attempts}")
    messages = [Message(role="user", content=user_message)]
    response = run_on_best_available(system_message, messages)

    # Extract result text
    result_text = ""
    for msg in response:
        if msg.role == "assistant" and isinstance(msg.content, str):
            result_text += msg.content

    # Check for part-completion (MVP stage or token exhaustion)
    if _detect_part_completion(project_name, stage_number, result_text):
        mark_stage_part_completion(project_name, stage_number, result_text[:500])
        return f"Stage {stage_number} partially complete — will continue next run"

    # Run acceptance evaluation
    passed, feedback = evaluate_stage(project_name, stage_number, system_message)
    advance_pipeline(project_name, stage_number, passed, feedback)

    status = "PASSED" if passed else "FAILED"
    logger.info(f"Pipeline: {project_name} stage {stage_number} acceptance: {status}")
    return f"Stage {stage_number} ({STAGES[stage_number]['name']}): {status}. {feedback[:200]}"


def _load_artifacts(project_name: str, stage_number: int) -> str:
    """Load input artifacts for a stage from previous stages."""
    project_dir = os.path.join(DATA_DIR, "projects", project_name)
    inputs = get_stage_inputs(stage_number)
    parts = []

    for artifact_path in inputs:
        full_path = os.path.join(project_dir, artifact_path)
        if os.path.exists(full_path):
            try:
                content = open(full_path).read()
                # Cap each artifact to avoid blowing up context
                if len(content) > 30000:
                    content = content[:30000] + "\n\n... (truncated)"
                parts.append(f"### Artifact: {artifact_path}\n\n{content}")
            except Exception as e:
                parts.append(f"### Artifact: {artifact_path}\n\n(Error reading: {e})")
        else:
            parts.append(f"### Artifact: {artifact_path}\n\n(Not yet created)")

    return "\n\n---\n\n".join(parts) if parts else "(No input artifacts for this stage)"


def _build_prompt(state: PipelineState, stage, instructions: str, artifact_contents: str) -> str:
    """Build the complete user message for a stage execution."""
    stage_defn = STAGES[stage.stage_number]

    parts = [
        f"# Pipeline Stage {stage.stage_number}: {stage_defn['name'].replace('_', ' ').title()}",
        f"",
        f"## Project: {state.project_name}",
        f"## Description: {state.description}",
        f"## Attempt: {stage.run_count + 1} of {stage.max_attempts}",
        f"",
        f"## Instructions",
        f"",
        instructions,
        f"",
    ]

    # Add previous artifacts
    if artifact_contents and artifact_contents != "(No input artifacts for this stage)":
        parts.extend([
            f"## Previous Stage Artifacts",
            f"",
            artifact_contents,
            f"",
        ])

    # Add notes from previous attempts
    if stage.notes.strip():
        parts.extend([
            f"## Notes from Previous Attempts",
            f"",
            stage.notes,
            f"",
        ])

    # Add output requirements
    parts.extend([
        f"## Output Requirements",
        f"",
        f"Save your output to these files using project_write_file(project='{state.project_name}', path='...', content='...'):",
    ])
    for artifact in stage.artifacts:
        parts.append(f"- `{artifact}`")

    if stage_defn["required_sections"]:
        parts.append(f"")
        parts.append(f"Required sections in the output:")
        for section in stage_defn["required_sections"]:
            parts.append(f"- ## {section}")

    return "\n".join(parts)


def _detect_part_completion(project_name: str, stage_number: int, result_text: str) -> bool:
    """Detect if a stage ended due to token/tool call exhaustion (part-completion)."""
    # Check for explicit part-completion signals
    lower = result_text.lower()
    if any(signal in lower for signal in [
        "part-completion",
        "ran out of tool calls",
        "token budget",
        "will continue",
        "to be continued",
        "continuing in next",
    ]):
        return True

    # For MVP stage, check if some but not all expected files exist
    if stage_number == 5:
        project_dir = os.path.join(DATA_DIR, "projects", project_name)
        mvp_dir = os.path.join(project_dir, "mvp")
        if os.path.isdir(mvp_dir):
            files = os.listdir(mvp_dir)
            has_readme = "README.md" in files
            has_code = any(f.endswith((".py", ".js", ".ts", ".html")) for f in files)
            has_tests = os.path.isdir(os.path.join(mvp_dir, "tests"))
            if has_code and not (has_readme and has_tests):
                return True  # Some files but not complete

    return False
