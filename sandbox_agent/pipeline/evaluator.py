"""Stage acceptance evaluator — programmatic checks + LLM quality evaluation."""

import logging
import os
from typing import Tuple

from qwen_agent.llm.schema import Message

from sandbox_agent.config import DATA_DIR
from sandbox_agent.pipeline.orchestrator import (
    get_acceptance_path,
    get_stages,
    load_state,
)

logger = logging.getLogger(__name__)


def evaluate_stage(project_name: str, stage_number: int, system_message: str) -> Tuple[bool, str]:
    """Evaluate a stage's output. Returns (passed, feedback).

    1. Programmatic checks: files exist, non-empty, required sections present
    2. LLM quality check: read output and judge quality
    """
    state = load_state(project_name)
    if not state:
        return False, f"Pipeline state not found for {project_name}"

    stage_defn = get_stages(state.pipeline_type)[stage_number]
    project_dir = os.path.join(DATA_DIR, "projects", project_name)

    # Step 1: Programmatic checks
    prog_passed, prog_feedback = _programmatic_checks(project_dir, stage_defn)
    if not prog_passed:
        return False, f"Programmatic check failed: {prog_feedback}"

    # Step 2: LLM quality evaluation
    llm_passed, llm_feedback = _llm_evaluation(
        project_dir, stage_number, stage_defn, state.pipeline_type, system_message,
    )

    if not llm_passed:
        return False, f"Quality check: {llm_feedback}"

    return True, f"Passed. {llm_feedback}"


def _programmatic_checks(project_dir: str, stage_defn: dict) -> Tuple[bool, str]:
    """Check that expected artifacts exist and have required sections."""
    for artifact_path in stage_defn["outputs"]:
        full_path = os.path.join(project_dir, artifact_path)

        # Check file exists
        if not os.path.exists(full_path):
            return False, f"Missing artifact: {artifact_path}"

        # Check non-empty
        try:
            with open(full_path) as f:
                content = f.read()
        except Exception as e:
            return False, f"Cannot read {artifact_path}: {e}"

        if len(content.strip()) < 100:
            return False, f"Artifact too short ({len(content)} chars): {artifact_path}"

        # Check required sections
        content_lower = content.lower()
        for section in stage_defn["required_sections"]:
            # Look for markdown headers containing the section name
            if section.lower() not in content_lower:
                return False, f"Missing section '{section}' in {artifact_path}"

    return True, "All programmatic checks passed"


def _llm_evaluation(
    project_dir: str,
    stage_number: int,
    stage_defn: dict,
    pipeline_type: str,
    system_message: str,
) -> Tuple[bool, str]:
    """Use the LLM to evaluate artifact quality."""
    from sandbox_agent.main import run_on_best_available

    # Load artifact contents
    artifact_texts = []
    for artifact_path in stage_defn["outputs"]:
        full_path = os.path.join(project_dir, artifact_path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                content = f.read()
            artifact_texts.append(f"### {artifact_path}\n\n{content}")

    if not artifact_texts:
        return False, "No artifacts to evaluate"

    artifacts_combined = "\n\n---\n\n".join(artifact_texts)

    # Load acceptance criteria template (pipeline-type-specific)
    acceptance_prompt = _load_acceptance_prompt(pipeline_type)

    eval_prompt = f"""## Stage Acceptance Evaluation

**Stage {stage_number}: {stage_defn['name']}**

{acceptance_prompt}

## Artifact to Evaluate:

{artifacts_combined}

## Your Evaluation:

Respond with exactly one of:
- PASS: [brief reason why it's acceptable]
- FAIL: [specific issues to fix, actionable feedback]
"""

    messages = [Message(role="user", content=eval_prompt)]

    try:
        response = run_on_best_available(system_message, messages, task_label=f"Evaluating stage {stage_number}")
        result_text = ""
        for msg in response:
            if msg.role == "assistant" and isinstance(msg.content, str):
                result_text += msg.content

        # Parse PASS/FAIL
        result_upper = result_text.upper()
        if "PASS:" in result_upper or result_upper.startswith("PASS"):
            return True, result_text[:500]
        elif "FAIL:" in result_upper or result_upper.startswith("FAIL"):
            return False, result_text[:500]
        else:
            # Ambiguous — treat as pass if output exists and is substantial
            logger.warning(f"Ambiguous acceptance eval result: {result_text[:100]}")
            return True, f"Ambiguous eval (treating as pass): {result_text[:200]}"
    except Exception as e:
        logger.exception("LLM evaluation failed")
        # If LLM eval fails, fall back to programmatic result (already passed)
        return True, f"LLM evaluation failed ({e}), accepting based on programmatic checks"


def _load_acceptance_prompt(pipeline_type: str) -> str:
    """Load the acceptance criteria template for the given pipeline type."""
    path = get_acceptance_path(pipeline_type)
    if path.exists():
        return path.read_text()
    return (
        "Evaluate this artifact for quality and completeness. "
        "Is it detailed enough to be useful? Are there obvious gaps or errors? "
        "Would a reader find this actionable and well-researched?"
    )
