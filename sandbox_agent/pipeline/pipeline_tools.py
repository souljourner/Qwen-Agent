"""Pipeline tools — agent-facing tools to start and monitor pipelines."""

import json
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.pipeline.orchestrator import (
    init_pipeline,
    list_all_pipelines,
    load_state,
    save_state,
    STAGES,
    NUM_STAGES,
)


def _validate_idea(description: str) -> tuple:
    """Check if the idea description is clear enough to start a pipeline.

    Returns (is_valid, feedback).
    """
    if len(description.strip()) < 50:
        return False, (
            "The description is too short. Please provide more detail:\n"
            "- Who is the target customer?\n"
            "- What pain point does this solve, or what pleasure does it provide?\n"
            "- How does the solution work at a high level?"
        )

    lower = description.lower()
    has_who = any(w in lower for w in ["customer", "user", "people", "business", "enterprise", "consumer", "client", "audience", "demographic"])
    has_what = any(w in lower for w in ["solve", "help", "enable", "provide", "platform", "tool", "app", "service", "product"])

    if not has_who:
        return False, (
            "The description doesn't clearly define who the target customer is. "
            "Please specify: Who will use this? What type of person or business? "
            "What is their demographic, industry, or situation?"
        )

    if not has_what:
        return False, (
            "The description doesn't clearly define what the solution does. "
            "Please specify: What pain point does this solve? "
            "What does the product/service actually do for the customer?"
        )

    return True, "Description is clear enough to proceed."


@register_tool("start_pipeline")
class StartPipeline(BaseTool):
    """Start a 6-stage Startup Builder Pipeline."""

    name = "start_pipeline"
    description = (
        "Start a 6-stage Startup Builder Pipeline to research, plan, and build an MVP for a startup idea. "
        "Stages: Market Research → BRD → PRD → VC Pitch → MVP → Review. "
        "Each stage runs independently with acceptance evaluation between stages. "
        "If the project already exists and is completed, it will rerun to improve artifacts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Project name (kebab-case, e.g., 'pet-health-ai').",
            },
            "description": {
                "type": "string",
                "description": (
                    "Clear description of the startup idea. Must include: "
                    "who the target customer is, what pain point it solves, "
                    "and how the solution works."
                ),
            },
        },
        "required": ["name", "description"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        name = params["name"].strip().lower().replace(" ", "-")
        description = params["description"]

        # Validate prerequisites
        is_valid, feedback = _validate_idea(description)
        if not is_valid:
            return f"Cannot start pipeline — idea needs clarification:\n\n{feedback}"

        # Initialize pipeline (creates project if needed, resets if completed)
        from sandbox_agent.tools.project_tools import CreateProject
        create_tool = CreateProject()
        create_tool.call(json.dumps({"name": name, "description": description}))

        state = init_pipeline(name, description)

        # Check if pipeline already has a scheduled task (stage 1 was already scheduled)
        has_task = any(s.task_id is not None for s in state.stages.values())
        if state.status == "running" and has_task:
            # Pipeline already in progress
            stage = state.stages.get(state.current_stage)
            return (
                f"Pipeline '{name}' is already running.\n"
                f"Current stage: {state.current_stage} ({stage.stage_name if stage else '?'})\n"
                f"Status: {stage.status if stage else '?'}\n"
                f"Use pipeline_status(project='{name}') for details."
            )

        # Schedule stage 1
        from sandbox_agent.pipeline.orchestrator import _schedule_stage
        _schedule_stage(state, 1)

        return (
            f"Pipeline started for '{name}'.\n"
            f"Stage 1 (Market Research) scheduled.\n"
            f"The pipeline will run through 6 stages automatically:\n"
            f"1. Market Research → 2. BRD → 3. PRD → 4. VC Pitch → 5. MVP → 6. Review\n"
            f"Use pipeline_status(project='{name}') to check progress."
        )


@register_tool("pipeline_status")
class PipelineStatusTool(BaseTool):
    """Check the status of a pipeline."""

    name = "pipeline_status"
    description = "Check the status of a Startup Builder Pipeline. Shows each stage, attempts, and notes."
    parameters = {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name. Omit to show all pipelines.",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        project = params.get("project")

        if project:
            state = load_state(project)
            if not state:
                return f"No pipeline found for project '{project}'."
            return _format_pipeline_status(state)
        else:
            pipelines = list_all_pipelines()
            if not pipelines:
                return "No active pipelines."
            parts = []
            for state in pipelines:
                parts.append(_format_pipeline_status(state))
            return "\n\n---\n\n".join(parts)


@register_tool("list_pipelines")
class ListPipelines(BaseTool):
    """List all projects with active pipelines."""

    name = "list_pipelines"
    description = "List all Startup Builder Pipelines (active, completed, or failed)."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        pipelines = list_all_pipelines()
        if not pipelines:
            return "No pipelines found."

        lines = ["Pipelines:\n"]
        for state in pipelines:
            stage_info = ""
            if state.current_stage > 0:
                stage = state.stages.get(state.current_stage)
                stage_info = f" — Stage {state.current_stage} ({stage.stage_name}): {stage.status}" if stage else ""
            lines.append(f"- **{state.project_name}**: {state.status}{stage_info}")
        return "\n".join(lines)


def _format_pipeline_status(state) -> str:
    """Format a pipeline state as a readable status table."""
    lines = [
        f"## Pipeline: {state.project_name}",
        f"**Status**: {state.status} | **Current Stage**: {state.current_stage}/{NUM_STAGES}",
        f"**Description**: {state.description[:200]}",
        f"",
        f"| # | Stage | Status | Attempts |",
        f"|---|-------|--------|----------|",
    ]

    for num in range(1, NUM_STAGES + 1):
        stage = state.stages.get(num)
        if stage:
            lines.append(f"| {num} | {stage.stage_name} | {stage.status} | {stage.run_count}/{stage.max_attempts} |")

    return "\n".join(lines)
