"""Pipeline tools — agent-facing tools to start and monitor pipelines."""

import json
import os
from typing import Callable, Tuple, Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.pipeline.orchestrator import (
    get_num_stages,
    init_pipeline,
    list_all_pipelines,
    load_state,
)


def _validate_idea(description: str) -> Tuple[bool, str]:
    """Check if a startup idea description is clear enough to start a pipeline."""
    if len(description.strip()) < 50:
        return False, (
            "The description is too short. Please provide more detail:\n"
            "- Who is the target customer?\n"
            "- What pain point does this solve, or what pleasure does it provide?\n"
            "- How does the solution work at a high level?"
        )

    lower = description.lower()
    has_who = any(w in lower for w in [
        "customer", "user", "people", "business", "enterprise",
        "consumer", "client", "audience", "demographic",
    ])
    has_what = any(w in lower for w in [
        "solve", "help", "enable", "provide", "platform", "tool", "app", "service", "product",
    ])

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


def _validate_strategy(description: str) -> Tuple[bool, str]:
    """Check if a trading strategy description is clear enough to start a pipeline.

    Requires: universe (what we trade) + signal (what triggers trades).
    Risk rules are recommended but optional — flagged in feedback when missing.
    """
    if len(description.strip()) < 50:
        return False, (
            "The description is too short. Please provide more detail:\n"
            "- What universe / assets will this trade (e.g., S&P 500 stocks, BTC, ETFs)?\n"
            "- What signal or alpha drives entries and exits (e.g., momentum, mean reversion, crossover)?\n"
            "- Risk rules (position sizing, stop loss, max drawdown) — optional but strongly recommended."
        )

    lower = description.lower()
    has_universe = any(w in lower for w in [
        "stocks", "stock", "equity", "equities", "crypto", "bitcoin", "btc", "eth",
        "forex", "currency", "etf", "index", "ticker", "universe", "futures", "options",
    ])
    has_signal = any(w in lower for w in [
        "signal", "alpha", "indicator", "momentum", "mean reversion", "crossover",
        "breakout", "factor", "trend", "sma", "ema", "rsi", "macd", "z-score",
    ])
    has_risk = any(w in lower for w in [
        "stop", "risk", "position", "drawdown", "exposure", "sizing",
    ])

    if not has_universe:
        return False, (
            "The description doesn't clearly define the universe. "
            "Please specify what assets this strategy trades (e.g., 'S&P 500 stocks', "
            "'top 10 crypto by market cap', 'SPY and QQQ ETFs')."
        )

    if not has_signal:
        return False, (
            "The description doesn't clearly define the entry/exit signal. "
            "Please specify what triggers trades (e.g., '20/50-day SMA crossover', "
            "'mean reversion on 2-sigma Bollinger band touch', 'momentum top decile rebalance')."
        )

    note = " Note: no explicit risk rules mentioned — stage 2 will need to define them." if not has_risk else ""
    return True, f"Strategy description is clear enough to proceed.{note}"


def _start_pipeline_impl(
    name: str,
    description: str,
    pipeline_type: str,
    validator: Callable[[str], Tuple[bool, str]],
    stage_sequence_label: str,
) -> str:
    """Shared implementation for starting a pipeline of any type."""
    name = name.strip().lower().replace(" ", "-")

    is_valid, feedback = validator(description)
    if not is_valid:
        return f"Cannot start pipeline — idea needs clarification:\n\n{feedback}"

    # Create project if it doesn't already exist
    from sandbox_agent.tools.project_tools import _project_dir
    pdir = _project_dir(name)
    if not os.path.exists(pdir):
        from sandbox_agent.tools.project_tools import CreateProject
        CreateProject().call(json.dumps({"name": name, "description": description}))

    state = init_pipeline(name, description, pipeline_type=pipeline_type)

    # Check if pipeline already has a scheduled task (stage 1 was already scheduled)
    has_task = any(s.task_id is not None for s in state.stages.values())
    if state.status == "running" and has_task:
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
        f"Pipeline started for '{name}' (type: {pipeline_type}).\n"
        f"Stage 1 scheduled. The pipeline will run through {get_num_stages(pipeline_type)} stages automatically:\n"
        f"{stage_sequence_label}\n"
        f"Use pipeline_status(project='{name}') to check progress."
    )


@register_tool("start_pipeline")
class StartPipeline(BaseTool):
    """Start a 6-stage Startup Builder Pipeline."""

    name = "start_pipeline"
    description = (
        "Start a 6-stage Startup Builder Pipeline to research, plan, and build an MVP for a startup idea. "
        "Stages: Market Research → BRD → PRD → VC Pitch → MVP → Review. "
        "Each stage runs independently with acceptance evaluation between stages. "
        "A global lock serializes all pipelines — if another pipeline (startup or trading) is currently "
        "running, this one will wait. "
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
        return _start_pipeline_impl(
            name=params["name"],
            description=params["description"],
            pipeline_type="startup",
            validator=_validate_idea,
            stage_sequence_label="1. Market Research → 2. BRD → 3. PRD → 4. VC Pitch → 5. MVP → 6. Review",
        )


@register_tool("start_trading_pipeline")
class StartTradingPipeline(BaseTool):
    """Start a 6-stage Trading Strategy Builder Pipeline."""

    name = "start_trading_pipeline"
    description = (
        "Start a 6-stage Trading Strategy Builder Pipeline to research, backtest, and scaffold paper "
        "trading for an algorithmic trading strategy. "
        "Stages: Strategy Research → Strategy Spec → Data Pipeline → Backtest → Paper Trading → Review. "
        "Uses the exec tool to install yfinance/pandas-ta/backtrader and actually run backtests. "
        "A global lock serializes all pipelines — if another pipeline (startup or trading) is currently "
        "running, this one will wait. "
        "Use this when the user describes a trading strategy idea with a universe, a signal, and (ideally) risk rules."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Project name (kebab-case, e.g., 'momentum-qqq').",
            },
            "description": {
                "type": "string",
                "description": (
                    "Clear description of the trading strategy. Must include: "
                    "the universe (what assets to trade), the signal (what triggers entries/exits), "
                    "and ideally risk rules (position sizing, stop loss, max drawdown)."
                ),
            },
        },
        "required": ["name", "description"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        return _start_pipeline_impl(
            name=params["name"],
            description=params["description"],
            pipeline_type="trading",
            validator=_validate_strategy,
            stage_sequence_label=(
                "1. Strategy Research → 2. Strategy Spec → 3. Data Pipeline → "
                "4. Backtest → 5. Paper Trading → 6. Review"
            ),
        )


@register_tool("pipeline_status")
class PipelineStatusTool(BaseTool):
    """Check the status of a pipeline."""

    name = "pipeline_status"
    description = "Check the status of a pipeline (startup or trading). Shows each stage, attempts, and notes."
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
    description = "List all pipelines (startup or trading, active/completed/failed)."
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
            lines.append(f"- **{state.project_name}** ({state.pipeline_type}): {state.status}{stage_info}")
        return "\n".join(lines)


def _format_pipeline_status(state) -> str:
    """Format a pipeline state as a readable status table."""
    num_stages = get_num_stages(state.pipeline_type)
    lines = [
        f"## Pipeline: {state.project_name}",
        f"**Type**: {state.pipeline_type} | **Status**: {state.status} | **Current Stage**: {state.current_stage}/{num_stages}",
        f"**Description**: {state.description[:200]}",
        f"",
        f"| # | Stage | Status | Attempts |",
        f"|---|-------|--------|----------|",
    ]

    for num in range(1, num_stages + 1):
        stage = state.stages.get(num)
        if stage:
            lines.append(f"| {num} | {stage.stage_name} | {stage.status} | {stage.run_count}/{stage.max_attempts} |")

    return "\n".join(lines)
