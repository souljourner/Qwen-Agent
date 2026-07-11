"""Stage runner — builds prompts, loads artifacts, runs agent for each pipeline stage."""

import json
import logging
import os
import re
from datetime import datetime
from typing import List, Optional

from qwen_agent.llm.schema import Message

from sandbox_agent.config import DATA_DIR
from sandbox_agent.pipeline.evaluator import evaluate_stage
from sandbox_agent.pipeline.models import PipelineState
from sandbox_agent.pipeline.orchestrator import (
    advance_pipeline,
    acquire_lock,
    get_stage_inputs,
    get_stages,
    load_stage_instructions,
    load_state,
    mark_stage_failed,
    mark_stage_part_completion,
    mark_stage_running,
    release_lock,
    save_state,
)

logger = logging.getLogger(__name__)


def _parse_stage_task_name(task_name: str) -> tuple:
    """Parse "pipeline:{project}:stage_{n}[optional-suffix]" -> (project, n).

    Tolerates agent-created variants like "pipeline:X:stage_4_verdict" — the
    old int(replace(...)) parser crashed on those and the failed task then
    re-queued forever."""
    parts = task_name.split(":")
    if len(parts) < 3:
        raise ValueError(f"Invalid pipeline task name: {task_name}")
    project_name = parts[1]
    m = re.match(r"stage_(\d+)", parts[2])
    if not m:
        raise ValueError(f"Cannot parse stage number from: {parts[2]}")
    stage_number = int(m.group(1))
    if stage_number < 1 or stage_number > 6:
        raise ValueError(f"Unknown stage number: {stage_number}")
    return project_name, stage_number


def run_pipeline_stage(task, system_message: str) -> str:
    """Run a single pipeline stage. Called by cron_loop when task name starts with 'pipeline:'.

    Args:
        task: The Task object from TaskQueue
        system_message: The base system message (SOUL.md + MEMORIES.md)

    Returns:
        Result text from the agent
    """
    project_name, stage_number = _parse_stage_task_name(task.name)

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

    stages_defn = get_stages(state.pipeline_type)
    if stage_number not in stages_defn:
        return f"Unknown stage {stage_number} for pipeline_type '{state.pipeline_type}'"

    stage = state.stages.get(stage_number)
    if not stage:
        return f"Stage {stage_number} not found in pipeline state"

    # Terminal guard: a finished stage must never re-run. Stale duplicate tasks
    # (incl. agent-created ones like "stage_4_verdict") used to re-execute
    # completed stages and corrupt state ("phantom promote").
    if stage.status in ("completed", "completed-no-more-attempts", "failed-no-more-attempts"):
        logger.warning(
            f"Pipeline {project_name} stage {stage_number}: skip — status '{stage.status}' is terminal")
        return f"Skip: stage {stage_number} is terminal ({stage.status}); not re-running."

    # Orphan guard: only the task the orchestrator scheduled may run the stage.
    # A stale/duplicate task with a different id is skipped.
    if stage.task_id and task_id and stage.task_id != task_id:
        logger.warning(
            f"Pipeline {project_name} stage {stage_number}: skip — task {task_id} is stale "
            f"(current task is {stage.task_id})")
        return f"Skip: task {task_id} is stale for stage {stage_number}; current task is {stage.task_id}."

    # Budget enforcement: budget_seconds was previously prompt-advisory only.
    # When the cumulative wall-clock since first start exceeds it, finalize the
    # stage (proceed best-effort or fail) instead of running the agent again.
    if stage.budget_seconds and stage.first_started_at:
        elapsed = (datetime.now() - stage.first_started_at).total_seconds()
        if elapsed > stage.budget_seconds:
            from sandbox_agent.pipeline.orchestrator import finalize_stage_budget_exhausted
            finalize_stage_budget_exhausted(project_name, stage_number)
            return (f"Stage {stage_number} budget exhausted "
                    f"({int(elapsed)}s > {stage.budget_seconds}s) — finalized without another run.")

    # Verdict-skip guard: for trading stages 5 and 6, skip execution entirely
    # when pipeline/verdict.md declares a rejection. The orchestrator already
    # sets status=completed_rejected at scheduling time, but a rogue/orphaned
    # task could still reach this point — this is the belt-and-suspenders.
    if state.pipeline_type == "trading" and stage_number in (5, 6):
        skipped = _check_verdict_skip(state, stage_number)
        if skipped:
            return skipped

    # Mark as running
    mark_stage_running(project_name, stage_number, task_id)

    # Load instruction markdown
    instructions = load_stage_instructions(stage_number, state.pipeline_type)

    # Load artifacts from previous stages
    artifact_contents = _load_artifacts(state, stage_number)

    # Build the user message
    user_message = _build_prompt(state, stage, instructions, artifact_contents)

    # Run the agent
    from sandbox_agent.model_tracker import set_agent_status, clear_agent_status
    stage_label = f"Pipeline: {project_name} stage {stage_number} ({stages_defn[stage_number]['name']})"
    logger.info(f"{stage_label}, attempt {stage.run_count + 1}/{stage.max_attempts}")
    set_agent_status(status="pipeline", current_task=stage_label)
    messages = [Message(role="user", content=user_message)]
    response = run_on_best_available(system_message, messages, task_label=stage_label)
    # Re-set status after run_on_best_available clears it
    set_agent_status(status="pipeline", current_task=stage_label)

    # Extract result text
    result_text = ""
    for msg in response:
        if msg.role == "assistant" and isinstance(msg.content, str):
            result_text += msg.content

    # Check for part-completion (MVP stage or token exhaustion)
    if _detect_part_completion(state, stage_number, result_text):
        mark_stage_part_completion(project_name, stage_number, result_text[:500])
        clear_agent_status()
        return f"Stage {stage_number} partially complete — will continue next run"

    # Run acceptance evaluation
    set_agent_status(status="pipeline", current_task=f"{stage_label} — evaluating")
    passed, feedback = evaluate_stage(project_name, stage_number, system_message)
    advance_pipeline(project_name, stage_number, passed, feedback)

    status = "PASSED" if passed else "FAILED"
    logger.info(f"Pipeline: {project_name} stage {stage_number} acceptance: {status}")
    clear_agent_status()
    return f"Stage {stage_number} ({stages_defn[stage_number]['name']}): {status}. {feedback[:200]}"


def _load_artifacts(state: PipelineState, stage_number: int) -> str:
    """Load input artifacts for a stage from previous stages."""
    project_dir = os.path.join(DATA_DIR, "projects", state.project_name)
    inputs = get_stage_inputs(stage_number, state.pipeline_type)
    parts = []

    stage_defn = get_stages(state.pipeline_type).get(stage_number, {})
    max_chars = int(stage_defn.get("input_max_chars", 120000))

    for artifact_path in inputs:
        full_path = os.path.join(project_dir, artifact_path)
        if os.path.exists(full_path):
            try:
                with open(full_path) as f:
                    content = f.read()
                # loop_state as a downstream INPUT (stages 3/6): the evaluator
                # needs the full history, the prompt doesn't — drop the bulky
                # run_notes to keep the artifact block lean and prompt-stable.
                if artifact_path.endswith("loop_state.json") and stage_number != 2:
                    try:
                        ls = json.loads(content)
                        if "run_notes" in ls:
                            n = len(ls.pop("run_notes") or [])
                            ls["run_notes_omitted"] = f"{n} run notes omitted from this view"
                        content = json.dumps(ls, indent=2)
                    except Exception:  # noqa: BLE001 — fall back to raw content
                        pass
                # Cap each artifact to avoid blowing up context
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n... (truncated)"
                parts.append(f"### Artifact: {artifact_path}\n\n{content}")
            except Exception as e:
                parts.append(f"### Artifact: {artifact_path}\n\n(Error reading: {e})")
        else:
            parts.append(f"### Artifact: {artifact_path}\n\n(Not yet created)")

    return "\n\n---\n\n".join(parts) if parts else "(No input artifacts for this stage)"


def _last_attempt_notes(notes: str, keep: int = 3) -> str:
    """Render only the last `keep` per-attempt feedback blocks. Full history
    stays in pipeline/state.json; the prompt tail stays bounded."""
    if not notes.strip():
        return ""
    marker = "### Attempt "
    blocks = notes.split(marker)
    head, attempts = blocks[0], blocks[1:]
    if len(attempts) <= keep:
        return notes
    kept = attempts[-keep:]
    omitted = len(attempts) - keep
    return (f"({omitted} earlier attempt note(s) omitted)\n\n"
            + marker + marker.join(kept))


def _build_prompt(state: PipelineState, stage, instructions: str, artifact_contents: str) -> str:
    """Build the complete user message for a stage execution.

    ORDERING IS LOAD-BEARING for the vLLM prefix cache: everything stable
    across part-completion runs of the same stage comes FIRST (header,
    instructions, prior-stage artifacts, output requirements); everything
    volatile (attempt counter, time budget, loop state, notes) comes LAST.
    With the system message unchanged, consecutive runs then share a
    many-thousand-token KV prefix instead of diverging ~200 tokens in at the
    attempt counter."""
    stage_defn = get_stages(state.pipeline_type)[stage.stage_number]

    # --- stable prefix ------------------------------------------------------
    parts = [
        f"# Pipeline Stage {stage.stage_number}: {stage_defn['name'].replace('_', ' ').title()}",
        f"",
        f"## Project: {state.project_name}",
        f"## Description: {state.description}",
        f"",
        f"## Instructions",
        f"",
        instructions,
        f"",
    ]

    if artifact_contents and artifact_contents != "(No input artifacts for this stage)":
        parts.extend([
            f"## Previous Stage Artifacts",
            f"",
            artifact_contents,
            f"",
        ])

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
    parts.append(f"")

    # Prior pipeline learnings (stage 1 only): seed a fresh pipeline with what
    # earlier pipelines concluded (auto-extracted from their stage-6 reviews).
    if stage.stage_number == 1:
        learnings_path = os.path.join(DATA_DIR, "learnings", "pipeline-learnings.md")
        if os.path.exists(learnings_path):
            try:
                learnings = open(learnings_path).read()[-8000:]
                parts.extend([
                    f"## Prior Pipeline Learnings",
                    f"Lessons from previously completed pipelines — factor them into "
                    f"your approach (avoid repeating known dead ends):",
                    f"",
                    learnings,
                    f"",
                ])
            except OSError:
                pass

    # --- volatile tail ------------------------------------------------------
    parts.extend([
        f"## Current Run State",
        f"",
        f"Attempt: {stage.run_count + 1} of {stage.max_attempts}",
        f"",
    ])

    # Time budget (only emitted when the stage definition sets one)
    budget_block = _build_time_budget_block(stage)
    if budget_block:
        parts.extend([budget_block, f""])

    # Loop state (trading stage 2 only): inject next_step, current_phase,
    # pilot_history, hypothesis_notes (all), and last 3 run_notes.
    loop_block = _build_loop_state_block(state, stage.stage_number)
    if loop_block:
        parts.extend([loop_block, f""])

    # MVP filesystem audit (startup stage 6 only): inject a real `ls mvp/` so
    # the review has ground truth about what code actually exists. Without this
    # the review hallucinates "Code Files Created: 0" based on inference from
    # the README.
    mvp_audit_block = _build_mvp_audit_block(state, stage.stage_number)
    if mvp_audit_block:
        parts.extend([mvp_audit_block, f""])

    # Metrics-hash pin (trading stage 4 only): the verdict must cite the exact
    # metrics stage 3 validated — the evaluator rejects a verdict without this
    # literal line or with a hash that no longer matches the file.
    if state.pipeline_type == "trading" and stage.stage_number == 4 and \
            getattr(state, "pinned_full_metrics_hash", None):
        parts.extend([
            f"## Validated Metrics Pin",
            f"Stage 3 validated backtest/full/metrics.json with content hash "
            f"`{state.pinned_full_metrics_hash}`. Your verdict.md MUST include this "
            f"exact line:",
            f"",
            f"Metrics Hash: {state.pinned_full_metrics_hash}",
            f"",
            f"Do NOT modify backtest/full/metrics.json — if the numbers changed, "
            f"stage 3 must re-validate first.",
            f"",
        ])

    # Notes from previous attempts (last 3 only — full history in state.json)
    rendered_notes = _last_attempt_notes(stage.notes)
    if rendered_notes:
        parts.extend([
            f"## Notes from Previous Attempts",
            f"",
            rendered_notes,
            f"",
        ])

    return "\n".join(parts)


def _format_hms(seconds: int) -> str:
    """Format a positive-or-zero duration in seconds as e.g. '6h 18m'."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _build_time_budget_block(stage) -> Optional[str]:
    """Return a `## Time Budget` markdown block, or None if the stage has no budget."""
    if not stage.budget_seconds:
        return None

    total = int(stage.budget_seconds)
    started = stage.first_started_at
    if started:
        elapsed = int((datetime.now() - started).total_seconds())
        started_str = started.strftime("%Y-%m-%d %H:%M:%S")
    else:
        elapsed = 0
        started_str = "(this is the first run — will be set now)"
    remaining = max(0, total - elapsed)

    return (
        "## Time Budget\n"
        f"- Total budget: {_format_hms(total)} ({total}s)\n"
        f"- First started: {started_str}\n"
        f"- Elapsed so far: {_format_hms(elapsed)}\n"
        f"- Remaining: {_format_hms(remaining)}\n"
        f"- Part-completions used: {stage.part_completion_count} of {stage.max_part_completions}\n"
        "\n"
        "This budget is cumulative across many part-completion runs, not a single subprocess "
        "call. Each `exec` call still has a hard 600s cap. Size each chunk to fit under that "
        "with margin, checkpoint progress to disk, and end your response with \"part-completion\" "
        "when you've spent most of this run's tool-call budget. Do NOT try to finish the entire "
        "stage in one run."
    )


_MVP_AUDIT_SKIP_DIRS = {".pytest_cache", "__pycache__", ".venv", "venv", "node_modules", ".git"}


def _build_mvp_audit_block(state: PipelineState, stage_number: int) -> Optional[str]:
    """For the startup review stage, inject a real filesystem listing of mvp/
    so the review reports on what actually exists instead of inferring from
    the README. Each row shows path + size + non-empty line count for code
    files so the reviewer has ground truth for 'how much code got written'.
    """
    if state.pipeline_type != "startup" or stage_number != 6:
        return None
    mvp_dir = os.path.join(DATA_DIR, "projects", state.project_name, "mvp")
    if not os.path.isdir(mvp_dir):
        return (
            "## MVP Filesystem Audit\n\n"
            "`mvp/` directory does not exist — the MVP stage produced no code."
        )
    rows: list = []
    code_exts = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb")
    total_files = 0
    total_code_lines = 0
    for root, dirs, files in os.walk(mvp_dir):
        dirs[:] = sorted(d for d in dirs if d not in _MVP_AUDIT_SKIP_DIRS)
        rel = os.path.relpath(root, mvp_dir)
        prefix = "" if rel == "." else f"{rel}/"
        for f in sorted(files):
            full = os.path.join(root, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            total_files += 1
            if f.endswith(code_exts):
                try:
                    with open(full) as fp:
                        non_empty = sum(1 for line in fp if line.strip())
                    total_code_lines += non_empty
                    rows.append(f"{prefix}{f} — {size} bytes, {non_empty} non-empty lines")
                except Exception:
                    rows.append(f"{prefix}{f} — {size} bytes")
            else:
                rows.append(f"{prefix}{f} — {size} bytes")
        if len(rows) > 200:
            rows.append("... (truncated)")
            break
    if not rows:
        return "## MVP Filesystem Audit\n\n`mvp/` is empty — no code was produced."
    summary = (
        f"Total files in mvp/: {total_files}. "
        f"Total non-empty lines across code files: {total_code_lines}."
    )
    listing = "\n".join(rows)
    return (
        "## MVP Filesystem Audit\n\n"
        "This is a **ground-truth listing** of the MVP directory. The review must "
        "reflect these real numbers, not infer from the README.\n\n"
        f"{summary}\n\n"
        f"```\n{listing}\n```"
    )


def _build_loop_state_block(state: PipelineState, stage_number: int) -> Optional[str]:
    """For trading stage 2, inject the research-loop state into the prompt.

    Renders `next_step`, `current_phase`, `oos_cutoff_date`, the full
    `pilot_history` (compact), ALL `hypothesis_notes` (bounded to 3), and the
    **last 3** `run_notes`. Older run_notes are intentionally dropped so the
    prompt doesn't grow unbounded as iterations accumulate.
    """
    if state.pipeline_type != "trading" or stage_number != 2:
        return None
    path = os.path.join(
        DATA_DIR, "projects", state.project_name, "strategy", "loop_state.json",
    )
    if not os.path.exists(path):
        return (
            "## Loop State\n\n"
            "`strategy/loop_state.json` does not exist yet — this is the first run. "
            "Execute the `init` step: read `research/data-landscape.md`, write "
            "`strategy/hypothesis_v1.md` (with `required_data_types` declared), "
            "`strategy/universe_v1.json`, and initialize `strategy/loop_state.json` "
            "with `oos_cutoff_date` frozen."
        )
    # Strip any non-canonical next_step / current_phase the agent wrote during
    # a prior part-completion run, so the rendered Loop State only ever shows
    # values from the evaluator's state machine.
    from sandbox_agent.pipeline.evaluator import sanitize_loop_state_file
    sanitize_loop_state_file(path)
    try:
        with open(path) as f:
            loop_state = json.load(f)
    except Exception as e:
        return f"## Loop State\n\nCannot parse loop_state.json: {e}"

    lines = ["## Loop State", ""]
    for key in (
        "next_step", "current_phase", "oos_cutoff_date",
        "hypothesis_count", "iteration_within_hypothesis", "total_iterations",
        "last_decision",
    ):
        if key in loop_state:
            lines.append(f"- **{key}**: `{loop_state[key]}`")

    history = loop_state.get("pilot_history") or []
    if history:
        lines.extend(["", "### pilot_history"])
        lines.append("| hyp | iter | sharpe | trades | failure_mode |")
        lines.append("|---|---|---|---|---|")
        for row in history:
            lines.append(
                f"| {row.get('hyp', '?')} | {row.get('iter', '?')} | "
                f"{row.get('sharpe', '?')} | {row.get('trades', '?')} | "
                f"{row.get('failure_mode') or ''} |"
            )

    hyp_notes = loop_state.get("hypothesis_notes") or []
    if hyp_notes:
        lines.extend(["", "### hypothesis_notes (all — carry cross-hypothesis lessons forward)"])
        for n in hyp_notes:
            lines.append(
                f"- **hyp {n.get('hyp', '?')}** (final_sharpe={n.get('final_sharpe', '?')}, "
                f"iterations_spent={n.get('iterations_spent', '?')}): "
                f"{n.get('why_abandoned', '')}"
            )
            for lesson in (n.get("lessons_for_future_hypotheses") or []):
                lines.append(f"    - {lesson}")

    run_notes = loop_state.get("run_notes") or []
    if run_notes:
        last_three = run_notes[-3:]
        lines.extend(["", "### run_notes (last 3)"])
        for n in last_three:
            lines.append(
                f"- **{n.get('id', '?')}** (step={n.get('step', '?')}, "
                f"hyp={n.get('hyp', '?')}, iter={n.get('iter', '?')})"
            )
            for k in ("what_i_did", "what_worked", "what_seemed_off", "suggested_next"):
                v = n.get(k)
                if v:
                    lines.append(f"    - **{k}**: {v}")

    return "\n".join(lines)


def _check_verdict_skip(state: PipelineState, stage_number: int) -> Optional[str]:
    """If the verdict is `reject`, mark the pipeline completed_rejected and
    return a short skip-result string. Otherwise returns None (proceed normally)."""
    verdict_path = os.path.join(
        DATA_DIR, "projects", state.project_name, "pipeline", "verdict.md",
    )
    if not os.path.exists(verdict_path):
        return None
    try:
        with open(verdict_path) as f:
            text = f.read().lower()
    except Exception:
        return None
    idx = text.find("final recommendation")
    if idx < 0:
        return None
    tail = text[idx:idx + 400]
    if "reject" not in tail:
        return None
    stage = state.stages[stage_number]
    stage.status = "completed"
    stage.acceptance_result = "skipped: verdict = reject"
    state.status = "completed_rejected"
    save_state(state)
    logger.info(
        f"Pipeline {state.project_name}: stage {stage_number} skipped "
        f"(verdict = reject); status=completed_rejected"
    )
    return f"Stage {stage_number} skipped: verdict = reject"


def _detect_part_completion(state: PipelineState, stage_number: int, result_text: str) -> bool:
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

    # Startup MVP stage heuristic — partial build if code exists but README/tests don't.
    # Trading pipelines rely on textual signals only.
    if state.pipeline_type == "startup" and stage_number == 5:
        project_dir = os.path.join(DATA_DIR, "projects", state.project_name)
        mvp_dir = os.path.join(project_dir, "mvp")
        if os.path.isdir(mvp_dir):
            files = os.listdir(mvp_dir)
            has_readme = "README.md" in files
            has_code = any(f.endswith((".py", ".js", ".ts", ".html")) for f in files)
            has_tests = os.path.isdir(os.path.join(mvp_dir, "tests"))
            if has_code and not (has_readme and has_tests):
                return True

    return False
