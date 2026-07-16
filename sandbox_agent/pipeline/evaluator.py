"""Stage acceptance evaluator — programmatic checks + LLM quality evaluation."""

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qwen_agent.llm.schema import Message

from sandbox_agent.config import DATA_DIR
from sandbox_agent.pipeline.orchestrator import (
    get_acceptance_path,
    get_stages,
    load_state,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured decision contract (trading stage 2 only).
# ---------------------------------------------------------------------------

DecisionType = str  # One of: converge | needs_more_data | iterate | infeasible_data
                    # | insufficient_signal | dead_end | terminate | reject
                    # | insane_metrics (pilot metrics failed consistency sanity)


@dataclass
class StageDecision:
    """Structured acceptance result for the trading research loop.

    Other stages continue to use the `(passed, feedback)` tuple contract;
    `evaluate_stage` flattens this back into a tuple with a decision marker
    embedded in the feedback so old call-sites keep working.
    """

    type: DecisionType
    passed: bool
    feedback: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    next_step: Optional[str] = None
    next_phase: Optional[str] = None

    def to_feedback(self) -> str:
        tag = f"[decision={self.type}]"
        if self.next_step:
            tag += f" [next_step={self.next_step}]"
        if self.next_phase:
            tag += f" [next_phase={self.next_phase}]"
        return f"{tag} {self.feedback}".strip()


def evaluate_stage(project_name: str, stage_number: int, system_message: str) -> Tuple[bool, str]:
    """Evaluate a stage's output. Returns (passed, feedback).

    1. Programmatic checks: files exist, non-empty, required sections present
    2. LLM quality check: read output and judge quality

    For trading stage 2 (the research loop), delegates to
    `evaluate_trading_stage_2_decision` and flattens the structured decision
    back into a `(passed, feedback)` tuple. Callers that want the full
    decision should invoke that function directly.
    """
    state = load_state(project_name)
    if not state:
        return False, f"Pipeline state not found for {project_name}"

    # Trading stage 2: structured decision.
    if state.pipeline_type == "trading" and stage_number == 2:
        decision = evaluate_trading_stage_2_decision(project_name)
        return decision.passed, decision.to_feedback()

    stage_defn = get_stages(state.pipeline_type)[stage_number]
    project_dir = os.path.join(DATA_DIR, "projects", project_name)

    # Step 1: Programmatic checks
    prog_passed, prog_feedback = _programmatic_checks(project_dir, stage_defn)
    if not prog_passed:
        return False, f"Programmatic check failed: {prog_feedback}"

    # Step 1b: Stage-specific programmatic checks (short-circuit before LLM eval
    # so the LLM-exception fallback in _llm_evaluation can't silently pass a
    # broken stage).
    stage_passed, stage_feedback = _stage_specific_checks(
        project_dir, state.pipeline_type, stage_number,
    )
    if not stage_passed:
        return False, f"Programmatic check failed: {stage_feedback}"

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

        # Check required sections — only for markdown artifacts. JSON/py outputs
        # won't contain human-readable section headings.
        if not artifact_path.lower().endswith(".md"):
            continue
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

    # For the startup MVP stage the declared artifact is only README.md, so
    # inject a real listing of mvp/ so the judge sees what code actually exists.
    if pipeline_type == "startup" and stage_number == 5:
        listing = _mvp_fs_listing(os.path.join(project_dir, "mvp"))
        artifact_texts.append(
            "### mvp/ contents (actual filesystem — ground truth)\n\n"
            "Use this to verify the README's claims about code files. If the "
            "README describes files that do not appear here, FAIL the stage.\n\n"
            f"```\n{listing}\n```"
        )

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


def _stage_specific_checks(
    project_dir: str, pipeline_type: str, stage_number: int,
) -> Tuple[bool, str]:
    """Dispatch to stage-specific programmatic checks. Returns (True, "") when
    the stage has no specific check or all checks pass.

    Stage-2 trading has its own structured-decision path
    (`evaluate_trading_stage_2_decision`) and does not route through here."""
    if pipeline_type == "trading":
        if stage_number == 3:
            acct_passed, acct_msg = _check_uses_vetted_accounting(project_dir)
            if not acct_passed:
                return False, acct_msg
            passed, msg = _check_full_validation_metrics(project_dir)
            if passed:
                # Contract: stamp the metrics file (schema_version, hash,
                # strategy_version) and pin the hash into pipeline state.
                # Stage 4's verdict must cite this exact hash — see
                # _check_verdict_matches_metrics.
                _stamp_and_pin_full_metrics(project_dir)
            return passed, msg
        if stage_number == 4:
            passed, msg = _check_verdict_matches_metrics(project_dir)
            if passed:
                # pipeline/metrics.json is evaluator-written (agents used to
                # write it with contradictory content — gates_passed:0 next to
                # a PROMOTE). Written only after the verdict gate passes.
                _write_pipeline_metrics_summary(project_dir)
            return passed, msg
        if stage_number == 5:
            return _check_paper_deploy_compiles(project_dir)
    if pipeline_type == "startup":
        if stage_number == 5:
            return _check_mvp_buildable(project_dir)
    return True, ""


# Matches any signed int or decimal, optionally trailed by %. Captures "0", "-0.5",
# "14.2%", "-18.4%", "87".
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?%?")

_PLACEHOLDER_TOKENS = (
    "TBD", "N/A", "X.X", "X.XX", "???", "Pending",
    "To be computed", "Coming soon", "<value>", "to be calculated",
)

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def _check_backtest_placeholders(project_dir: str) -> Tuple[bool, str]:
    """For trading stage 4: require real numbers under each metric heading and
    reject placeholder tokens. Runs before the LLM eval so flaky LLM behavior
    can't accept a stubbed results.md."""
    path = os.path.join(project_dir, "backtest", "results.md")
    if not os.path.exists(path):
        # Already flagged by the generic check — shouldn't reach here.
        return False, f"Missing artifact: backtest/results.md"

    try:
        with open(path) as f:
            text = f.read()
    except Exception as e:
        return False, f"Cannot read backtest/results.md: {e}"

    sections = _split_md_sections(text)

    for metric in ("sharpe", "max drawdown", "cagr", "win rate"):
        body = _find_section_body(sections, metric)
        if body is None:
            return False, f"Missing '{metric}' heading in backtest/results.md"
        # Placeholder tokens come first: "TBD" is a more actionable error than
        # "no numeric value" when both fire on the same section.
        lower_body = body.lower()
        for token in _PLACEHOLDER_TOKENS:
            if token.lower() in lower_body:
                return False, (
                    f"'{metric}' section contains placeholder '{token}' in "
                    f"backtest/results.md — run the backtest and paste real numbers"
                )
        if not _NUMERIC_RE.search(body):
            return False, f"'{metric}' section has no numeric value in backtest/results.md"

    return True, ""


def _split_md_sections(text: str):
    """Split markdown into a list of (heading, body) pairs. Lines before the
    first heading are dropped. Heading is stored with its trimmed raw text
    (without the leading `#`s); body is the text between this heading and the
    next (or EOF)."""
    lines = text.splitlines()
    sections = []  # list of (heading_text, [body_lines])
    current = None
    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            if current is not None:
                sections.append((current[0], "\n".join(current[1])))
            current = (m.group(1), [])
        else:
            if current is not None:
                current[1].append(line)
    if current is not None:
        sections.append((current[0], "\n".join(current[1])))
    return sections


def _find_section_body(sections, metric_keyword: str):
    """Return the body of the first section whose lowercased heading contains
    `metric_keyword` (already lowercased). Returns None when no match."""
    for heading, body in sections:
        if metric_keyword in heading.lower():
            return body
    return None


# Phrases in backtest/results.md that indicate the agent backtested on
# fake/synthetic signals rather than running the real pipeline. These are
# self-admissions — if the agent writes them, we trust the confession and
# reject the stage. Added after a PREM14A run passed acceptance with
# "Sharpe 2.90" from a hash-generated signal that ignored the real LLM
# classifications the pipeline was supposed to use.
_MOCK_SIGNAL_PHRASES = (
    "hash-based simulation",
    "hash-based signal",
    "hash based simulation",
    "hash based signal",
    "deterministic hash",
    "mimics llm output",
    "mimics llm analysis",
    "mock signal",
    "mock signals",
    "simulated signal",
    "simulated signals",
    "synthetic signal",
    "synthetic signals",
    "fake signal",
    "fake signals",
    "placeholder signal",
    "placeholder signals",
)


def _check_backtest_uses_real_signals(project_dir: str) -> Tuple[bool, str]:
    """For trading stage 4: reject `backtest/results.md` when it self-admits
    to running on fake signals. The earlier placeholder check only rejects
    obvious `TBD`/`N/A` tokens — it can't tell whether convincing-looking
    numbers came from a real LLM-on-filings pipeline or from a hash function
    masquerading as one. This check scans for phrases the agent writes when
    it took the mock path (e.g. 'Signal Generation: Deterministic hash-based
    simulation'), and blocks advancement with a specific pointer at the
    offending phrase."""
    path = os.path.join(project_dir, "backtest", "results.md")
    if not os.path.exists(path):
        return False, "Missing artifact: backtest/results.md"
    try:
        with open(path) as f:
            text = f.read()
    except Exception as e:
        return False, f"Cannot read backtest/results.md: {e}"

    lower = text.lower()
    for phrase in _MOCK_SIGNAL_PHRASES:
        if phrase in lower:
            return False, (
                f"backtest/results.md self-identifies as using fake signals "
                f"('{phrase}') — rerun the backtest on real LLM classifications "
                f"from the filings pipeline, not a synthetic stand-in"
            )

    return True, ""


_MVP_MIN_NONEMPTY_LINES = 20
_MVP_MIN_CODE_FILES = 1
_MVP_SKIP_DIRS = {".pytest_cache", "__pycache__", ".venv", "venv", "node_modules", ".git"}
_MVP_DEP_MANIFESTS = (
    "requirements.txt", "pyproject.toml", "package.json",
    "Cargo.toml", "go.mod", "Gemfile",
)


def _mvp_code_files(mvp_dir: str, ext: str) -> List[str]:
    """Walk mvp/ skipping caches/vendor dirs. Return absolute file paths ending in ext."""
    out: List[str] = []
    for root, dirs, files in os.walk(mvp_dir):
        dirs[:] = [d for d in dirs if d not in _MVP_SKIP_DIRS]
        for f in files:
            if f.endswith(ext):
                out.append(os.path.join(root, f))
    return out


def _mvp_fs_listing(mvp_dir: str) -> str:
    """Render a compact tree of mvp/ for injection into the LLM eval prompt so
    the judge sees the real filesystem instead of trusting the README's claims."""
    if not os.path.isdir(mvp_dir):
        return "(mvp/ directory does not exist)"
    rows: List[str] = []
    for root, dirs, files in os.walk(mvp_dir):
        dirs[:] = sorted(d for d in dirs if d not in _MVP_SKIP_DIRS)
        rel = os.path.relpath(root, mvp_dir)
        prefix = "" if rel == "." else f"{rel}/"
        for f in sorted(files):
            full = os.path.join(root, f)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            rows.append(f"{prefix}{f} ({size} bytes)")
        if len(rows) > 200:
            rows.append("... (truncated)")
            break
    return "\n".join(rows) if rows else "(mvp/ is empty)"


def _check_mvp_buildable(project_dir: str) -> Tuple[bool, str]:
    """Stage 5 startup: verify mvp/ contains actually-runnable code, not just a
    well-written README describing phantom software. Fails if the MVP directory
    has no code files, if the code doesn't compile, or if there's no dependency
    manifest — catches the moon-ai failure mode where the LLM judge accepted a
    comprehensive README while the code never got written (or got written but
    broken)."""
    mvp_dir = os.path.join(project_dir, "mvp")
    if not os.path.isdir(mvp_dir):
        return False, "mvp/ directory does not exist — MVP stage produced no code"

    py_files = _mvp_code_files(mvp_dir, ".py")
    js_files = _mvp_code_files(mvp_dir, ".js") + _mvp_code_files(mvp_dir, ".ts")
    code_files = py_files + js_files

    non_trivial: List[str] = []
    for path in code_files:
        try:
            with open(path) as f:
                non_empty = sum(1 for line in f if line.strip())
        except Exception:
            continue
        if non_empty >= _MVP_MIN_NONEMPTY_LINES:
            non_trivial.append(path)

    if len(non_trivial) < _MVP_MIN_CODE_FILES:
        return False, (
            f"mvp/ contains no non-trivial code files "
            f"(need ≥{_MVP_MIN_CODE_FILES} with ≥{_MVP_MIN_NONEMPTY_LINES} non-empty lines). "
            f"Found {len(code_files)} code files total, {len(non_trivial)} with enough content. "
            f"A README describing code that wasn't written does not pass."
        )

    for path in py_files:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", path],
                capture_output=True, text=True, timeout=20,
            )
        except subprocess.TimeoutExpired:
            return False, f"{os.path.relpath(path, project_dir)}: py_compile timed out (20s)"
        except Exception as e:
            return False, f"{os.path.relpath(path, project_dir)}: py_compile error: {e}"
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            rel = os.path.relpath(path, project_dir)
            return False, f"{rel} does not compile:\n{stderr[:500]}"

    has_manifest = any(
        os.path.exists(os.path.join(mvp_dir, m)) for m in _MVP_DEP_MANIFESTS
    )
    if not has_manifest and py_files:
        return False, (
            f"mvp/ has Python code but no dependency manifest "
            f"(expected one of: {', '.join(_MVP_DEP_MANIFESTS)}). "
            f"A reader cannot install and run the MVP."
        )

    return True, ""


def _check_paper_deploy_compiles(project_dir: str) -> Tuple[bool, str]:
    """For trading stage 5: paper/deploy.py must parse as Python. Enforce
    programmatically via `python -m py_compile` before the LLM eval, so a
    syntax error can't slip through when the evaluator LLM is flaky."""
    deploy_path = os.path.join(project_dir, "paper", "deploy.py")
    if not os.path.exists(deploy_path):
        return False, "Missing artifact: paper/deploy.py"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", deploy_path],
            capture_output=True, text=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        return False, "paper/deploy.py: py_compile timed out (20s)"
    except Exception as e:
        return False, f"paper/deploy.py: py_compile error: {e}"
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        return False, f"paper/deploy.py does not compile:\n{stderr[:500]}"
    return True, ""


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


# ---------------------------------------------------------------------------
# Trading stage 2 (research loop): structured decision + gates.
# ---------------------------------------------------------------------------

_CONVERGE_SHARPE = 0.8
_CONVERGE_TRADES = 30
_PLATEAU_DELTA = 0.15
_NEEDS_MORE_DATA_SHARPE = 1.2
_INSUFFICIENT_TRADES = 10
_INSUFFICIENT_SIGNAL_ITERS = 2
_MAX_CACHE_BUCKET_SHARE = 0.6
_MIN_CACHE_DISTINCT = 3
_MIN_CACHE_COVERAGE = 0.8
_INFEASIBLE_COVERAGE = 0.5

_HYPOTHESIS_CEILING = 3
_TOTAL_ITER_CEILING = 24

# Canonical state-machine values. The evaluator is the only writer of these
# fields; any non-canonical value in loop_state.json was authored by the agent
# (LLM narrative drift) and MUST be discarded — both on evaluator entry and
# whenever the prompt builder renders Loop State, so the agent never sees a
# value the state machine doesn't recognize.
CANONICAL_NEXT_STEPS = frozenset({
    "init", "fetch_data", "extract_features", "run_pilot_backtest",
    "revise_hypothesis", "revise_data_processing", "revise_strategy_code",
    "extend_pilot_window",
})
CANONICAL_NEXT_PHASES = frozenset({
    "pilot", "revise_hypothesis", "revise_data_processing",
    "revise_strategy_code", "extend_pilot_window",
})
# Sentinel written by the evaluator on converge/terminate so a subsequent
# prompt render shows the agent "no work" rather than a stale step.
ADVANCE_SENTINEL = "_advance"


def sanitize_loop_state_file(loop_state_path: str) -> bool:
    """Strip non-canonical next_step / current_phase from loop_state.json on disk.

    Returns True if the file was modified. Safe to call whether or not the file
    exists; missing / unreadable files are a no-op. Called from:
      - evaluator (start of stage-2 decision) — so gates run against a clean state
      - stage_runner._build_loop_state_block — so the prompt never shows drift

    Agent-authored narrative like `next_step=awaiting_trades` is the exact kind
    of value we discard here; ADVANCE_SENTINEL is preserved because *we* wrote it.
    """
    if not os.path.exists(loop_state_path):
        return False
    try:
        with open(loop_state_path) as f:
            loop_state = json.load(f)
    except Exception:
        return False
    changed = False
    ns = loop_state.get("next_step")
    if ns and ns != ADVANCE_SENTINEL and ns not in CANONICAL_NEXT_STEPS:
        logger.info(f"sanitize_loop_state: discarding non-canonical next_step={ns!r}")
        loop_state["next_step"] = None
        changed = True
    cp = loop_state.get("current_phase")
    if cp and cp not in CANONICAL_NEXT_PHASES:
        logger.info(f"sanitize_loop_state: discarding non-canonical current_phase={cp!r}")
        loop_state["current_phase"] = None
        changed = True

    # Quarantine pilot_history rows with insane metrics (consistency-based —
    # see metrics_sanity; legitimate short-strategy blowups are NOT flagged).
    # Rejected rows move to pilot_history_rejected with their violations
    # attached, so _decide_from_pilot_history never bases a decision on
    # fabricated numbers (observed: +1,399,591% returns entering history).
    from sandbox_agent.pipeline.metrics_sanity import validate_metrics
    history = loop_state.get("pilot_history") or []
    kept, rejected = [], []
    for row in history:
        violations = validate_metrics(row) if isinstance(row, dict) else ["row is not a dict"]
        if violations:
            row = dict(row) if isinstance(row, dict) else {"raw": row}
            row["_violations"] = violations
            rejected.append(row)
        else:
            kept.append(row)
    if rejected:
        logger.warning(
            f"sanitize_loop_state: quarantined {len(rejected)} pilot_history row(s) "
            f"with insane metrics")
        loop_state["pilot_history"] = kept
        loop_state.setdefault("pilot_history_rejected", []).extend(rejected)
        changed = True

    if changed:
        try:
            with open(loop_state_path, "w") as f:
                json.dump(loop_state, f, indent=2)
        except Exception as e:
            logger.warning(f"sanitize_loop_state: cannot write {loop_state_path}: {e}")
            return False
    return changed


def evaluate_trading_stage_2_decision(project_name: str) -> StageDecision:
    """Run all stage-2 programmatic gates and compute the structured decision.

    The decision table is the source of truth for `next_step` / `next_phase`;
    the LLM's narrative note is advisory only. On `converge` (pass), the stage
    advances. On any other decision (`iterate`, `revise_*`, `dead_end`,
    `terminate`, or gate failure → `infeasible_data`), the stage re-runs
    with `next_step`/`next_phase` persisted back into `loop_state.json`.
    """
    project_dir = os.path.join(DATA_DIR, "projects", project_name)
    loop_state_path = os.path.join(project_dir, "strategy", "loop_state.json")

    if not os.path.exists(loop_state_path):
        return StageDecision(
            type="iterate",
            passed=False,
            feedback=(
                "strategy/loop_state.json does not exist yet. On the first run, "
                "perform the `init` step: write hypothesis_v1.md, universe_v1.json, "
                "and a loop_state.json with oos_cutoff_date frozen."
            ),
            next_step="init",
            next_phase="pilot",
        )

    # Discard any non-canonical next_step / current_phase the agent wrote
    # during part-completion runs. The evaluator is the only authority.
    sanitize_loop_state_file(loop_state_path)

    try:
        with open(loop_state_path) as f:
            loop_state = json.load(f)
    except Exception as e:
        return StageDecision(
            type="iterate",
            passed=False,
            feedback=f"Cannot parse strategy/loop_state.json: {e}",
            next_step="init",
        )

    # Gate 1: hypothesis declares feasible data types.
    feas_passed, feas_msg = _check_hypothesis_data_feasibility(project_dir, loop_state)
    if not feas_passed:
        return _write_decision(
            loop_state_path, loop_state,
            StageDecision(
                type="infeasible_data",
                passed=False,
                feedback=feas_msg,
                next_step="revise_hypothesis",
                next_phase="revise_hypothesis",
            ),
        )

    # Gate 2: OOS cutoff is frozen and pilot data respects it.
    oos_passed, oos_msg = _check_pilot_avoids_oos(project_dir, loop_state)
    if not oos_passed:
        return _write_decision(
            loop_state_path, loop_state,
            StageDecision(
                type="infeasible_data",
                passed=False,
                feedback=oos_msg,
                next_step="revise_strategy_code",
                next_phase="revise_strategy_code",
            ),
        )

    # Gate 3: strategy code references the llm_cache (positive lineage).
    lineage_passed, lineage_msg = _check_strategy_reads_llm_cache(project_dir)
    if not lineage_passed:
        return _write_decision(
            loop_state_path, loop_state,
            StageDecision(
                type="infeasible_data",
                passed=False,
                feedback=lineage_msg,
                next_step="revise_strategy_code",
                next_phase="revise_strategy_code",
            ),
        )

    # Gate 3b: vetted accounting lineage — backtests must use the Ledger/
    # compute_metrics module, not hand-rolled equity math.
    acct_passed, acct_msg = _check_uses_vetted_accounting(project_dir)
    if not acct_passed:
        return _write_decision(
            loop_state_path, loop_state,
            StageDecision(
                type="infeasible_data",
                passed=False,
                feedback=acct_msg,
                next_step="revise_strategy_code",
                next_phase="revise_strategy_code",
            ),
        )

    # Gate 4: llm_cache variance (not a degenerate single-class classifier).
    var_passed, var_msg = _check_llm_cache_variance(project_dir)
    if not var_passed:
        return _write_decision(
            loop_state_path, loop_state,
            StageDecision(
                type="infeasible_data",
                passed=False,
                feedback=var_msg,
                next_step="revise_data_processing",
                next_phase="revise_data_processing",
            ),
        )

    # Gate 5: cache coverage over declared universe.
    cov_passed, cov_msg, cov_ratio = _check_cache_coverage(project_dir, loop_state)
    if not cov_passed:
        # Coverage < 50% is infeasible; 50-80% is iterate/extend_pilot_window.
        if cov_ratio < _INFEASIBLE_COVERAGE:
            return _write_decision(
                loop_state_path, loop_state,
                StageDecision(
                    type="infeasible_data",
                    passed=False,
                    feedback=cov_msg,
                    next_step="revise_data_processing",
                    next_phase="revise_data_processing",
                ),
            )
        return _write_decision(
            loop_state_path, loop_state,
            StageDecision(
                type="needs_more_data",
                passed=False,
                feedback=cov_msg,
                next_step="extend_pilot_window",
                next_phase="extend_pilot_window",
            ),
        )

    # Gate 6: latest pilot results.md does not self-admit to fake signals.
    mock_passed, mock_msg = _check_pilot_uses_real_signals(project_dir, loop_state)
    if not mock_passed:
        return _write_decision(
            loop_state_path, loop_state,
            StageDecision(
                type="infeasible_data",
                passed=False,
                feedback=mock_msg,
                next_step="revise_strategy_code",
                next_phase="revise_strategy_code",
            ),
        )

    # Gate 7: numeric sanity (consistency-based; short blowups allowed).
    # (a) If the sanitizer quarantined the NEWEST pilot row, the just-run pilot
    # produced insane metrics — tell the agent to fix its ledger, not iterate.
    rejected = loop_state.get("pilot_history_rejected") or []
    kept = loop_state.get("pilot_history") or []
    if rejected:
        def _ver(row):
            try:
                return int(row.get("version", 0) or 0)
            except (TypeError, ValueError):
                return 0
        newest_rejected = max(_ver(r) for r in rejected)
        newest_kept = max((_ver(r) for r in kept), default=0)
        if newest_rejected > newest_kept:
            bad = max(rejected, key=_ver)
            return _write_decision(
                loop_state_path, loop_state,
                StageDecision(
                    type="insane_metrics",
                    passed=False,
                    feedback=(
                        "The latest pilot backtest produced internally inconsistent "
                        f"metrics and was quarantined: {'; '.join(bad.get('_violations', []))}. "
                        "This is almost always broken accounting in the backtest script "
                        "(e.g. trading past a blowup, mis-computed equity). Fix the "
                        "ledger/accounting code and re-run the pilot. The quarantined row "
                        "is in pilot_history_rejected for reference."
                    ),
                    next_step="revise_strategy_code",
                    next_phase="revise_strategy_code",
                ),
            )
    # (b) metrics_latest.json — same sanity bar as the history rows.
    from sandbox_agent.pipeline.metrics_sanity import validate_metrics as _vm
    metrics_latest = os.path.join(project_dir, "backtest", "pilot", "metrics_latest.json")
    if os.path.exists(metrics_latest):
        try:
            with open(metrics_latest) as f:
                latest = json.load(f)
            violations = _vm(latest)
        except Exception as e:
            violations = [f"cannot parse metrics_latest.json: {e}"]
        if violations:
            return _write_decision(
                loop_state_path, loop_state,
                StageDecision(
                    type="insane_metrics",
                    passed=False,
                    feedback=(
                        "backtest/pilot/metrics_latest.json fails numeric sanity: "
                        f"{'; '.join(violations)}. Fix the backtest accounting and re-run."
                    ),
                    next_step="revise_strategy_code",
                    next_phase="revise_strategy_code",
                ),
            )

    # All gates passed — now consult pilot_history for the actual decision.
    return _write_decision(
        loop_state_path, loop_state,
        _decide_from_pilot_history(loop_state),
    )


def _decide_from_pilot_history(loop_state: dict) -> StageDecision:
    """Compute the structured decision from `pilot_history` alone. Assumes all
    programmatic gates have already passed. Never mutates `loop_state`."""
    history: List[dict] = loop_state.get("pilot_history", []) or []
    if not history:
        return StageDecision(
            type="iterate",
            passed=False,
            feedback="pilot_history is empty — run a pilot backtest",
            next_step="run_pilot_backtest",
            next_phase="pilot",
        )

    last = history[-1]
    last_sharpe = float(last.get("sharpe", 0.0))
    last_trades = int(last.get("trades", 0))
    total_iters = int(loop_state.get("total_iterations", len(history)))
    hypothesis_count = int(loop_state.get("hypothesis_count", 1))

    # Plateau check: last 2 iters within this hypothesis.
    current_hyp = int(last.get("hyp", hypothesis_count))
    within_hyp = [h for h in history if int(h.get("hyp", 0)) == current_hyp]
    plateau = False
    if len(within_hyp) >= 2:
        deltas = [
            abs(float(within_hyp[-1].get("sharpe", 0.0)) - float(within_hyp[-2].get("sharpe", 0.0))),
        ]
        if len(within_hyp) >= 3:
            deltas.append(
                abs(float(within_hyp[-2].get("sharpe", 0.0)) - float(within_hyp[-3].get("sharpe", 0.0)))
            )
        plateau = all(d < _PLATEAU_DELTA for d in deltas[:2]) and len(deltas) >= 2

    # Terminate: exhausted budget in structural ways.
    if hypothesis_count >= _HYPOTHESIS_CEILING and (
        last_sharpe < _CONVERGE_SHARPE or last_trades < _CONVERGE_TRADES
    ):
        return StageDecision(
            type="terminate",
            passed=True,  # advance to verdict stage, which will write 'reject'
            feedback=(
                f"Hypothesis ceiling ({_HYPOTHESIS_CEILING}) reached without convergence. "
                f"Last: sharpe={last_sharpe:.2f}, trades={last_trades}. "
                f"Advancing to full validation / verdict for rejection."
            ),
            metrics={"last_sharpe": last_sharpe, "last_trades": last_trades},
        )
    if total_iters > _TOTAL_ITER_CEILING:
        return StageDecision(
            type="terminate",
            passed=True,
            feedback=(
                f"Total-iteration ceiling ({_TOTAL_ITER_CEILING}) exceeded. "
                f"Advancing to full validation / verdict."
            ),
            metrics={"last_sharpe": last_sharpe, "last_trades": last_trades},
        )

    # Converge: plateaued at acceptable performance.
    if (
        plateau
        and last_sharpe >= _CONVERGE_SHARPE
        and last_trades >= _CONVERGE_TRADES
    ):
        return StageDecision(
            type="converge",
            passed=True,
            feedback=(
                f"Converged: sharpe={last_sharpe:.2f} (≥{_CONVERGE_SHARPE}), "
                f"trades={last_trades} (≥{_CONVERGE_TRADES}), plateau confirmed."
            ),
            metrics={"last_sharpe": last_sharpe, "last_trades": last_trades},
        )

    # Needs more data: promising sharpe but trade count/sample is thin.
    if last_sharpe > _NEEDS_MORE_DATA_SHARPE and last_trades < _CONVERGE_TRADES:
        return StageDecision(
            type="needs_more_data",
            passed=False,
            feedback=(
                f"Promising sharpe ({last_sharpe:.2f}) but only {last_trades} trades — "
                f"extend the pilot window before trusting the number."
            ),
            next_step="extend_pilot_window",
            next_phase="extend_pilot_window",
        )

    # Insufficient signal: >=2 recent iters with <10 trades → revise hypothesis.
    thin_iters = [h for h in within_hyp[-_INSUFFICIENT_SIGNAL_ITERS:] if int(h.get("trades", 0)) < _INSUFFICIENT_TRADES]
    if len(thin_iters) >= _INSUFFICIENT_SIGNAL_ITERS:
        return StageDecision(
            type="insufficient_signal",
            passed=False,
            feedback=(
                f"{len(thin_iters)} recent iters with <{_INSUFFICIENT_TRADES} trades — "
                f"hypothesis is too narrow; revise it."
            ),
            next_step="revise_hypothesis",
            next_phase="revise_hypothesis",
        )

    # Dead end: plateau without hitting converge bar → rotate to new hypothesis.
    if plateau:
        return StageDecision(
            type="dead_end",
            passed=False,
            feedback=(
                f"Plateaued at sharpe={last_sharpe:.2f} below converge bar "
                f"({_CONVERGE_SHARPE}). Rotating to a new hypothesis."
            ),
            next_step="revise_hypothesis",
            next_phase="revise_hypothesis",
        )

    # Default: iterate — last run improved things OR we haven't plateaued yet.
    return StageDecision(
        type="iterate",
        passed=False,
        feedback=(
            f"Iterating: sharpe={last_sharpe:.2f}, trades={last_trades}. "
            f"No plateau, no convergence yet."
        ),
        next_step="run_pilot_backtest",
        next_phase="pilot",
    )


def _write_decision(
    loop_state_path: str, loop_state: dict, decision: StageDecision,
) -> StageDecision:
    """Persist next_step / next_phase / last_decision into loop_state.json.

    The evaluator is the single writer of these fields — any value the agent
    wrote during part-completion runs is unconditionally overwritten here, even
    on converge/terminate (where we write ADVANCE_SENTINEL so stale narrative
    doesn't survive the stage boundary)."""
    if decision.next_step:
        if decision.next_step not in CANONICAL_NEXT_STEPS:
            raise AssertionError(
                f"internal: decision.next_step={decision.next_step!r} is not in "
                f"CANONICAL_NEXT_STEPS — evaluator bug, not agent drift"
            )
        loop_state["next_step"] = decision.next_step
    else:
        loop_state["next_step"] = ADVANCE_SENTINEL
    if decision.next_phase:
        if decision.next_phase not in CANONICAL_NEXT_PHASES:
            raise AssertionError(
                f"internal: decision.next_phase={decision.next_phase!r} is not in "
                f"CANONICAL_NEXT_PHASES — evaluator bug, not agent drift"
            )
        loop_state["current_phase"] = decision.next_phase
    else:
        loop_state["current_phase"] = None
    loop_state["last_decision"] = decision.type
    try:
        with open(loop_state_path, "w") as f:
            json.dump(loop_state, f, indent=2)
    except Exception as e:
        logger.warning(f"Cannot write loop_state.json: {e}")
    return decision


# ---- Trading stage-2 gates ----


def _check_hypothesis_data_feasibility(
    project_dir: str, loop_state: dict,
) -> Tuple[bool, str]:
    """Require the current hypothesis_v{N}.md to declare required_data_types,
    and verify each type appears in research/data-landscape.md. Runs fast,
    before any expensive fetch/extract work."""
    hypothesis_count = int(loop_state.get("hypothesis_count", 1))
    hyp_path = os.path.join(
        project_dir, "strategy", f"hypothesis_v{hypothesis_count}.md",
    )
    if not os.path.exists(hyp_path):
        # On the very first run we won't have written this yet — allow
        # through so the agent can run the `init` step. The missing-file
        # case short-circuits above in evaluate_trading_stage_2_decision.
        return True, ""

    try:
        with open(hyp_path) as f:
            hyp_text = f.read()
    except Exception as e:
        return False, f"Cannot read {hyp_path}: {e}"

    required = _extract_required_data_types(hyp_text)
    if not required:
        return False, (
            f"strategy/hypothesis_v{hypothesis_count}.md is missing a "
            f"`required_data_types` declaration. Add a section listing the "
            f"data source names this hypothesis depends on (must match names "
            f"in research/data-landscape.md)."
        )

    landscape_path = os.path.join(project_dir, "research", "data-landscape.md")
    if not os.path.exists(landscape_path):
        return False, "Missing artifact: research/data-landscape.md (stage 1 output)"

    try:
        with open(landscape_path) as f:
            landscape = f.read().lower()
    except Exception as e:
        return False, f"Cannot read research/data-landscape.md: {e}"

    missing = [r for r in required if not _matches_landscape(r, landscape)]
    if missing:
        return False, (
            f"Hypothesis requires data types {missing} but research/data-landscape.md "
            f"does not mention them. Pick hypotheses that can be realized with "
            f"available sources, or broaden the landscape first. Declare types as "
            f"plain names matching the landscape, e.g. "
            f"`required_data_types: [yfinance_ohlcv, SEC 8-K]`."
        )
    return True, ""


def _normalize_data_type(item: str) -> str:
    """Reduce a declared data-type item to its bare name.

    Formatting must never fail a hypothesis (a Sharpe-2.221 strategy was once
    abandoned because its bullets carried backticks + descriptions): take the
    first backtick span if present, else strip markdown emphasis/quotes and cut
    any trailing description. Description separators require surrounding
    spaces (" — ", " - ", ": ") so hyphenated names like "SEC 8-K" survive.
    """
    tick = re.search(r"`([^`]+)`", item)
    if tick:
        return tick.group(1).strip()
    s = item.strip().strip("*_").strip('"').strip("'")
    s = re.split(r"\s+[—–-]\s+|:\s+", s, maxsplit=1)[0]
    return re.sub(r"\s+", " ", s.strip().strip("*_").strip('"').strip("'"))


def _matches_landscape(needle: str, landscape_lower: str) -> bool:
    """Tolerant containment: normalized substring first, then a word-level
    fallback (all significant words present) for punctuation/order variance."""
    norm = re.sub(r"[`*\"']", "", needle.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return False
    clean_landscape = re.sub(r"[`*\"']", "", landscape_lower)
    if norm in clean_landscape:
        return True
    words = [w for w in re.split(r"[^a-z0-9_-]+", norm) if len(w) > 2]
    return bool(words) and all(w in clean_landscape for w in words)


def _extract_required_data_types(hyp_text: str) -> List[str]:
    """Parse `required_data_types` from hypothesis markdown.

    Accepts either a YAML/JSON-style list:
        required_data_types: [SEC 8-K, IEX news feed]
    or a markdown list under a `## Required Data Types` / `## required_data_types`
    heading. Items are normalized (backticks/emphasis/descriptions stripped) —
    see _normalize_data_type.
    """
    inline = re.search(
        r"required_data_types\s*[:=]\s*\[([^\]]+)\]", hyp_text, re.IGNORECASE,
    )
    if inline:
        items = inline.group(1).split(",")
        return [_normalize_data_type(it) for it in items if it.strip()]

    section = re.search(
        r"#+\s*required[_ ]data[_ ]types[^\n]*\n(.+?)(?:\n#+|\Z)",
        hyp_text, re.IGNORECASE | re.DOTALL,
    )
    if section:
        lines = section.group(1).splitlines()
        items = []
        for line in lines:
            m = re.match(r"\s*[-*]\s*(.+?)\s*$", line)
            if m:
                items.append(_normalize_data_type(m.group(1)))
        return [it for it in items if it]
    return []


def _check_pilot_avoids_oos(
    project_dir: str, loop_state: dict,
) -> Tuple[bool, str]:
    """Assert (a) `oos_cutoff_date` has not been rewritten since the previous
    run; (b) no pilot code references `oos/`; (c) every parquet/csv in
    `data/processed/pilot/` has a max-date strictly before the cutoff.

    The cutoff-immutability check compares against the value we saw previously
    (tracked in a side file so a single step that tries to rewrite the cutoff
    in loop_state.json gets caught on the NEXT invocation's acceptance)."""
    cutoff = loop_state.get("oos_cutoff_date")
    if not cutoff:
        # First run: cutoff not yet chosen. Allow through; `init` must write it.
        return True, ""

    # Static-string check on pilot code for `oos/` references.
    pilot_code_dirs = [
        os.path.join(project_dir, "backtest", "pilot"),
        os.path.join(project_dir, "strategy"),
    ]
    offenders = []
    for d in pilot_code_dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(".py"):
                continue
            p = os.path.join(d, name)
            try:
                with open(p) as f:
                    txt = f.read()
            except Exception:
                continue
            if "oos/" in txt or "data/processed/oos" in txt:
                offenders.append(os.path.relpath(p, project_dir))
    if offenders:
        return False, (
            f"Pilot/strategy code references the OOS set: {offenders}. "
            f"The loop stage is forbidden from reading `oos/` — that window "
            f"is reserved for stage 3 (full validation)."
        )

    # Cutoff-stability check.
    shadow = os.path.join(project_dir, "strategy", ".oos_cutoff_shadow")
    if os.path.exists(shadow):
        try:
            prev = open(shadow).read().strip()
        except Exception:
            prev = ""
        if prev and prev != str(cutoff):
            return False, (
                f"oos_cutoff_date changed from {prev!r} to {cutoff!r} — "
                f"the cutoff is frozen on first fetch and must never be rewritten."
            )
    try:
        os.makedirs(os.path.dirname(shadow), exist_ok=True)
        with open(shadow, "w") as f:
            f.write(str(cutoff))
    except Exception:
        pass
    return True, ""


def _check_strategy_reads_llm_cache(project_dir: str) -> Tuple[bool, str]:
    """At least one `backtest/pilot/strategy_v*.py` must reference
    `data/processed/llm_cache/pilot/`. Positive lineage — a strategy that
    doesn't look at the cache can't possibly be using the research."""
    pilot_dir = os.path.join(project_dir, "backtest", "pilot")
    if not os.path.isdir(pilot_dir):
        return False, (
            "No backtest/pilot/ directory — write strategy_v1.py that loads "
            "data/processed/llm_cache/pilot/ classifications and runs a backtest."
        )
    strategy_files = [
        os.path.join(pilot_dir, n)
        for n in os.listdir(pilot_dir)
        if n.startswith("strategy_v") and n.endswith(".py")
    ]
    if not strategy_files:
        return False, (
            "No strategy_v*.py in backtest/pilot/ — write one that reads "
            "data/processed/llm_cache/pilot/."
        )
    for p in strategy_files:
        try:
            with open(p) as f:
                txt = f.read()
        except Exception:
            continue
        if "llm_cache/pilot" in txt or "llm_cache/pilot/" in txt:
            return True, ""
    return False, (
        "No strategy file under backtest/pilot/ references "
        "`data/processed/llm_cache/pilot/`. The strategy must be driven by "
        "the LLM-extracted features, not fabricated signals."
    )


def _check_uses_vetted_accounting(project_dir: str) -> Tuple[bool, str]:
    """Every strategy/backtest script must import
    `sandbox_agent.trading_accounting` (the vetted Ledger + compute_metrics).
    Hand-rolled equity/return/drawdown accounting is how impossible numbers
    (post-blowup trading, -235% drawdowns) entered pilot_history."""
    script_dirs = [
        os.path.join(project_dir, "backtest", "pilot"),
        os.path.join(project_dir, "backtest", "full"),
    ]
    scripts = []
    for d in script_dirs:
        if os.path.isdir(d):
            scripts.extend(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".py"))
    if not scripts:
        return True, ""  # nothing to check yet (other gates demand the scripts)
    offenders = []
    for p in scripts:
        try:
            with open(p) as f:
                txt = f.read()
        except Exception:
            continue
        if "trading_accounting" not in txt:
            offenders.append(os.path.relpath(p, project_dir))
    if offenders:
        return False, (
            "Strategy/backtest script(s) do not use the vetted accounting module: "
            f"{', '.join(offenders[:5])}. All backtests MUST use "
            "`from sandbox_agent.trading_accounting import Ledger, compute_metrics` "
            "for fills, equity, and metrics — hand-rolled accounting fails acceptance."
        )
    return True, ""


def _check_llm_cache_variance(project_dir: str) -> Tuple[bool, str]:
    """Reject if classifier output is degenerate: any one value appears
    > 60% of the time, or < 3 distinct values exist. A monotone classifier
    can't drive a strategy."""
    cache_dir = os.path.join(project_dir, "data", "processed", "llm_cache", "pilot")
    if not os.path.isdir(cache_dir):
        return False, (
            "No data/processed/llm_cache/pilot/ directory — run the "
            "`extract_features` step with llm_batch() and cache results."
        )

    values: List[str] = []
    for name in os.listdir(cache_dir):
        if not name.endswith(".json"):
            continue
        p = os.path.join(cache_dir, name)
        try:
            with open(p) as f:
                entry = json.load(f)
        except Exception:
            continue
        val = _extract_classification(entry)
        if val is not None:
            values.append(str(val).lower().strip())

    if len(values) < _INSUFFICIENT_TRADES:
        return False, (
            f"llm_cache has only {len(values)} parseable entries — too few "
            f"to assess variance. Extract more before running backtests."
        )

    distinct = set(values)
    if len(distinct) < _MIN_CACHE_DISTINCT:
        return False, (
            f"llm_cache classifier output has only {len(distinct)} distinct values "
            f"({sorted(distinct)}) — needs at least {_MIN_CACHE_DISTINCT}. "
            f"Revise the extraction prompt to force meaningful differentiation."
        )

    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    dominant = max(counts.values()) / len(values)
    if dominant > _MAX_CACHE_BUCKET_SHARE:
        top = max(counts.items(), key=lambda kv: kv[1])
        return False, (
            f"llm_cache is degenerate: {top[0]!r} appears in {dominant:.0%} of "
            f"{len(values)} entries (ceiling {_MAX_CACHE_BUCKET_SHARE:.0%}). "
            f"Rewrite the classifier prompt — this extraction isn't discriminating."
        )
    return True, ""


def _extract_classification(entry: Any) -> Optional[str]:
    """Pull a classification string out of a cache entry. Tolerates common
    shapes: {'result': 'pos'}, {'classification': 'neg'}, {'sentiment': ...},
    or a bare string."""
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, dict):
        return None
    for key in ("classification", "result", "sentiment", "label", "category", "output"):
        if key in entry:
            v = entry[key]
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                # Nested shape like {"ticker": ..., "sentiment": "pos"}
                for k2 in ("classification", "sentiment", "label"):
                    if k2 in v and isinstance(v[k2], str):
                        return v[k2]
    return None


def _check_cache_coverage(
    project_dir: str, loop_state: dict,
) -> Tuple[bool, str, float]:
    """Compare cached entries against the declared universe. Returns
    (passed, message, coverage_ratio). coverage_ratio is 0.0 on missing data."""
    hypothesis_count = int(loop_state.get("hypothesis_count", 1))
    universe_path = os.path.join(
        project_dir, "strategy", f"universe_v{hypothesis_count}.json",
    )
    if not os.path.exists(universe_path):
        # Fall back to earlier universes if present.
        for v in range(hypothesis_count - 1, 0, -1):
            alt = os.path.join(project_dir, "strategy", f"universe_v{v}.json")
            if os.path.exists(alt):
                universe_path = alt
                break
        else:
            return True, "", 1.0  # Nothing to check — skip.

    try:
        with open(universe_path) as f:
            universe = json.load(f)
    except Exception as e:
        return False, f"Cannot read {universe_path}: {e}", 0.0

    tickers: List[str] = []
    if isinstance(universe, list):
        tickers = [str(t).upper() for t in universe]
    elif isinstance(universe, dict):
        raw = universe.get("tickers") or universe.get("universe") or []
        tickers = [str(t).upper() for t in raw]
    if not tickers:
        return True, "", 1.0

    cache_dir = os.path.join(project_dir, "data", "processed", "llm_cache", "pilot")
    if not os.path.isdir(cache_dir):
        return False, (
            f"No llm_cache — cannot assess coverage of the {len(tickers)}-ticker universe."
        ), 0.0

    covered: set = set()
    for name in os.listdir(cache_dir):
        if not name.endswith(".json"):
            continue
        p = os.path.join(cache_dir, name)
        try:
            with open(p) as f:
                entry = json.load(f)
        except Exception:
            continue
        tk = _extract_ticker(entry)
        if tk:
            covered.add(tk.upper())

    ratio = len(covered & set(tickers)) / max(1, len(tickers))
    if ratio < _MIN_CACHE_COVERAGE:
        return False, (
            f"llm_cache covers only {len(covered & set(tickers))} of "
            f"{len(tickers)} declared tickers ({ratio:.0%} < {_MIN_CACHE_COVERAGE:.0%})."
        ), ratio
    return True, "", ratio


def _extract_ticker(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        for key in ("ticker", "symbol"):
            if key in entry and isinstance(entry[key], str):
                return entry[key]
    return None


def _check_pilot_uses_real_signals(
    project_dir: str, loop_state: dict,
) -> Tuple[bool, str]:
    """Belt-and-suspenders phrase blacklist against the latest pilot results.md."""
    hypothesis_count = int(loop_state.get("hypothesis_count", 1))
    iter_within = int(loop_state.get("iteration_within_hypothesis", 1))
    candidates = [
        os.path.join(
            project_dir, "backtest", "pilot",
            f"results_v{iter_within}.md",
        ),
    ]
    # Also accept a sequenced file with hypothesis prefix.
    pilot_dir = os.path.join(project_dir, "backtest", "pilot")
    if os.path.isdir(pilot_dir):
        candidates.extend(
            os.path.join(pilot_dir, n)
            for n in os.listdir(pilot_dir)
            if n.startswith("results_") and n.endswith(".md")
        )

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                lower = f.read().lower()
        except Exception:
            continue
        for phrase in _MOCK_SIGNAL_PHRASES:
            if phrase in lower:
                return False, (
                    f"{os.path.relpath(path, project_dir)} self-identifies as using "
                    f"fake signals ({phrase!r}) — rerun the pilot backtest against "
                    f"the real llm_cache."
                )
    return True, ""


# ---------------------------------------------------------------------------
# Trading stage 3 (full validation) and stage 4 (verdict) gates.
# ---------------------------------------------------------------------------

_FULL_VALIDATION_GATES = {
    "oos_is_ratio_min": 0.5,
    "walk_forward_win_min": 0.6,
    "min_trades": 100,
    "min_t_stat": 2.0,
    "min_deflated_sharpe": 0.0,
    "turnover_tolerance": 0.5,
}


def _check_full_validation_metrics(project_dir: str) -> Tuple[bool, str]:
    """Stage-3 gate. Reads backtest/full/metrics.json (machine-readable) and
    enforces: OOS/IS Sharpe >= 0.5, walk-forward benchmark-beat >= 60%,
    total trades >= 100, |t-stat| > 2.0, deflated Sharpe > 0, turnover matches
    declared holding period within ± 50%. Any failure rejects the stage."""
    path = os.path.join(project_dir, "backtest", "full", "metrics.json")
    if not os.path.exists(path):
        return False, "Missing artifact: backtest/full/metrics.json"

    try:
        with open(path) as f:
            m = json.load(f)
    except Exception as e:
        return False, f"Cannot parse backtest/full/metrics.json: {e}"

    # Numeric sanity first — statistical gates are meaningless on internally
    # inconsistent numbers (see metrics_sanity; short blowups allowed).
    from sandbox_agent.pipeline.metrics_sanity import validate_metrics as _vm
    sanity_errs = _vm(m)
    if sanity_errs:
        return False, "Full-validation metrics fail numeric sanity: " + "; ".join(sanity_errs)

    def _num(key: str) -> Optional[float]:
        v = m.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    errs: List[str] = []

    pilot_sharpe = _num("pilot_sharpe")
    oos_sharpe = _num("oos_sharpe")
    if pilot_sharpe is None or oos_sharpe is None:
        errs.append("metrics.json missing pilot_sharpe or oos_sharpe")
    elif pilot_sharpe <= 0:
        errs.append(
            f"pilot_sharpe={pilot_sharpe:.2f} must be positive for the OOS-ratio check"
        )
    else:
        ratio = oos_sharpe / pilot_sharpe
        if ratio < _FULL_VALIDATION_GATES["oos_is_ratio_min"]:
            errs.append(
                f"oos_sharpe/pilot_sharpe = {ratio:.2f} < "
                f"{_FULL_VALIDATION_GATES['oos_is_ratio_min']:.2f} — OOS collapse"
            )

    wf = _num("walk_forward_win_rate")
    if wf is None:
        errs.append("metrics.json missing walk_forward_win_rate")
    elif wf < _FULL_VALIDATION_GATES["walk_forward_win_min"]:
        errs.append(
            f"walk_forward_win_rate={wf:.2f} < "
            f"{_FULL_VALIDATION_GATES['walk_forward_win_min']:.2f}"
        )

    trades = _num("total_trades")
    if trades is None:
        errs.append("metrics.json missing total_trades")
    elif trades < _FULL_VALIDATION_GATES["min_trades"]:
        errs.append(f"total_trades={int(trades)} < {_FULL_VALIDATION_GATES['min_trades']}")

    tstat = _num("t_stat_daily_returns")
    if tstat is None:
        errs.append("metrics.json missing t_stat_daily_returns")
    elif abs(tstat) < _FULL_VALIDATION_GATES["min_t_stat"]:
        errs.append(
            f"|t_stat_daily_returns|={abs(tstat):.2f} < "
            f"{_FULL_VALIDATION_GATES['min_t_stat']:.2f}"
        )

    dsr = _num("deflated_sharpe")
    if dsr is None:
        errs.append("metrics.json missing deflated_sharpe")
    elif dsr <= _FULL_VALIDATION_GATES["min_deflated_sharpe"]:
        errs.append(
            f"deflated_sharpe={dsr:.2f} <= {_FULL_VALIDATION_GATES['min_deflated_sharpe']}"
        )

    turnover = _num("turnover")
    declared = _num("declared_turnover")
    if turnover is not None and declared is not None and declared > 0:
        dev = abs(turnover - declared) / declared
        if dev > _FULL_VALIDATION_GATES["turnover_tolerance"]:
            errs.append(
                f"turnover={turnover:.2f} deviates from declared {declared:.2f} "
                f"by {dev:.0%} (> {_FULL_VALIDATION_GATES['turnover_tolerance']:.0%})"
            )

    if errs:
        return False, "Full validation gate(s) failed: " + "; ".join(errs)
    return True, ""


def _stamp_and_pin_full_metrics(project_dir: str) -> None:
    """After stage 3 passes: stamp backtest/full/metrics.json with the contract
    fields and pin its canonical hash into pipeline state. Evaluator-owned —
    the agent never computes these, which is what makes the pin trustworthy."""
    from sandbox_agent.pipeline.metrics_sanity import stamp_metrics_file
    from sandbox_agent.pipeline.orchestrator import load_state, save_state

    path = os.path.join(project_dir, "backtest", "full", "metrics.json")
    project_name = os.path.basename(os.path.normpath(project_dir))
    try:
        # Best-effort strategy version from the metrics file itself.
        with open(path) as f:
            version = str(json.load(f).get("strategy_version") or "unversioned")
        stamped = stamp_metrics_file(path, strategy_version=version)
        state = load_state(project_name)
        if state:
            state.pinned_full_metrics_hash = stamped["content_hash"]
            save_state(state)
            logger.info(
                f"Stage 3 pass: pinned full-metrics hash {stamped['content_hash'][:12]}… "
                f"for {project_name}")
    except Exception:
        logger.exception("Failed to stamp/pin full metrics (stage 3 still passes)")


def _write_pipeline_metrics_summary(project_dir: str) -> None:
    """After stage 4 passes: the EVALUATOR writes pipeline/metrics.json
    (verdict, gates_passed, hash). Previously agent-written, which produced
    files claiming gates_passed:0 / verdict:null alongside a PROMOTE."""
    from sandbox_agent.pipeline.metrics_sanity import canonical_hash
    from sandbox_agent.pipeline.orchestrator import load_state

    project_name = os.path.basename(os.path.normpath(project_dir))
    try:
        verdict_text = open(os.path.join(project_dir, "pipeline", "verdict.md")).read()
        rec = _extract_final_recommendation(verdict_text) or "unknown"
        full_path = os.path.join(project_dir, "backtest", "full", "metrics.json")
        with open(full_path) as f:
            full_metrics = json.load(f)
        state = load_state(project_name)
        summary = {
            "verdict": rec,
            "gates_passed": True,
            "metrics_hash": canonical_hash(full_metrics),
            "pinned_hash": state.pinned_full_metrics_hash if state else None,
            "strategy_version": full_metrics.get("strategy_version"),
            "written_by": "evaluator",
        }
        out = os.path.join(project_dir, "pipeline", "metrics.json")
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        os.replace(tmp, out)
    except Exception:
        logger.exception("Failed to write pipeline/metrics.json summary (stage 4 still passes)")


def _check_verdict_matches_metrics(project_dir: str) -> Tuple[bool, str]:
    """Stage-4 gate. Re-run the stage-3 gate against metrics.json; require the
    verdict's Final Recommendation to agree: all pass → 'promote', any fail →
    'reject'. Also enforce the metrics-hash pin: the metrics file must be the
    EXACT one stage 3 passed (canonical hash match) and the verdict must cite
    it — otherwise the verdict was rendered on different numbers than the file
    now holds ("phantom promote"). Mismatches fail acceptance."""
    verdict_path = os.path.join(project_dir, "pipeline", "verdict.md")
    if not os.path.exists(verdict_path):
        return False, "Missing artifact: pipeline/verdict.md"

    try:
        with open(verdict_path) as f:
            verdict_text = f.read()
    except Exception as e:
        return False, f"Cannot read pipeline/verdict.md: {e}"

    rec = _extract_final_recommendation(verdict_text)
    if rec is None:
        return False, (
            "pipeline/verdict.md is missing a `Final Recommendation` line "
            "with a `promote` or `reject` keyword."
        )

    # Metrics-hash pin (skipped for legacy pipelines with no pin).
    from sandbox_agent.pipeline.metrics_sanity import canonical_hash
    from sandbox_agent.pipeline.orchestrator import load_state
    project_name = os.path.basename(os.path.normpath(project_dir))
    state = load_state(project_name)
    pin = state.pinned_full_metrics_hash if state else None
    if pin:
        full_path = os.path.join(project_dir, "backtest", "full", "metrics.json")
        try:
            with open(full_path) as f:
                current_hash = canonical_hash(json.load(f))
        except Exception as e:
            return False, f"Cannot hash backtest/full/metrics.json: {e}"
        if current_hash != pin:
            return False, (
                f"Metrics changed since stage 3 passed (hash {current_hash[:12]}… != "
                f"pinned {pin[:12]}…) — the verdict would be judging numbers that never "
                f"passed validation. Stage 3 must re-run on the current metrics. "
                f"RESET_STAGE:3"
            )
        if f"Metrics Hash: {pin}" not in verdict_text:
            return False, (
                f"pipeline/verdict.md must cite the validated metrics with the literal "
                f"line `Metrics Hash: {pin}` so the verdict is pinned to the exact "
                f"numbers stage 3 passed. Add that line and re-submit."
            )

    metrics_passed, metrics_msg = _check_full_validation_metrics(project_dir)
    expected = "promote" if metrics_passed else "reject"
    if rec != expected:
        return False, (
            f"Verdict mismatch: metrics {'passed' if metrics_passed else 'failed'} "
            f"the full-validation gate ({metrics_msg or 'all pass'}), so the "
            f"recommendation must be {expected!r}, not {rec!r}. Rewrite the verdict."
        )
    return True, ""


def _extract_final_recommendation(verdict_text: str) -> Optional[str]:
    """Return 'promote' or 'reject' from the verdict markdown, or None if
    neither can be located. Looks for a `Final Recommendation` line and scans
    the next 400 chars."""
    lower = verdict_text.lower()
    idx = lower.find("final recommendation")
    if idx < 0:
        return None
    tail = lower[idx:idx + 400]
    if "promote" in tail and "reject" not in tail:
        return "promote"
    if "reject" in tail and "promote" not in tail:
        return "reject"
    # Both mentioned — take the one closer to the heading.
    prom_idx = tail.find("promote")
    rej_idx = tail.find("reject")
    if prom_idx < 0 and rej_idx < 0:
        return None
    if prom_idx < 0:
        return "reject"
    if rej_idx < 0:
        return "promote"
    return "promote" if prom_idx < rej_idx else "reject"
