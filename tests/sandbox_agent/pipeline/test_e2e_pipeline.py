"""End-to-end trading-pipeline integration test with a scripted fake LLM.

Weakness #6: every catastrophic bug this system has had (phantom promote,
112-attempt runaways, state divergence) lived in the SEAMS between
components — task queue → stage runner → guards → evaluator gates →
orchestrator advancement — which unit tests can't see. This harness drives
the REAL machine end-to-end: start_trading_pipeline schedules stage 1, a
scripted "player" stands in for the LLM (writing the artifacts a real agent
would), and the loop runs task-by-task through all gates to a terminal state.

Only two things are stubbed: run_on_best_available (the LLM) and
_llm_evaluation (LLM quality judgment). Everything else is production code.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime

import pytest

import sandbox_agent.pipeline.evaluator as ev
import sandbox_agent.pipeline.orchestrator as orch
from qwen_agent.llm.schema import Message
from sandbox_agent.pipeline.metrics_sanity import canonical_hash
from sandbox_agent.pipeline.stage_runner import run_pipeline_stage
from sandbox_agent.scheduler.task_queue import TaskQueue

TICKERS = [f"T{i:02d}" for i in range(12)]           # >10 parseable cache entries
CLASSES = ["buy", "sell", "hold"]                     # 3 distinct, max bucket <= 60%

GOOD_FULL_METRICS = {
    "pilot_sharpe": 1.2, "oos_sharpe": 1.0, "walk_forward_win_rate": 0.75,
    "pilot_sortino": 1.6, "oos_sortino": 1.3,
    "pilot_annualized_return_pct": 14.0, "oos_annualized_return_pct": 11.0,
    "total_trades": 250, "t_stat_daily_returns": 3.1, "deflated_sharpe": 0.4,
    "turnover": 10.0, "declared_turnover": 9.0,
    "return_pct": 40.0, "dd_pct": -18.0,
}
BAD_FULL_METRICS = dict(GOOD_FULL_METRICS, oos_sortino=0.1)  # OOS collapse → gates fail


class Player:
    """Scripted stand-in for the LLM: writes the artifacts a real agent would."""

    def __init__(self, data_dir: str, project: str, full_metrics: dict):
        self.data_dir = data_dir
        self.project = project
        self.full_metrics = full_metrics
        self.stages_played = []

    def _p(self, rel: str) -> str:
        path = os.path.join(self.data_dir, "projects", self.project, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _w(self, rel: str, content: str) -> None:
        with open(self._p(rel), "w") as f:
            f.write(content)

    def __call__(self, system_message, messages, task_label=""):
        # task_label: "Pipeline: {project} stage {n} ({name})"
        stage = int(task_label.split(" stage ")[1].split(" ")[0])
        self.stages_played.append(stage)
        getattr(self, f"stage_{stage}")()
        return [Message(role="assistant", content=f"Stage {stage} artifacts written.")]

    def stage_1(self):
        self._w("research/data-landscape.md",
                "# Landscape\n\n## Data Sources\n- EDGAR filings\n- price history\n\n"
                "## Sample Extractions\nclean prose sample\n\n"
                "## Alpha Rationale\nfilings lead prices\n")

    def stage_2(self):
        self._w("strategy/hypothesis_v1.md",
                "# Hypothesis v1\n\nrequired_data_types: [EDGAR filings, price history]\n\n"
                "Filings language predicts drift.\n")
        self._w("strategy/universe_v1.json", json.dumps(TICKERS))
        for i, tk in enumerate(TICKERS):
            self._w(f"data/processed/llm_cache/pilot/{tk.lower()}_1.json",
                    json.dumps({"ticker": tk, "classification": CLASSES[i % 3]}))
        self._w("backtest/pilot/strategy_v1.py",
                "# reads data/processed/llm_cache/pilot/ signals\n"
                "from sandbox_agent.trading_accounting import Ledger, compute_metrics\n")
        self._w("backtest/pilot/results_v1.md",
                "# Pilot v1\nReal signals from filings classifications drove trades.\n")
        self._w("backtest/pilot/metrics_latest.json",
                json.dumps({"return_pct": 14.0, "dd_pct": -9.0, "trades": 45, "sharpe": 1.02}))
        # 3-iteration plateau above converge thresholds (sharpe>=0.8, trades>=30)
        history = [{"hyp": 1, "iter": i + 1, "version": i + 1,
                    "sharpe": s, "trades": 45}
                   for i, s in enumerate([0.95, 1.00, 1.02])]
        self._w("strategy/loop_state.json", json.dumps({
            "oos_cutoff_date": "2025-06-01",
            "hypothesis_count": 1,
            "total_iterations": 3,
            "pilot_history": history,
        }))

    def stage_3(self):
        self._w("backtest/full/metrics.json", json.dumps(self.full_metrics))
        self._w("backtest/full/results.md",
                "# Full Validation\n\n## OOS Sharpe\n1.0\n\n## Sortino\n"
                "pilot 1.6 / OOS 1.3\n\n## Annualized Return\n"
                "pilot 14.0% / OOS 11.0%\n\n## Walk-Forward\n75%\n\n"
                "## Benchmark Comparison\nbeats\n\n## Trade Count\n250\n\n"
                "## t-Statistic\n3.1\n\n## Turnover\n10\n")
        self._w("backtest/full/strategy.py",
                "# uses data/processed/llm_cache/ signals\n"
                "from sandbox_agent.trading_accounting import Ledger, compute_metrics\n")

    def stage_4(self):
        state = orch.load_state(self.project)
        pin = state.pinned_full_metrics_hash
        if pin:
            rec, hash_line = "promote", f"\nMetrics Hash: {pin}\n"
        else:
            rec, hash_line = "reject", ""
        self._w("pipeline/verdict.md",
                f"# Verdict\n\n## Final Recommendation\n{rec}\n{hash_line}\n"
                f"## Rationale\ngates {'passed' if pin else 'failed'}; numbers cited.\n\n"
                f"## Strategy Summary\nfilings-driven long/short.\n"
                f"## Execution Strategy\n### Entry Criteria\nbreakout close.\n"
                f"### Exit Criteria\n10d hold.\n### Position Sizing\n10% each.\n"
                f"### Portfolio Strategy\nmax 5.\n### Risk Management\nstop 10%.\n")

    def stage_5(self):
        self._w("paper/deploy.py",
                "\"\"\"Paper-trading scaffold for the promoted strategy.\"\"\"\n"
                "import json\n\n\n"
                "def place_order(symbol, qty, side):\n"
                "    print(f'paper order: {side} {qty} {symbol}')\n\n\n"
                "def kill_switch(reason):\n"
                "    print(f'HALT: {reason}')\n\n\n"
                "if __name__ == '__main__':\n"
                "    place_order('T01', 100, 'sell_short')\n")
        self._w("paper/README.md",
                "# Paper Trading Scaffold\n\n"
                "## Broker Integration\nAlpaca paper API via env-configured keys.\n\n"
                "## Monitoring\nDaily equity snapshot appended to logs/paper.jsonl.\n\n"
                "## Kill Switch\nkill_switch() halts on drawdown beyond -15%.\n")

    def stage_6(self):
        self._w("pipeline/review.md",
                "# Review\n\n## Performance Summary\nsolid\n\n## Robustness\nfine\n\n"
                "## Deployment Readiness\nready\n\n## Learnings\n- plateau early, validate once.\n")


@pytest.fixture
def rig(monkeypatch):
    """Wire the whole machine into a temp DATA_DIR with a fake LLM."""
    d = tempfile.mkdtemp()
    for target in ("sandbox_agent.pipeline.orchestrator.DATA_DIR",
                   "sandbox_agent.pipeline.evaluator.DATA_DIR",
                   "sandbox_agent.pipeline.stage_runner.DATA_DIR",
                   "sandbox_agent.tools.project_tools.DATA_DIR",
                   "sandbox_agent.tools.self_edit_tools.DATA_DIR"):
        monkeypatch.setattr(target, d)
    monkeypatch.setattr("sandbox_agent.tools.project_tools.PROJECTS_DIR",
                        os.path.join(d, "projects"))
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE",
                        os.path.join(d, "pipeline.lock"))
    import sandbox_agent.tools.git_autocommit as ga
    monkeypatch.setattr(ga, "autocommit", lambda *a, **k: None)
    import sandbox_agent.tools.notification_tools as nt
    monkeypatch.setattr(nt.RequestUser, "call", lambda self, p, **k: "ok")
    # LLM quality judgment stubbed — this harness tests the MACHINE.
    monkeypatch.setattr(ev, "_llm_evaluation", lambda *a, **k: (True, "stubbed quality pass"))

    tq = TaskQueue(data_dir=d)
    import sandbox_agent.scheduler.scheduler_tools as st
    st.set_task_queue(tq)

    yield d, tq
    st.set_task_queue(None) if hasattr(st, "set_task_queue") else None
    shutil.rmtree(d)


def _drive(tq: TaskQueue, project: str, max_steps: int = 40) -> list:
    """Pump the task queue like the cron loop would, one task at a time."""
    executed = []
    for _ in range(max_steps):
        state = orch.load_state(project)
        if state and state.status in ("completed", "completed_rejected", "failed"):
            break
        due = tq.get_due_tasks()
        if not due:
            break
        task = due[0]
        result = run_pipeline_stage(task, "system message")
        executed.append((task.name, result))
        tq.update_task(task.id, status="completed", result=result[:200])
    return executed


def _start(d, tq, project, full_metrics, monkeypatch):
    player = Player(d, project, full_metrics)
    monkeypatch.setattr("sandbox_agent.main.run_on_best_available", player)
    os.makedirs(os.path.join(d, "projects", project), exist_ok=True)
    state = orch.init_pipeline(project, "test strategy", pipeline_type="trading")
    orch.save_state(state)
    orch._schedule_stage(state, 1)
    return player


def test_full_pipeline_promote_path(rig, monkeypatch):
    d, tq = rig
    player = _start(d, tq, "e2e-promote", GOOD_FULL_METRICS, monkeypatch)

    executed = _drive(tq, "e2e-promote")

    state = orch.load_state("e2e-promote")
    assert state.status == "completed", [e[1] for e in executed]
    assert player.stages_played == [1, 2, 3, 4, 5, 6]
    for n in range(1, 7):
        assert state.stages[n].status == "completed", (n, state.stages[n].acceptance_result)

    # Metrics contract held end-to-end: pin set at stage 3, cited at stage 4,
    # summary written by the EVALUATOR (not the player).
    assert state.pinned_full_metrics_hash
    proj = os.path.join(d, "projects", "e2e-promote")
    verdict = open(os.path.join(proj, "pipeline", "verdict.md")).read()
    assert f"Metrics Hash: {state.pinned_full_metrics_hash}" in verdict
    summary = json.load(open(os.path.join(proj, "pipeline", "metrics.json")))
    assert summary["verdict"] == "promote"
    assert summary["written_by"] == "evaluator"
    on_disk = json.load(open(os.path.join(proj, "backtest", "full", "metrics.json")))
    assert canonical_hash(on_disk) == state.pinned_full_metrics_hash

    # Self-improvement side effects fired.
    assert "review.md" in open(os.path.join(d, "HEARTBEAT.md")).read()
    learnings = open(os.path.join(d, "learnings", "pipeline-learnings.md")).read()
    assert "plateau early" in learnings


def test_full_pipeline_reject_path(rig, monkeypatch):
    d, tq = rig
    player = _start(d, tq, "e2e-reject", BAD_FULL_METRICS, monkeypatch)

    executed = _drive(tq, "e2e-reject", max_steps=60)

    state = orch.load_state("e2e-reject")
    assert state.status == "completed_rejected", [e[1] for e in executed]
    # Stage 3's gates fail every attempt → exhausts to best-effort completion.
    assert state.stages[3].status == "completed-no-more-attempts"
    # Stage 4 wrote a reject verdict that the gate accepted.
    verdict = open(os.path.join(d, "projects", "e2e-reject", "pipeline", "verdict.md")).read()
    assert "reject" in verdict.lower()
    assert state.stages[4].status == "completed"
    # Stages 5 and 6 must be SKIPPED — a rejected strategy is never scaffolded.
    assert 5 not in player.stages_played
    assert 6 not in player.stages_played
    assert state.pinned_full_metrics_hash is None      # never validated


def test_stale_duplicate_task_cannot_rerun_finished_stage(rig, monkeypatch):
    """Regression: the phantom-promote class — a stray duplicate task re-running
    a completed stage. The orphan/terminal guards must skip it."""
    d, tq = rig
    player = _start(d, tq, "e2e-guard", GOOD_FULL_METRICS, monkeypatch)
    _drive(tq, "e2e-guard")
    assert orch.load_state("e2e-guard").status == "completed"

    rogue = tq.add_task(name="pipeline:e2e-guard:stage_4_verdict",
                        description="agent-created duplicate", schedule_type="at",
                        run_at=datetime.now())
    out = run_pipeline_stage(rogue, "system message")
    assert "skip" in out.lower()
    assert player.stages_played.count(4) == 1          # never re-executed
