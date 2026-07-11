"""Pipeline data models — state tracking for multi-stage project builds."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

StageStatus = Literal[
    "scheduled",
    "running",
    "part-completion",
    "completed",
    "completed-no-more-attempts",
    "failed",
    "failed-no-more-attempts",
]

PipelineStatus = Literal[
    "running",
    "completed",
    "completed_rejected",
    "failed",
    "paused",
]


# Statuses written by older code versions, normalized on load so historical
# state files keep loading (observed live: 'complete', 'skipped',
# 'skipped-rejected' — current code writes 'completed' with an
# acceptance_result note for verdict-skipped stages).
_LEGACY_STATUS_MAP = {
    "complete": "completed",
    "skipped": "completed",
    "skipped-rejected": "completed",
}


class StageState(BaseModel):
    """State of a single pipeline stage."""

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_legacy_status(cls, v):
        return _LEGACY_STATUS_MAP.get(v, v)

    stage_number: int
    stage_name: str
    status: StageStatus = "scheduled"
    run_count: int = 0
    max_attempts: int = 5
    notes: str = ""
    task_id: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_error: Optional[str] = None
    acceptance_result: Optional[str] = None
    part_completion_count: int = 0
    max_part_completions: int = 20
    first_started_at: Optional[datetime] = None
    budget_seconds: Optional[int] = None


class PipelineState(BaseModel):
    """Full pipeline state for one project."""

    project_name: str
    description: str
    pipeline_type: str = "startup"
    current_stage: int = 0
    status: PipelineStatus = "running"
    stages: Dict[int, StageState] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    lock_holder: Optional[str] = None
    pilot_llm_cost_usd: float = 0.0
    # Set by the evaluator when trading stage 3 passes: canonical hash of the
    # exact backtest/full/metrics.json that passed the gates. Stage 4's verdict
    # must cite this hash — if the metrics change afterwards, the skew fails
    # acceptance and stage 3 re-runs (prevents "phantom promote" where the
    # verdict was rendered on different metrics than the file now holds).
    # None = legacy pipeline predating this contract; the hash gate is skipped.
    pinned_full_metrics_hash: Optional[str] = None
