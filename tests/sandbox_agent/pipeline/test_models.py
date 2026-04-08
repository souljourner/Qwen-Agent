"""Tests for pipeline data models."""

from datetime import datetime

from sandbox_agent.pipeline.models import PipelineState, StageState


class TestStageState:

    def test_defaults(self):
        stage = StageState(stage_number=1, stage_name="market_research")
        assert stage.status == "scheduled"
        assert stage.run_count == 0
        assert stage.max_attempts == 5
        assert stage.notes == ""
        assert stage.artifacts == []

    def test_all_statuses_valid(self):
        for status in ["scheduled", "running", "part-completion", "completed",
                        "completed-no-more-attempts", "failed", "failed-no-more-attempts"]:
            stage = StageState(stage_number=1, stage_name="test", status=status)
            assert stage.status == status

    def test_serialization_roundtrip(self):
        stage = StageState(
            stage_number=3, stage_name="prd", status="running",
            run_count=2, notes="Fix the scope section",
            artifacts=["product/prd.md"],
        )
        data = stage.model_dump(mode="json")
        restored = StageState(**data)
        assert restored.stage_number == 3
        assert restored.run_count == 2
        assert restored.notes == "Fix the scope section"


class TestPipelineState:

    def test_defaults(self):
        state = PipelineState(project_name="test", description="Test idea")
        assert state.current_stage == 0
        assert state.status == "running"
        assert state.stages == {}

    def test_with_stages(self):
        stages = {
            1: StageState(stage_number=1, stage_name="market_research"),
            2: StageState(stage_number=2, stage_name="brd"),
        }
        state = PipelineState(
            project_name="my-project",
            description="An AI thing",
            stages=stages,
            current_stage=1,
        )
        assert len(state.stages) == 2
        assert state.stages[1].stage_name == "market_research"

    def test_serialization_roundtrip(self):
        state = PipelineState(
            project_name="test",
            description="Test",
            current_stage=2,
            status="running",
            stages={
                1: StageState(stage_number=1, stage_name="market_research", status="completed"),
                2: StageState(stage_number=2, stage_name="brd", status="running", run_count=1),
            },
        )
        data = state.model_dump(mode="json")
        restored = PipelineState(**data)
        assert restored.current_stage == 2
        assert restored.stages[1].status == "completed"
        assert restored.stages[2].run_count == 1
