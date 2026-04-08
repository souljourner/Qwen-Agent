"""Tests for pipeline tools — start_pipeline, pipeline_status, list_pipelines."""

import json
import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.pipeline_tools import (
    ListPipelines,
    PipelineStatusTool,
    StartPipeline,
    _validate_idea,
)


@pytest.fixture
def tmp_data_dir(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.pipeline.orchestrator.LOCK_FILE", os.path.join(d, "pipeline.lock"))
    monkeypatch.setattr("sandbox_agent.tools.project_tools.DATA_DIR", d)
    monkeypatch.setattr("sandbox_agent.tools.project_tools.PROJECTS_DIR", os.path.join(d, "projects"))
    os.makedirs(os.path.join(d, "projects"), exist_ok=True)
    yield d
    shutil.rmtree(d)


class TestValidateIdea:

    def test_too_short(self):
        valid, feedback = _validate_idea("An app")
        assert not valid
        assert "too short" in feedback

    def test_missing_who(self):
        valid, feedback = _validate_idea(
            "A platform that uses machine learning to optimize supply chains and reduce waste in manufacturing processes"
        )
        assert not valid
        assert "target customer" in feedback.lower() or "who" in feedback.lower()

    def test_missing_what(self):
        valid, feedback = _validate_idea(
            "For enterprise customers and small business owners in the healthcare demographic aged 30 to 50"
        )
        assert not valid
        assert "solution" in feedback.lower() or "what" in feedback.lower()

    def test_valid_idea(self):
        valid, feedback = _validate_idea(
            "An AI-powered platform that helps busy pet owners monitor their pet's health "
            "through smart collar sensors and provides early disease detection alerts to customers."
        )
        assert valid

    def test_valid_with_business_keyword(self):
        valid, feedback = _validate_idea(
            "A tool for small business owners that solves the problem of tracking employee "
            "time and attendance using AI-powered facial recognition."
        )
        assert valid


class TestStartPipeline:

    def test_invalid_idea_returns_questions(self, tmp_data_dir):
        tool = StartPipeline()
        result = tool.call(json.dumps({"name": "test", "description": "An app"}))
        assert "Cannot start pipeline" in result
        assert "clarification" in result

    def test_valid_idea_starts_pipeline(self, tmp_data_dir, monkeypatch):
        # Mock the task queue to avoid actual scheduling
        from sandbox_agent.scheduler import scheduler_tools
        from sandbox_agent.scheduler.task_queue import TaskQueue
        tq = TaskQueue(data_dir=tmp_data_dir)
        monkeypatch.setattr(scheduler_tools, "_task_queue", tq)

        tool = StartPipeline()
        result = tool.call(json.dumps({
            "name": "pet-health",
            "description": "An AI-powered platform that helps busy pet owners monitor their pet's health through smart collar sensors and provides early disease detection alerts to customers.",
        }))
        assert "Pipeline started" in result
        assert "pet-health" in result
        assert "Stage 1" in result

    def test_already_running_returns_status(self, tmp_data_dir, monkeypatch):
        from sandbox_agent.scheduler import scheduler_tools
        from sandbox_agent.scheduler.task_queue import TaskQueue
        tq = TaskQueue(data_dir=tmp_data_dir)
        monkeypatch.setattr(scheduler_tools, "_task_queue", tq)

        tool = StartPipeline()
        desc = "An AI-powered platform that helps busy pet owners monitor their pet's health through smart collar sensors and provides early disease detection alerts to customers."
        tool.call(json.dumps({"name": "pet-health", "description": desc}))

        # Try to start again
        result = tool.call(json.dumps({"name": "pet-health", "description": desc}))
        assert "already running" in result


class TestPipelineStatus:

    def test_no_pipeline(self, tmp_data_dir):
        tool = PipelineStatusTool()
        result = tool.call(json.dumps({"project": "nonexistent"}))
        assert "No pipeline found" in result

    def test_shows_status(self, tmp_data_dir, monkeypatch):
        from sandbox_agent.pipeline.orchestrator import init_pipeline
        init_pipeline("test-project", "Test idea")

        tool = PipelineStatusTool()
        result = tool.call(json.dumps({"project": "test-project"}))
        assert "test-project" in result
        assert "market_research" in result

    def test_shows_all_pipelines(self, tmp_data_dir):
        from sandbox_agent.pipeline.orchestrator import init_pipeline
        init_pipeline("project-a", "Idea A")
        init_pipeline("project-b", "Idea B")

        tool = PipelineStatusTool()
        result = tool.call("{}")
        assert "project-a" in result
        assert "project-b" in result


class TestListPipelines:

    def test_empty(self, tmp_data_dir):
        tool = ListPipelines()
        result = tool.call("{}")
        assert "No pipelines" in result

    def test_lists_pipelines(self, tmp_data_dir):
        from sandbox_agent.pipeline.orchestrator import init_pipeline
        init_pipeline("alpha", "Alpha idea")
        init_pipeline("beta", "Beta idea")

        tool = ListPipelines()
        result = tool.call("{}")
        assert "alpha" in result
        assert "beta" in result


class TestToolRegistration:

    def test_start_pipeline_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "start_pipeline" in TOOL_REGISTRY

    def test_pipeline_status_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "pipeline_status" in TOOL_REGISTRY

    def test_list_pipelines_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "list_pipelines" in TOOL_REGISTRY
