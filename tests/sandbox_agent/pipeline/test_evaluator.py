"""Tests for pipeline stage evaluator — programmatic checks."""

import os
import shutil
import tempfile

import pytest

from sandbox_agent.pipeline.evaluator import _programmatic_checks
from sandbox_agent.pipeline.orchestrator import STARTUP_STAGES as STAGES


@pytest.fixture
def tmp_project_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestProgrammaticChecks:

    def test_missing_artifact_fails(self, tmp_project_dir):
        stage_defn = STAGES[1]
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert not passed
        assert "Missing artifact" in feedback

    def test_empty_artifact_fails(self, tmp_project_dir):
        stage_defn = STAGES[1]
        # Create the artifact but make it too short
        artifact_path = os.path.join(tmp_project_dir, "research")
        os.makedirs(artifact_path, exist_ok=True)
        with open(os.path.join(artifact_path, "market-research.md"), "w") as f:
            f.write("short")
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert not passed
        assert "too short" in feedback

    def test_missing_section_fails(self, tmp_project_dir):
        stage_defn = STAGES[1]
        artifact_path = os.path.join(tmp_project_dir, "research")
        os.makedirs(artifact_path, exist_ok=True)
        # Write enough content but missing required sections
        with open(os.path.join(artifact_path, "market-research.md"), "w") as f:
            f.write("# Market Research Report\n\n" + "Lorem ipsum dolor sit amet. " * 50)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert not passed
        assert "Missing section" in feedback

    def test_valid_artifact_passes(self, tmp_project_dir):
        stage_defn = STAGES[1]
        artifact_path = os.path.join(tmp_project_dir, "research")
        os.makedirs(artifact_path, exist_ok=True)
        content = (
            "# Market Research\n\n"
            "## Market Size\nThe total addressable market is $10B. " * 10 + "\n\n"
            "## Competitors\nThere are several key competitors. " * 10 + "\n\n"
            "## Target Customers\nOur target demographic is. " * 10 + "\n\n"
            "## Timing\nThe market is ready because. " * 10
        )
        with open(os.path.join(artifact_path, "market-research.md"), "w") as f:
            f.write(content)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert passed

    def test_brd_checks(self, tmp_project_dir):
        stage_defn = STAGES[2]
        brd_path = os.path.join(tmp_project_dir, "business")
        os.makedirs(brd_path, exist_ok=True)
        content = (
            "# BRD\n\n"
            "## Branding\nOur brand is. " * 10 + "\n\n"
            "## Legal\nKey legal considerations. " * 10 + "\n\n"
            "## Scalability\nWe scale by. " * 10 + "\n\n"
            "## Operations\nDay to day operations. " * 10 + "\n\n"
            "## Finance\nOur revenue model. " * 10
        )
        with open(os.path.join(brd_path, "brd.md"), "w") as f:
            f.write(content)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert passed

    def test_mvp_checks_readme_exists(self, tmp_project_dir):
        stage_defn = STAGES[5]
        mvp_path = os.path.join(tmp_project_dir, "mvp")
        os.makedirs(mvp_path, exist_ok=True)
        with open(os.path.join(mvp_path, "README.md"), "w") as f:
            f.write("# MVP\n\n" + "This is the MVP documentation. " * 20)
        passed, feedback = _programmatic_checks(tmp_project_dir, stage_defn)
        assert passed  # MVP only requires README.md to exist (no required sections)
