"""Tests for the exec shell tool."""

import os
import json

import pytest


@pytest.fixture
def exec_tool(tmp_path, monkeypatch):
    """Create exec tool with temp DATA_DIR."""
    import sandbox_agent.tools.exec_tool as et
    import sandbox_agent.config as cfg

    data_dir = str(tmp_path)
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(et, "DATA_DIR", data_dir)

    # Create a project dir
    project_dir = os.path.join(data_dir, "projects", "test-proj")
    os.makedirs(project_dir)

    from sandbox_agent.tools.exec_tool import ExecTool
    return ExecTool()


class TestBasicExecution:
    def test_echo(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": "echo hello"}))
        assert "Exit 0" in result
        assert "hello" in result

    def test_exit_code(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": "false"}))
        assert "Exit 1" in result

    def test_stderr(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": "echo error >&2"}))
        assert "error" in result

    def test_pipes(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": "echo hello world | wc -w"}))
        assert "Exit 0" in result
        assert "2" in result

    def test_and_chain(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": "echo first && echo second"}))
        assert "first" in result
        assert "second" in result


class TestWorkdir:
    def test_default_workdir(self, exec_tool, tmp_path):
        result = exec_tool.call(json.dumps({"command": "pwd"}))
        assert str(tmp_path) in result

    def test_project_workdir(self, exec_tool, tmp_path):
        result = exec_tool.call(json.dumps({
            "command": "pwd",
            "project": "test-proj",
        }))
        assert "test-proj" in result

    def test_missing_project(self, exec_tool):
        result = exec_tool.call(json.dumps({
            "command": "pwd",
            "project": "nonexistent",
        }))
        assert "not found" in result

    def test_workdir_escape_blocked(self, exec_tool):
        result = exec_tool.call(json.dumps({
            "command": "pwd",
            "workdir": "/etc",
        }))
        assert "Error" in result


class TestTimeout:
    def test_timeout(self, exec_tool):
        result = exec_tool.call(json.dumps({
            "command": "sleep 300",
            "timeout": 2,
        }))
        assert "Timed out" in result

    def test_timeout_capped(self, exec_tool):
        # Timeout should be capped at MAX_TIMEOUT (600)
        result = exec_tool.call(json.dumps({
            "command": "echo fast",
            "timeout": 9999,
        }))
        assert "Exit 0" in result


class TestBlocklist:
    def test_rm_rf_root(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": "rm -rf /"}))
        assert "blocked" in result.lower()

    def test_rm_fr_root(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": "rm -fr /"}))
        assert "blocked" in result.lower()

    def test_fork_bomb(self, exec_tool):
        result = exec_tool.call(json.dumps({"command": ":(){ :|:& };:"}))
        assert "blocked" in result.lower()

    def test_safe_rm_allowed(self, exec_tool):
        # rm on a specific file should NOT be blocked
        result = exec_tool.call(json.dumps({"command": "rm -f /tmp/nonexistent"}))
        assert "blocked" not in result.lower()


class TestOutputTruncation:
    def test_large_output_truncated(self, exec_tool):
        # Generate output larger than 64k chars
        result = exec_tool.call(json.dumps({
            "command": "python3 -c \"print('x' * 100000)\"",
        }))
        assert "TRUNCATED" in result or "truncated" in result.lower() or len(result) < 100000


class TestProjectVenv:
    """Per-project venv isolation: exec with project= activates that project's
    .venv so pip installs and `python` are isolated from other projects."""

    def _mk_venv(self, project_dir):
        bindir = os.path.join(project_dir, ".venv", "bin")
        os.makedirs(bindir, exist_ok=True)
        open(os.path.join(bindir, "python"), "w").close()
        return os.path.join(project_dir, ".venv")

    def test_activates_when_venv_exists(self, tmp_path):
        import sandbox_agent.tools.exec_tool as et
        proj = str(tmp_path / "proj")
        venv = self._mk_venv(proj)
        base = {"PATH": "/usr/bin", "PYTHONHOME": "/should/be/removed"}
        env = et._project_venv_env(base, proj)
        assert env["VIRTUAL_ENV"] == venv
        assert env["PATH"].startswith(os.path.join(venv, "bin") + os.pathsep)
        assert "PYTHONHOME" not in env          # cleared so the venv python wins
        assert base.get("PATH") == "/usr/bin"   # original not mutated

    def test_noop_when_no_venv(self, tmp_path):
        import sandbox_agent.tools.exec_tool as et
        proj = str(tmp_path / "proj")
        os.makedirs(proj)
        base = {"PATH": "/usr/bin"}
        env = et._project_venv_env(base, proj)
        assert "VIRTUAL_ENV" not in env
        assert env["PATH"] == "/usr/bin"

    def test_ensure_skips_when_venv_present(self, tmp_path, monkeypatch):
        import sandbox_agent.tools.exec_tool as et
        proj = str(tmp_path / "proj")
        self._mk_venv(proj)
        calls = []
        monkeypatch.setattr(et.subprocess, "run", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(et, "PROJECT_VENV_ENABLED", True)
        et._ensure_project_venv(proj)
        assert calls == []   # already there → don't invoke uv

    def test_ensure_disabled_by_flag(self, tmp_path, monkeypatch):
        import sandbox_agent.tools.exec_tool as et
        proj = str(tmp_path / "proj")
        os.makedirs(proj)
        calls = []
        monkeypatch.setattr(et.subprocess, "run", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(et, "PROJECT_VENV_ENABLED", False)
        et._ensure_project_venv(proj)
        assert calls == []

    def test_ensure_creates_with_uv_when_missing(self, tmp_path, monkeypatch):
        import sandbox_agent.tools.exec_tool as et
        proj = str(tmp_path / "proj")
        os.makedirs(proj)
        calls = []
        monkeypatch.setattr(et.subprocess, "run", lambda *a, **k: calls.append(list(a[0])))
        monkeypatch.setattr(et, "PROJECT_VENV_ENABLED", True)
        monkeypatch.setattr(et, "UV_BIN", "uv")
        et._ensure_project_venv(proj)
        assert len(calls) == 1
        assert calls[0][:2] == ["uv", "venv"]
        assert calls[0][2].endswith(os.path.join("proj", ".venv"))
