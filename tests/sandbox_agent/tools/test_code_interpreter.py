"""Tests for the local code interpreter (stateless subprocess-per-call)."""

import pytest

from sandbox_agent.tools import code_interpreter as ci
from sandbox_agent.tools.code_interpreter import LocalCodeInterpreter, _build_script, _execute_code


def _has_pandas():
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def interpreter():
    """A shared tool instance — it's stateless, so sharing is harmless."""
    return LocalCodeInterpreter()


class TestCodeExecution:

    def test_simple_print(self, interpreter):
        result = interpreter.call('{"code": "print(42)"}')
        assert "42" in result
        assert "Exit 0" in result

    def test_math(self, interpreter):
        result = interpreter.call('{"code": "print(2 + 3)"}')
        assert "5" in result

    def test_multiline(self, interpreter):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = interpreter.call(f'{{"code": {repr(code)}}}')
        assert "30" in result

    def test_state_does_not_persist(self, interpreter):
        """Each call is a fresh process — a variable set in one call is gone in the next."""
        interpreter.call('{"code": "test_var = 123"}')
        result = interpreter.call('{"code": "print(test_var)"}')
        assert "NameError" in result
        assert "123" not in result

    def test_numpy_available(self, interpreter):
        result = interpreter.call('{"code": "import numpy as np; print(np.array([1,2,3]).sum())"}')
        assert "6" in result

    @pytest.mark.skipif(not _has_pandas(), reason="pandas not installed locally")
    def test_pandas_available(self, interpreter):
        result = interpreter.call('{"code": "import pandas as pd; print(pd.DataFrame({\'a\': [1,2,3]}).shape)"}')
        assert "(3, 1)" in result

    def test_requests_available(self, interpreter):
        result = interpreter.call('{"code": "import requests; print(type(requests.get))"}')
        assert "function" in result

    def test_path_vars_available(self, interpreter):
        result = interpreter.call('{"code": "print(PROJECTS_DIR == os.path.join(DATA_DIR, \'projects\'))"}')
        assert "True" in result

    def test_error_handling(self, interpreter):
        result = interpreter.call('{"code": "1/0"}')
        assert "ZeroDivisionError" in result
        # subprocess exited non-zero on the traceback
        assert "Exit 0" not in result

    def test_input_disabled(self, interpreter):
        result = interpreter.call('{"code": "input()"}')
        assert "NotImplementedError" in result

    def test_empty_code(self, interpreter):
        result = interpreter.call('{"code": ""}')
        assert result
        assert "Exit 0" in result

    def test_timeout_kills_the_process(self):
        """A hung script is killed at the timeout and returns promptly."""
        import time
        t0 = time.monotonic()
        result = _execute_code("import time; time.sleep(60)", timeout=2)
        elapsed = time.monotonic() - t0
        assert result.startswith("Timed out after 2s")
        assert elapsed < 15  # killed near the 2s mark, not after 60s

    def test_start_new_session_isolation(self, interpreter):
        """`kill 0` / killpg inside the script kills only the script's own
        process group — the test runner survives (trivially: this line runs)."""
        code = "import os, signal; os.killpg(os.getpgrp(), signal.SIGKILL)"
        result = interpreter.call(f'{{"code": {repr(code)}}}')
        assert result  # we got a result, and we're still alive
        assert "Exit 0" not in result  # the script was killed by the signal

class TestScriptBuilder:

    def test_prelude_sets_path_vars_and_disables_input(self):
        script = _build_script("print('hi')")
        assert "DATA_DIR = os.getenv('DATA_DIR'" in script
        assert "PROJECTS_DIR = os.path.join(DATA_DIR, 'projects')" in script
        assert "Python input() function is disabled" in script
        assert script.rstrip().endswith("print('hi')")

    def test_prelude_includes_llm_call_when_bridge_configured(self, monkeypatch):
        monkeypatch.setattr(ci, "_llm_init_code", "def llm_call(prompt, system='', think=False):\n    return 'stub'\n")
        script = _build_script("pass")
        assert "def llm_call(" in script

    def test_prelude_omits_llm_call_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(ci, "_llm_init_code", None)
        script = _build_script("pass")
        assert "def llm_call(" not in script


class TestToolRegistration:

    def test_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "code_interpreter" in TOOL_REGISTRY

    def test_schema(self):
        tool = LocalCodeInterpreter()
        func = tool.function
        assert func["name"] == "code_interpreter"
        assert "code" in func["parameters"]["properties"]

    def test_description_reflects_stateless(self):
        desc = LocalCodeInterpreter().description
        assert "FRESH" in desc or "fresh" in desc
        assert "persist" in desc.lower()
        assert "Variables persist between calls" not in desc
