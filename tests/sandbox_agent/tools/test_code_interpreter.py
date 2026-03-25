"""Tests for the local code interpreter."""

import pytest

from sandbox_agent.tools.code_interpreter import LocalCodeInterpreter, _execute_code


def _has_pandas():
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def interpreter():
    """Shared interpreter instance (kernel is expensive to start)."""
    return LocalCodeInterpreter()


class TestCodeExecution:

    def test_simple_print(self, interpreter):
        result = interpreter.call('{"code": "print(42)"}')
        assert "42" in result

    def test_math(self, interpreter):
        result = interpreter.call('{"code": "print(2 + 3)"}')
        assert "5" in result

    def test_expression_result(self, interpreter):
        result = interpreter.call('{"code": "print(1 + 1)"}')
        assert "2" in result

    def test_multiline(self, interpreter):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = interpreter.call(f'{{"code": {repr(code)}}}')
        assert "30" in result

    def test_persistent_state(self, interpreter):
        """Variables persist between calls (same kernel)."""
        interpreter.call('{"code": "test_var = 123"}')
        result = interpreter.call('{"code": "print(test_var)"}')
        assert "123" in result

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

    def test_error_handling(self, interpreter):
        result = interpreter.call('{"code": "1/0"}')
        assert "ZeroDivisionError" in result

    def test_input_disabled(self, interpreter):
        result = interpreter.call('{"code": "input()"}')
        assert "NotImplementedError" in result

    def test_empty_code(self, interpreter):
        result = interpreter.call('{"code": ""}')
        assert result  # Should return "(no output)" or similar


class TestToolRegistration:

    def test_registered(self):
        from qwen_agent.tools.base import TOOL_REGISTRY
        assert "code_interpreter" in TOOL_REGISTRY

    def test_schema(self):
        tool = LocalCodeInterpreter()
        func = tool.function
        assert func["name"] == "code_interpreter"
        assert "code" in func["parameters"]["properties"]
