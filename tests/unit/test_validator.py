"""Tests for code validator - SECURITY CRITICAL."""

import pytest

from rlm.execution.validator import CodeValidator, ValidationResult


class TestCodeValidator:
    """Tests for CodeValidator - these are SECURITY CRITICAL tests."""

    @pytest.fixture
    def validator(self) -> CodeValidator:
        """Create a validator instance."""
        return CodeValidator()

    # Safe code tests
    def test_valid_simple_code(self, validator: CodeValidator) -> None:
        """Test that simple valid code passes."""
        code = "x = 1 + 2"
        result = validator.validate(code)
        assert result.is_valid

    def test_valid_function(self, validator: CodeValidator) -> None:
        """Test valid function definition."""
        code = """
def calculate(a, b):
    return a + b

result = calculate(1, 2)
"""
        result = validator.validate(code)
        assert result.is_valid

    def test_valid_list_operations(self, validator: CodeValidator) -> None:
        """Test valid list operations."""
        code = """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
filtered = [x for x in numbers if x > 2]
"""
        result = validator.validate(code)
        assert result.is_valid

    def test_valid_string_operations(self, validator: CodeValidator) -> None:
        """Test valid string operations."""
        code = """
text = "Hello, World!"
upper = text.upper()
length = len(text)
"""
        result = validator.validate(code)
        assert result.is_valid

    # Dangerous code tests - MUST FAIL
    def test_import_os_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: import os must be blocked."""
        code = "import os"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("os" in e.lower() for e in result.errors)

    def test_import_subprocess_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: import subprocess must be blocked."""
        code = "import subprocess"
        result = validator.validate(code)
        assert not result.is_valid

    def test_import_sys_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: import sys must be blocked."""
        code = "import sys"
        result = validator.validate(code)
        assert not result.is_valid

    def test_from_os_import_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: from os import must be blocked."""
        code = "from os import system"
        result = validator.validate(code)
        assert not result.is_valid

    def test_eval_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: eval() must be blocked."""
        code = "eval('1+1')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_exec_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: exec() must be blocked."""
        code = "exec('x = 1')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_open_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: open() must be blocked."""
        code = "open('/etc/passwd')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_dunder_import_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: __import__ must be blocked."""
        code = "__import__('os')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_dunder_builtins_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: __builtins__ access must be blocked."""
        code = "x = __builtins__"
        result = validator.validate(code)
        assert not result.is_valid

    def test_dunder_class_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: __class__ attribute access must be blocked."""
        code = "x = ''.__class__"
        result = validator.validate(code)
        assert not result.is_valid

    def test_dunder_subclasses_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: __subclasses__ must be blocked."""
        code = "x = object.__subclasses__()"
        result = validator.validate(code)
        assert not result.is_valid

    def test_globals_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: globals() must be blocked."""
        code = "globals()"
        result = validator.validate(code)
        assert not result.is_valid

    def test_locals_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: locals() must be blocked."""
        code = "locals()"
        result = validator.validate(code)
        assert not result.is_valid

    def test_getattr_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: getattr must be blocked."""
        code = "getattr(object, '__class__')"
        result = validator.validate(code)
        assert not result.is_valid

    # Network-related imports
    def test_socket_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: socket module must be blocked."""
        code = "import socket"
        result = validator.validate(code)
        assert not result.is_valid

    def test_requests_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: requests module must be blocked."""
        code = "import requests"
        result = validator.validate(code)
        assert not result.is_valid

    def test_urllib_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: urllib must be blocked."""
        code = "import urllib"
        result = validator.validate(code)
        assert not result.is_valid

    # Pickle/serialization attacks
    def test_pickle_blocked(self, validator: CodeValidator) -> None:
        """SECURITY: pickle must be blocked."""
        code = "import pickle"
        result = validator.validate(code)
        assert not result.is_valid

    # Syntax errors
    def test_syntax_error(self, validator: CodeValidator) -> None:
        """Test syntax error detection."""
        code = "def f(\n"  # Invalid syntax
        result = validator.validate(code)
        assert not result.is_valid
        assert any("syntax" in e.lower() for e in result.errors)

    # Allowed imports
    def test_math_allowed(self, validator: CodeValidator) -> None:
        """Test that math module is allowed."""
        code = "import math\nx = math.sqrt(4)"
        result = validator.validate(code)
        assert result.is_valid

    def test_re_allowed(self, validator: CodeValidator) -> None:
        """Test that re module is allowed."""
        code = "import re\nresult = re.search('test', 'test string')"
        result = validator.validate(code)
        assert result.is_valid

    def test_json_allowed(self, validator: CodeValidator) -> None:
        """Test that json module is allowed."""
        code = "import json\ndata = json.loads('{\"key\": \"value\"}')"
        result = validator.validate(code)
        assert result.is_valid

    # Complexity warnings
    def test_high_complexity_warning(self, validator: CodeValidator) -> None:
        """Test complexity warning for large code."""
        # Generate complex code
        code = "\n".join([f"x{i} = {i}" for i in range(200)])
        result = validator.validate(code)
        # Should still be valid but may have warnings
        # This depends on max_complexity setting

    # Edge cases
    def test_empty_code(self, validator: CodeValidator) -> None:
        """Test empty code."""
        result = validator.validate("")
        assert result.is_valid  # Empty code is technically valid

    def test_comment_only(self, validator: CodeValidator) -> None:
        """Test code with only comments."""
        code = "# This is a comment"
        result = validator.validate(code)
        assert result.is_valid
