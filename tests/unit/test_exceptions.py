"""Tests for exceptions module."""

import pytest

from rlm.exceptions import (
    CodeExecutionError,
    ConfigurationError,
    ContextError,
    LLMError,
    MaxDepthExceededError,
    RLMException,
    SecurityError,
    TimeoutError,
    ValidationError,
)


class TestRLMException:
    """Tests for base RLMException."""

    def test_basic_exception(self) -> None:
        """Test basic exception creation."""
        exc = RLMException("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.details == {}

    def test_exception_with_details(self) -> None:
        """Test exception with details."""
        exc = RLMException("Test error", details={"key": "value"})
        assert "key" in str(exc)
        assert exc.details["key"] == "value"


class TestMaxDepthExceededError:
    """Tests for MaxDepthExceededError."""

    def test_error_message(self) -> None:
        """Test error message format."""
        exc = MaxDepthExceededError(current_depth=5, max_depth=3)
        assert "5" in str(exc)
        assert "3" in str(exc)
        assert exc.current_depth == 5
        assert exc.max_depth == 3


class TestCodeExecutionError:
    """Tests for CodeExecutionError."""

    def test_with_code(self) -> None:
        """Test error with code snippet."""
        exc = CodeExecutionError(
            "Execution failed",
            code="x = 1 / 0",
            original_error=ZeroDivisionError(),
        )
        assert exc.code == "x = 1 / 0"
        assert exc.original_error is not None

    def test_code_truncation(self) -> None:
        """Test that long code is truncated in details."""
        long_code = "x = 1\n" * 200
        exc = CodeExecutionError("Failed", code=long_code)
        assert len(exc.details.get("code_snippet", "")) <= 510


class TestSecurityError:
    """Tests for SecurityError."""

    def test_with_violation_info(self) -> None:
        """Test error with violation details."""
        exc = SecurityError(
            "Security violation",
            violation_type="forbidden_import",
            blocked_operation="import os",
        )
        assert exc.violation_type == "forbidden_import"
        assert exc.blocked_operation == "import os"


class TestLLMError:
    """Tests for LLMError."""

    def test_with_provider_info(self) -> None:
        """Test error with provider information."""
        exc = LLMError(
            "API error",
            provider="openai",
            model="gpt-4",
            retries=3,
        )
        assert exc.provider == "openai"
        assert exc.model == "gpt-4"
        assert exc.retries == 3


class TestTimeoutError:
    """Tests for custom TimeoutError."""

    def test_timeout_info(self) -> None:
        """Test timeout error details."""
        exc = TimeoutError(
            "Execution timed out",
            timeout_seconds=5.0,
            operation="code_execution",
        )
        assert exc.timeout_seconds == 5.0
        assert exc.operation == "code_execution"
