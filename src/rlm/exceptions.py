"""Custom exceptions for the RLM system.

This module defines the exception hierarchy for handling various error
conditions in the Recursive Language Model system.
"""

from typing import Any


class RLMException(Exception):
    """Base exception for all RLM-related errors.

    All custom exceptions in the RLM system inherit from this class,
    making it easy to catch any RLM-specific error.

    Args:
        message: Human-readable error message
        details: Optional dictionary with additional context

    Example:
        >>> raise RLMException("Something went wrong", details={"context": "test"})
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(RLMException):
    """Raised when configuration is invalid or incomplete.

    This exception is raised during initialization when required
    configuration values are missing or invalid.
    """

    pass


class MaxDepthExceededError(RLMException):
    """Raised when recursion depth limit is exceeded.

    This is a safety mechanism to prevent infinite recursion and
    excessive API costs.

    Args:
        current_depth: The depth at which the limit was hit
        max_depth: The configured maximum depth
    """

    def __init__(self, current_depth: int, max_depth: int) -> None:
        super().__init__(
            f"Maximum recursion depth exceeded: {current_depth} > {max_depth}",
            details={"current_depth": current_depth, "max_depth": max_depth},
        )
        self.current_depth = current_depth
        self.max_depth = max_depth


class CodeExecutionError(RLMException):
    """Raised when generated code execution fails.

    This can occur due to syntax errors, runtime errors, or
    security violations in the generated code.

    Args:
        message: Description of the error
        code: The code that failed to execute
        original_error: The underlying exception, if any
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        details = {}
        if code:
            # Truncate code for logging safety
            details["code_snippet"] = code[:500] + "..." if len(code) > 500 else code
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details=details)
        self.code = code
        self.original_error = original_error


class SecurityError(RLMException):
    """Raised when a security violation is detected.

    This is raised when generated code attempts forbidden operations
    such as file access, network calls, or dangerous imports.

    Args:
        message: Description of the security violation
        violation_type: Category of the violation (e.g., "forbidden_import")
        blocked_operation: The specific operation that was blocked
    """

    def __init__(
        self,
        message: str,
        violation_type: str | None = None,
        blocked_operation: str | None = None,
    ) -> None:
        details = {}
        if violation_type:
            details["violation_type"] = violation_type
        if blocked_operation:
            details["blocked_operation"] = blocked_operation
        super().__init__(message, details=details)
        self.violation_type = violation_type
        self.blocked_operation = blocked_operation


class LLMError(RLMException):
    """Raised when an LLM API call fails.

    This exception wraps errors from LLM providers (OpenAI, Anthropic, etc.)
    and includes retry information.

    Args:
        message: Description of the error
        provider: The LLM provider (e.g., "openai", "anthropic")
        model: The model that was called
        retries: Number of retries attempted
        original_error: The underlying API error
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        retries: int = 0,
        original_error: Exception | None = None,
    ) -> None:
        details = {
            "provider": provider,
            "model": model,
            "retries": retries,
        }
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details=details)
        self.provider = provider
        self.model = model
        self.retries = retries
        self.original_error = original_error


class ContextError(RLMException):
    """Raised when there's an issue with context management.

    This can occur when context is too large, corrupted, or
    when chunk operations fail.
    """

    pass


class ValidationError(RLMException):
    """Raised when input validation fails.

    This is raised when user inputs or generated content
    fails validation checks.
    """

    pass


class TimeoutError(RLMException):
    """Raised when an operation times out.

    This is separate from the built-in TimeoutError to provide
    RLM-specific context.

    Args:
        message: Description of what timed out
        timeout_seconds: The timeout value that was exceeded
        operation: The operation that timed out
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
    ) -> None:
        details = {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation
