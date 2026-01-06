"""Safe code execution for the RLM system.

This module provides sandboxed code execution using RestrictedPython,
ensuring that LLM-generated code cannot perform dangerous operations.

Security is CRITICAL in this module. All code execution goes through
multiple validation layers before being run in a restricted environment.

Example:
    >>> from rlm.execution import SandboxEnvironment, CodeValidator
    >>> validator = CodeValidator()
    >>> if validator.validate(code):
    ...     sandbox = SandboxEnvironment(timeout=5)
    ...     result = await sandbox.execute(code, context_api)
"""

from rlm.execution.sandbox import SandboxEnvironment
from rlm.execution.validator import CodeValidator, ValidationResult

__all__ = [
    "SandboxEnvironment",
    "CodeValidator",
    "ValidationResult",
]
