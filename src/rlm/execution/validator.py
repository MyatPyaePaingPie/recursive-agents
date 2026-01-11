"""Code validation for security.

This module provides AST-based code validation to detect potentially
dangerous operations before code execution.

SECURITY: This is a critical security component. Any changes must be
carefully reviewed for security implications.
"""

import ast
from dataclasses import dataclass, field
from typing import Any

from rlm.utils.logging import get_logger

logger = get_logger(__name__)


# Forbidden names that should never be used
FORBIDDEN_NAMES = frozenset({
    # Dangerous builtins
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "file",
    "input",
    "raw_input",
    # Attribute access that could be exploited
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__builtins__",
    "__dict__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    # Module-level
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    # System interaction
    "exit",
    "quit",
    "help",
    "copyright",
    "credits",
    "license",
})

# Forbidden imports
FORBIDDEN_IMPORTS = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
    "marshal",
    "shelve",
    "dbm",
    "sqlite3",
    "importlib",
    "builtins",
    "__builtins__",
    "ctypes",
    "multiprocessing",
    "threading",
    "concurrent",
    "asyncio",
    "signal",
    "resource",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "mmap",
})

# Allowed safe modules (whitelist approach)
ALLOWED_MODULES = frozenset({
    "math",
    "re",
    "json",
    "collections",
    "itertools",
    "functools",
    "operator",
    "string",
    "textwrap",
    "datetime",
    "time",
    "random",
    "statistics",
    "copy",
    "typing",
})


@dataclass
class ValidationResult:
    """Result of code validation.

    Attributes:
        is_valid: Whether the code passed validation
        errors: List of validation error messages
        warnings: List of validation warnings
        ast_tree: Parsed AST if successful
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ast_tree: ast.AST | None = None


class CodeValidator:
    """Validates code for safe execution.

    This validator performs multiple security checks:
    1. Syntax validation (AST parsing)
    2. Forbidden name detection
    3. Dangerous import detection
    4. Attribute access validation
    5. Complexity analysis

    Example:
        >>> validator = CodeValidator()
        >>> result = validator.validate("x = 1 + 2")
        >>> result.is_valid
        True
        >>> result = validator.validate("import os")
        >>> result.is_valid
        False
    """

    def __init__(
        self,
        forbidden_names: frozenset[str] | None = None,
        forbidden_imports: frozenset[str] | None = None,
        allowed_modules: frozenset[str] | None = None,
        max_complexity: int = 100,
    ) -> None:
        """Initialize the validator.

        Args:
            forbidden_names: Override default forbidden names
            forbidden_imports: Override default forbidden imports
            allowed_modules: Override default allowed modules
            max_complexity: Maximum allowed code complexity
        """
        self.forbidden_names = forbidden_names or FORBIDDEN_NAMES
        self.forbidden_imports = forbidden_imports or FORBIDDEN_IMPORTS
        self.allowed_modules = allowed_modules or ALLOWED_MODULES
        self.max_complexity = max_complexity

    def validate(self, code: str) -> ValidationResult:
        """Validate code for safe execution.

        Args:
            code: Python code to validate

        Returns:
            ValidationResult with validation status and details
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Step 1: Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Syntax error: {e}"],
            )

        # Step 2: Check for forbidden names
        name_errors = self._check_forbidden_names(tree)
        errors.extend(name_errors)

        # Step 3: Check imports
        import_errors = self._check_imports(tree)
        errors.extend(import_errors)

        # Step 4: Check attribute access
        attr_errors = self._check_attribute_access(tree)
        errors.extend(attr_errors)

        # Step 5: Check complexity
        complexity_warnings = self._check_complexity(tree)
        warnings.extend(complexity_warnings)

        # Step 6: Check for dangerous patterns
        pattern_errors = self._check_dangerous_patterns(code)
        errors.extend(pattern_errors)

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(f"Code validation failed: {errors}")
        elif warnings:
            logger.debug(f"Code validation warnings: {warnings}")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            ast_tree=tree if is_valid else None,
        )

    def _check_forbidden_names(self, tree: ast.AST) -> list[str]:
        """Check for forbidden name usage."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in self.forbidden_names:
                    errors.append(f"Forbidden name: '{node.id}'")

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.forbidden_names:
                        errors.append(f"Forbidden function call: '{node.func.id}'")

        return errors

    def _check_imports(self, tree: ast.AST) -> list[str]:
        """Check for forbidden imports."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.forbidden_imports:
                        errors.append(f"Forbidden import: '{alias.name}'")
                    elif module not in self.allowed_modules:
                        errors.append(f"Disallowed import: '{alias.name}'")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.forbidden_imports:
                        errors.append(f"Forbidden import from: '{node.module}'")
                    elif module not in self.allowed_modules:
                        errors.append(f"Disallowed import from: '{node.module}'")

        return errors

    def _check_attribute_access(self, tree: ast.AST) -> list[str]:
        """Check for dangerous attribute access."""
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Check for dunder attributes
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    errors.append(f"Forbidden dunder attribute: '{node.attr}'")
                elif node.attr in self.forbidden_names:
                    errors.append(f"Forbidden attribute: '{node.attr}'")

        return errors

    def _check_complexity(self, tree: ast.AST) -> list[str]:
        """Check code complexity."""
        warnings = []

        # Count nodes as simple complexity measure
        node_count = sum(1 for _ in ast.walk(tree))

        if node_count > self.max_complexity:
            warnings.append(
                f"High complexity: {node_count} AST nodes "
                f"(max recommended: {self.max_complexity})"
            )

        # Count loops (potential infinite loops)
        loop_count = sum(
            1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))
        )
        if loop_count > 10:
            warnings.append(f"Many loops detected: {loop_count}")

        return warnings

    def _check_dangerous_patterns(self, code: str) -> list[str]:
        """Check for dangerous string patterns."""
        errors = []

        dangerous_patterns = [
            ("lambda.*__", "Lambda with dunder access"),
            (r"\(\s*\)\s*\[", "Empty tuple subscript (potential exploit)"),
            ("getattr.*__", "getattr with dunder"),
            ("setattr.*__", "setattr with dunder"),
            (r"\\x[0-9a-fA-F]{2}", "Hex escape sequences"),
            (r"\\u[0-9a-fA-F]{4}", "Unicode escape sequences"),
        ]

        import re

        for pattern, description in dangerous_patterns:
            if re.search(pattern, code):
                errors.append(f"Dangerous pattern detected: {description}")

        return errors


class SecurityAuditor:
    """Audits code execution for security logging.

    This class logs all code executions for security auditing purposes.
    """

    def __init__(self) -> None:
        """Initialize the security auditor."""
        self.audit_log: list[dict[str, Any]] = []

    def log_execution(
        self,
        code: str,
        validation_result: ValidationResult,
        execution_result: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Log a code execution attempt.

        Args:
            code: The code that was (attempted to be) executed
            validation_result: Result of code validation
            execution_result: Result of execution if successful
            error: Error if execution failed
        """
        from datetime import datetime

        entry = {
            "timestamp": datetime.now().isoformat(),
            "code_length": len(code),
            "code_hash": hash(code),
            "validation_passed": validation_result.is_valid,
            "validation_errors": validation_result.errors,
            "execution_success": error is None,
            "error": str(error) if error else None,
        }

        self.audit_log.append(entry)
        logger.info(
            f"Security audit: validation={validation_result.is_valid}, "
            f"execution={'success' if error is None else 'failed'}"
        )

    def get_audit_summary(self) -> dict[str, Any]:
        """Get summary of audit log."""
        if not self.audit_log:
            return {"total": 0}

        return {
            "total": len(self.audit_log),
            "validation_passed": sum(
                1 for e in self.audit_log if e["validation_passed"]
            ),
            "execution_success": sum(
                1 for e in self.audit_log if e["execution_success"]
            ),
        }
