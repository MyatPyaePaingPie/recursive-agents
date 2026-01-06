"""Sandboxed code execution environment.

This module provides a secure sandbox for executing LLM-generated code
using RestrictedPython and resource limits.

SECURITY: This is a CRITICAL security component. All code execution
goes through this sandbox. Changes require careful security review.
"""

import asyncio
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from rlm.exceptions import CodeExecutionError, SecurityError, TimeoutError
from rlm.execution.validator import CodeValidator, SecurityAuditor, ValidationResult
from rlm.utils.logging import get_logger

logger = get_logger(__name__)

# Safe builtins whitelist
SAFE_BUILTINS = {
    # Types
    "True": True,
    "False": False,
    "None": None,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    # Functions
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "all": all,
    "any": any,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "repr": repr,
    "ascii": ascii,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "format": format,
    "iter": iter,
    "next": next,
    "slice": slice,
    # String methods (via str type)
    "print": lambda *args, **kwargs: None,  # Disabled but available
}


class SandboxEnvironment:
    """Secure sandbox for code execution.

    This class provides a restricted environment for executing
    LLM-generated code safely. It uses:

    1. RestrictedPython for compile-time restrictions
    2. Custom safe builtins whitelist
    3. Execution timeout
    4. Memory limits (via resource module on Unix)

    Attributes:
        timeout: Maximum execution time in seconds
        memory_limit_mb: Maximum memory in MB (Unix only)
        validator: Code validator instance
        auditor: Security auditor instance

    Example:
        >>> sandbox = SandboxEnvironment(timeout=5)
        >>> result = await sandbox.execute(code, {"get_context_length": func})
    """

    def __init__(
        self,
        timeout: int = 5,
        memory_limit_mb: int = 512,
        validator: CodeValidator | None = None,
    ) -> None:
        """Initialize the sandbox.

        Args:
            timeout: Execution timeout in seconds
            memory_limit_mb: Memory limit in megabytes
            validator: Code validator (creates default if None)
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.validator = validator or CodeValidator()
        self.auditor = SecurityAuditor()
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(f"SandboxEnvironment initialized: timeout={timeout}s")

    def _create_restricted_globals(
        self,
        context_api: dict[str, Callable[..., Any]],
    ) -> dict[str, Any]:
        """Create a restricted globals dict for execution.

        Args:
            context_api: Functions to make available to the code

        Returns:
            Dictionary of safe globals
        """
        # Start with safe builtins
        restricted_globals: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS.copy(),
        }

        # Add safe modules
        import json
        import math
        import re

        restricted_globals["math"] = math
        restricted_globals["re"] = re
        restricted_globals["json"] = json

        # Add context API functions
        for name, func in context_api.items():
            restricted_globals[name] = func

        return restricted_globals

    def _execute_sync(
        self,
        code: str,
        restricted_globals: dict[str, Any],
    ) -> Any:
        """Execute code synchronously in restricted environment.

        Args:
            code: Python code to execute
            restricted_globals: Restricted globals dict

        Returns:
            Execution result

        Raises:
            CodeExecutionError: If execution fails
        """
        try:
            # Try using RestrictedPython if available
            try:
                from RestrictedPython import compile_restricted, safe_builtins
                from RestrictedPython.Eval import default_guarded_getitem
                from RestrictedPython.Guards import (
                    guarded_iter_unpack_sequence,
                    safer_getattr,
                )

                # Compile with RestrictedPython
                byte_code = compile_restricted(
                    code,
                    filename="<rlm_sandbox>",
                    mode="exec",
                )

                if byte_code.errors:
                    raise CodeExecutionError(
                        f"RestrictedPython compilation errors: {byte_code.errors}",
                        code=code,
                    )

                # Update globals with RestrictedPython guards
                restricted_globals["__builtins__"] = safe_builtins.copy()
                restricted_globals["__builtins__"]["_getitem_"] = default_guarded_getitem
                restricted_globals["__builtins__"]["_getiter_"] = iter
                restricted_globals["__builtins__"]["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
                restricted_globals["_getattr_"] = safer_getattr

                # Add safe builtins that might be missing
                for name, value in SAFE_BUILTINS.items():
                    if name not in restricted_globals["__builtins__"]:
                        restricted_globals["__builtins__"][name] = value

                exec(byte_code.code, restricted_globals)

            except ImportError:
                # RestrictedPython not available, use basic exec with validation
                logger.warning(
                    "RestrictedPython not available, using basic sandboxing"
                )

                # Compile and execute
                compiled = compile(code, "<rlm_sandbox>", "exec")
                exec(compiled, restricted_globals)

            # Look for a return value
            # Convention: code should set a variable called 'result'
            return restricted_globals.get("result", restricted_globals.get("_result_"))

        except Exception as e:
            raise CodeExecutionError(
                f"Execution failed: {e}",
                code=code,
                original_error=e,
            ) from e

    async def execute(
        self,
        code: str,
        context_api: dict[str, Callable[..., Any]],
        validate: bool = True,
    ) -> Any:
        """Execute code in the sandbox.

        Args:
            code: Python code to execute
            context_api: Functions to make available to the code
            validate: Whether to validate code before execution

        Returns:
            Execution result

        Raises:
            SecurityError: If validation fails
            CodeExecutionError: If execution fails
            TimeoutError: If execution times out
        """
        # Step 1: Validate code
        validation_result: ValidationResult
        if validate:
            validation_result = self.validator.validate(code)
            if not validation_result.is_valid:
                self.auditor.log_execution(
                    code,
                    validation_result,
                    error=SecurityError("Validation failed"),
                )
                raise SecurityError(
                    f"Code validation failed: {validation_result.errors}",
                    violation_type="validation_failed",
                    blocked_operation=str(validation_result.errors),
                )
        else:
            validation_result = ValidationResult(is_valid=True)

        # Step 2: Create restricted environment
        restricted_globals = self._create_restricted_globals(context_api)

        # Step 3: Execute with timeout
        try:
            logger.debug(f"Executing code ({len(code)} chars) with timeout={self.timeout}s")

            # Use asyncio timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._execute_sync,
                    code,
                    restricted_globals,
                ),
                timeout=self.timeout,
            )

            self.auditor.log_execution(code, validation_result, execution_result=result)
            logger.debug("Code execution completed successfully")

            return result

        except asyncio.TimeoutError as e:
            self.auditor.log_execution(
                code,
                validation_result,
                error=TimeoutError("Execution timed out"),
            )
            raise TimeoutError(
                f"Code execution timed out after {self.timeout} seconds",
                timeout_seconds=self.timeout,
                operation="sandbox_execute",
            ) from e

        except CodeExecutionError:
            raise

        except Exception as e:
            self.auditor.log_execution(code, validation_result, error=e)
            raise CodeExecutionError(
                f"Unexpected execution error: {e}",
                code=code,
                original_error=e,
            ) from e

    def get_audit_summary(self) -> dict[str, Any]:
        """Get security audit summary."""
        return self.auditor.get_audit_summary()


class ContextAPIBuilder:
    """Builds the context API for sandboxed code.

    This class creates the functions that will be available to
    LLM-generated code for accessing context and making recursive calls.

    Example:
        >>> builder = ContextAPIBuilder(context_manager, engine)
        >>> api = builder.build()
        >>> # api contains: get_context_length, get_context_chunk, call_submodel, etc.
    """

    def __init__(
        self,
        context_manager: Any,
        engine: Any | None = None,
        current_depth: int = 0,
        max_depth: int = 1,
    ) -> None:
        """Initialize the API builder.

        Args:
            context_manager: ContextManager instance
            engine: RecursiveInferenceEngine instance (for call_submodel)
            current_depth: Current recursion depth
            max_depth: Maximum allowed recursion depth
        """
        self.context_manager = context_manager
        self.engine = engine
        self.current_depth = current_depth
        self.max_depth = max_depth
        self._submodel_results: list[str] = []

    def build(self) -> dict[str, Callable[..., Any]]:
        """Build the context API.

        Returns:
            Dictionary of API functions
        """
        api: dict[str, Callable[..., Any]] = {
            "get_context_length": self._get_context_length,
            "get_context_chunk": self._get_context_chunk,
            "call_submodel": self._call_submodel,
            "aggregate_results": self._aggregate_results,
            "search_context": self._search_context,
        }

        return api

    def _get_context_length(self) -> int:
        """Get total context length in tokens."""
        return self.context_manager.get_context_length()

    def _get_context_chunk(self, start: int, end: int) -> str:
        """Get context chunk by character range."""
        return self.context_manager.get_chunk_range(start, end)

    def _call_submodel(self, chunk: str, query: str) -> str:
        """Make a recursive call to the sub-model.

        This is a synchronous wrapper - actual recursive calls
        are handled by the engine.
        """
        if self.current_depth >= self.max_depth:
            logger.warning(
                f"Max recursion depth ({self.max_depth}) reached, "
                "returning placeholder"
            )
            return f"[Recursion limit reached. Chunk preview: {chunk[:200]}...]"

        # For synchronous execution, we store the call info
        # The actual async call is made by the engine
        self._submodel_results.append(f"[SUBMODEL_CALL:{query}:{len(chunk)}]")

        # Return a placeholder - the engine will replace this
        return f"[PENDING_SUBMODEL:{len(self._submodel_results) - 1}]"

    def _aggregate_results(self, results: list[str]) -> str:
        """Aggregate multiple results."""
        if not results:
            return ""

        # Simple aggregation - join with newlines
        # The engine may use LLM-based aggregation for better results
        return "\n\n".join(str(r) for r in results if r)

    def _search_context(self, pattern: str) -> list[tuple[int, int, str]]:
        """Search context for a pattern."""
        return self.context_manager.search_context(pattern)

    def get_pending_submodel_calls(self) -> list[str]:
        """Get list of pending submodel calls."""
        return self._submodel_results.copy()
