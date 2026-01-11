"""Security tests for the sandbox environment.

CRITICAL: These tests verify that malicious code cannot escape the sandbox.
All tests should PASS for the system to be considered secure.
"""

import pytest

from rlm.exceptions import CodeExecutionError, SecurityError
from rlm.execution import CodeValidator, SandboxEnvironment


@pytest.fixture
def sandbox() -> SandboxEnvironment:
    """Create a sandbox for testing."""
    return SandboxEnvironment(timeout=2)


@pytest.fixture
def validator() -> CodeValidator:
    """Create a validator for testing."""
    return CodeValidator()


class TestMaliciousCodeBlocking:
    """Tests for blocking malicious code patterns."""

    # System access attempts
    @pytest.mark.parametrize(
        "code,description",
        [
            ("import os; os.system('ls')", "os.system command execution"),
            ("import subprocess; subprocess.run(['ls'])", "subprocess execution"),
            ("import shutil; shutil.rmtree('/')", "shutil file operations"),
            ("from os import *", "wildcard os import"),
            ("__import__('os').system('ls')", "dunder import bypass"),
        ],
    )
    def test_system_access_blocked(
        self, validator: CodeValidator, code: str, description: str
    ) -> None:
        """SECURITY: System access must be blocked."""
        result = validator.validate(code)
        assert not result.is_valid, f"Failed to block: {description}"

    # File access attempts
    @pytest.mark.parametrize(
        "code,description",
        [
            ("open('/etc/passwd').read()", "file read"),
            ("open('/tmp/test', 'w').write('x')", "file write"),
            ("import pathlib; pathlib.Path('/etc/passwd')", "pathlib access"),
        ],
    )
    def test_file_access_blocked(
        self, validator: CodeValidator, code: str, description: str
    ) -> None:
        """SECURITY: File access must be blocked."""
        result = validator.validate(code)
        assert not result.is_valid, f"Failed to block: {description}"

    # Network access attempts
    @pytest.mark.parametrize(
        "code,description",
        [
            ("import socket; socket.socket()", "socket creation"),
            ("import urllib.request", "urllib import"),
            ("import http.client", "http client import"),
            ("import ftplib", "ftp library import"),
            ("import smtplib", "smtp library import"),
        ],
    )
    def test_network_access_blocked(
        self, validator: CodeValidator, code: str, description: str
    ) -> None:
        """SECURITY: Network access must be blocked."""
        result = validator.validate(code)
        assert not result.is_valid, f"Failed to block: {description}"

    # Code execution/eval attempts
    @pytest.mark.parametrize(
        "code,description",
        [
            ("eval('1+1')", "eval function"),
            ("exec('x=1')", "exec function"),
            ("compile('x=1', '', 'exec')", "compile function"),
        ],
    )
    def test_dynamic_execution_blocked(
        self, validator: CodeValidator, code: str, description: str
    ) -> None:
        """SECURITY: Dynamic code execution must be blocked."""
        result = validator.validate(code)
        assert not result.is_valid, f"Failed to block: {description}"

    # Python internals access
    @pytest.mark.parametrize(
        "code,description",
        [
            ("().__class__.__bases__[0].__subclasses__()", "class hierarchy access"),
            ("globals()['__builtins__']", "builtins access via globals"),
            ("x.__class__.__mro__", "MRO access"),
            ("getattr(x, '__class__')", "getattr dunder access"),
        ],
    )
    def test_internals_access_blocked(
        self, validator: CodeValidator, code: str, description: str
    ) -> None:
        """SECURITY: Python internals access must be blocked."""
        result = validator.validate(code)
        assert not result.is_valid, f"Failed to block: {description}"

    # Pickle deserialization attacks
    @pytest.mark.parametrize(
        "code,description",
        [
            ("import pickle", "pickle import"),
            ("import marshal", "marshal import"),
            ("import shelve", "shelve import"),
        ],
    )
    def test_serialization_blocked(
        self, validator: CodeValidator, code: str, description: str
    ) -> None:
        """SECURITY: Dangerous serialization must be blocked."""
        result = validator.validate(code)
        assert not result.is_valid, f"Failed to block: {description}"


class TestSandboxExecution:
    """Tests for sandbox execution security."""

    @pytest.mark.asyncio
    async def test_safe_code_executes(self, sandbox: SandboxEnvironment) -> None:
        """Test that safe code executes correctly."""
        code = """
result = 1 + 2
"""
        # This should execute without error
        context_api = {}
        result = await sandbox.execute(code, context_api)
        assert result == 3

    @pytest.mark.asyncio
    async def test_timeout_enforced(self, sandbox: SandboxEnvironment) -> None:
        """SECURITY: Infinite loops must be stopped by timeout."""
        code = """
while True:
    pass
"""
        from rlm.exceptions import TimeoutError

        with pytest.raises((TimeoutError, Exception)):
            await sandbox.execute(code, {})

    @pytest.mark.asyncio
    async def test_malicious_code_rejected(
        self, sandbox: SandboxEnvironment
    ) -> None:
        """SECURITY: Malicious code must be rejected."""
        code = "import os"

        with pytest.raises((SecurityError, Exception)):
            await sandbox.execute(code, {})


class TestResourceExhaustion:
    """Tests for resource exhaustion attacks."""

    @pytest.mark.asyncio
    async def test_memory_bomb_blocked(self, sandbox: SandboxEnvironment) -> None:
        """SECURITY: Memory exhaustion attempts must be handled."""
        code = """
x = [0] * (10**9)  # Try to allocate huge list
"""
        # Should either timeout or raise memory error
        try:
            await sandbox.execute(code, {})
            pytest.fail("Memory bomb should have been blocked")
        except Exception:
            pass  # Expected to fail

    @pytest.mark.asyncio
    async def test_recursion_bomb_blocked(
        self, sandbox: SandboxEnvironment
    ) -> None:
        """SECURITY: Recursive stack overflow must be handled."""
        code = """
def f():
    return f()
f()
"""
        # Should either timeout or raise recursion error
        try:
            await sandbox.execute(code, {})
            pytest.fail("Recursion bomb should have been blocked")
        except Exception:
            pass  # Expected to fail
