"""Pytest configuration and fixtures for RLM tests."""

import asyncio
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from rlm.config import RLMConfig
from rlm.context import ContextManager, SemanticChunking
from rlm.execution import CodeValidator, SandboxEnvironment
from rlm.models.base import BaseLLM, LLMResponse


class MockLLM(BaseLLM):
    """Mock LLM for testing without API calls."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="mock-model")
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return mock response."""
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        response_text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        return LLMResponse(
            content=response_text,
            tokens_used=100,
            prompt_tokens=50,
            completion_tokens=50,
            model=self.model,
            finish_reason="stop",
        )

    async def generate_code(
        self,
        task_description: str,
        context_info: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return mock code."""
        return """
result = "Mock code result"
"""


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a mock LLM for testing."""
    return MockLLM()


@pytest.fixture
def config() -> RLMConfig:
    """Create test configuration."""
    return RLMConfig(
        max_recursion_depth=1,
        default_chunk_size=1000,
        chunking_strategy="semantic",
    )


@pytest.fixture
def context_manager() -> ContextManager:
    """Create a context manager for testing."""
    return ContextManager(
        strategy=SemanticChunking(),
        max_chunk_tokens=1000,
    )


@pytest.fixture
def sandbox() -> SandboxEnvironment:
    """Create a sandbox for testing."""
    return SandboxEnvironment(timeout=5)


@pytest.fixture
def validator() -> CodeValidator:
    """Create a code validator for testing."""
    return CodeValidator()


@pytest.fixture
def sample_context() -> str:
    """Create sample context for testing."""
    return """
# Sample Document

## Introduction

This is a sample document for testing the Recursive Language Model system.
It contains multiple sections and paragraphs to test chunking and processing.

## Main Content

The main content section contains detailed information about various topics.
Each paragraph provides context that can be processed recursively.

### Subsection A

This subsection covers topic A in detail. It includes several sentences
that provide important information about the subject matter.

### Subsection B

Topic B is equally important and contains complementary information.
The content here relates to what was discussed in subsection A.

## Conclusion

In conclusion, this document serves as a test fixture for the RLM system.
It demonstrates how documents with structure can be processed effectively.
""" * 5  # Repeat to make it larger


@pytest.fixture
def sample_code_context() -> str:
    """Create sample code context for testing."""
    return '''
def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total


def calculate_average(numbers: list[int]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0.0
    return calculate_sum(numbers) / len(numbers)


class Calculator:
    """A simple calculator class."""

    def __init__(self) -> None:
        self.history: list[float] = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(result)
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(result)
        return result
'''


# Async event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
