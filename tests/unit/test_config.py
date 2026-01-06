"""Tests for configuration module."""

import os

import pytest

from rlm.config import ExecutionConfig, LLMConfig, RLMConfig


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()

        assert config.provider == "ollama"
        assert config.model == "mistral:7b"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.api_key is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.5,
            max_tokens=4000,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 4000


class TestExecutionConfig:
    """Tests for ExecutionConfig."""

    def test_default_values(self) -> None:
        """Test default execution config."""
        config = ExecutionConfig()

        assert config.timeout == 5
        assert config.memory_limit_mb == 512
        assert config.enable_docker is False

    def test_timeout_bounds(self) -> None:
        """Test timeout validation bounds."""
        # Valid timeout
        config = ExecutionConfig(timeout=30)
        assert config.timeout == 30

        # Invalid timeout should raise
        with pytest.raises(ValueError):
            ExecutionConfig(timeout=0)

        with pytest.raises(ValueError):
            ExecutionConfig(timeout=100)  # > max


class TestRLMConfig:
    """Tests for RLMConfig."""

    def test_default_values(self) -> None:
        """Test default RLM config."""
        config = RLMConfig()

        # Paper recommends depth of 1
        assert config.max_recursion_depth == 1
        assert config.default_chunk_size == 4000
        assert config.chunking_strategy == "semantic"
        assert config.enable_caching is True

    def test_nested_configs(self) -> None:
        """Test nested LLM and execution configs."""
        config = RLMConfig()

        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.execution, ExecutionConfig)
        assert config.llm.provider == "ollama"
        assert config.execution.timeout == 5

    def test_get_api_key(self) -> None:
        """Test API key retrieval."""
        # Set env var for test
        os.environ["OPENAI_API_KEY"] = "test-key"

        try:
            config = RLMConfig()
            assert config.get_api_key("openai") == "test-key"
            assert config.get_api_key("ollama") is None
            assert config.get_api_key("unknown") is None
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_high_depth_warning(self) -> None:
        """Test warning for high recursion depth."""
        with pytest.warns(UserWarning, match="exceeds paper recommendation"):
            RLMConfig(max_recursion_depth=5)

    def test_depth_bounds(self) -> None:
        """Test recursion depth bounds."""
        # Valid depths
        RLMConfig(max_recursion_depth=1)
        RLMConfig(max_recursion_depth=50)

        # Invalid depths
        with pytest.raises(ValueError):
            RLMConfig(max_recursion_depth=0)

        with pytest.raises(ValueError):
            RLMConfig(max_recursion_depth=51)
