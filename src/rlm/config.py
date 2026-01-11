"""Configuration management for the RLM system.

This module provides centralized configuration using Pydantic settings,
supporting environment variables, .env files, and programmatic configuration.

Example:
    >>> from rlm.config import RLMConfig
    >>> config = RLMConfig()  # Loads from environment
    >>> config.max_recursion_depth
    10
"""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuration for LLM providers.

    Attributes:
        provider: The LLM provider to use
        model: The specific model name
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        api_key: API key for cloud providers (not needed for Ollama)
        base_url: Custom API base URL (for Ollama or proxies)
    """

    model_config = SettingsConfigDict(
        env_prefix="RLM_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["ollama", "groq", "openai", "anthropic"] = Field(
        default="ollama",
        description="LLM provider to use",
    )
    model: str = Field(
        default="mistral:7b",
        description="Model name/identifier",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=2000,
        ge=1,
        le=32000,
        description="Maximum tokens in response",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for cloud providers",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom API base URL",
    )


class ExecutionConfig(BaseSettings):
    """Configuration for code execution sandbox.

    Attributes:
        timeout: Maximum execution time in seconds
        memory_limit_mb: Maximum memory usage in megabytes
        enable_docker: Use Docker for stronger isolation
        allowed_builtins: Whitelist of allowed Python builtins
    """

    model_config = SettingsConfigDict(
        env_prefix="RLM_EXEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    timeout: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Execution timeout in seconds",
    )
    memory_limit_mb: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Memory limit in MB",
    )
    enable_docker: bool = Field(
        default=False,
        description="Use Docker for sandbox isolation",
    )


class RLMConfig(BaseSettings):
    """Main configuration for the RLM system.

    This is the primary configuration class that aggregates all settings
    for the Recursive Language Model system.

    Attributes:
        max_recursion_depth: Maximum allowed recursion depth (paper uses 1)
        default_chunk_size: Default chunk size in tokens
        chunking_strategy: Default chunking strategy
        enable_caching: Enable LLM response caching
        cache_ttl: Cache time-to-live in seconds
        log_level: Logging level
        llm: LLM provider configuration
        execution: Code execution configuration

    Example:
        >>> config = RLMConfig(max_recursion_depth=1)  # Match paper
        >>> config.llm.provider
        'ollama'
    """

    model_config = SettingsConfigDict(
        env_prefix="RLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core settings
    max_recursion_depth: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Maximum recursion depth (paper recommends 1)",
    )
    default_chunk_size: int = Field(
        default=4000,
        ge=100,
        le=100000,
        description="Default chunk size in tokens",
    )
    chunking_strategy: Literal["fixed", "semantic", "hierarchical", "adaptive"] = Field(
        default="semantic",
        description="Default chunking strategy",
    )

    # Caching
    enable_caching: bool = Field(
        default=True,
        description="Enable LLM response caching",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # API keys (loaded from environment)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    @field_validator("max_recursion_depth")
    @classmethod
    def warn_high_depth(cls, v: int) -> int:
        """Warn if recursion depth is higher than paper recommendation."""
        if v > 1:
            import warnings

            warnings.warn(
                f"Recursion depth {v} exceeds paper recommendation of 1. "
                "Higher depths may increase costs significantly.",
                UserWarning,
                stacklevel=2,
            )
        return v

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider.

        Args:
            provider: The provider name

        Returns:
            The API key or None if not configured
        """
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "ollama": None,  # Ollama doesn't need a key
        }
        return key_map.get(provider)
