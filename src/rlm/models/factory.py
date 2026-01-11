"""Factory for creating LLM instances.

This module provides a simple factory function for creating LLM instances
based on the provider name, making it easy to switch between providers.

Example:
    >>> from rlm.models.factory import create_llm
    >>> llm = create_llm("ollama", model="mistral:7b")
    >>> llm = create_llm("groq", api_key="...", model="llama-3.1-8b-instant")
"""

import os

from rlm.exceptions import ConfigurationError
from rlm.models.base import BaseLLM
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


# Environment variable names for each provider
API_KEY_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _get_api_key(provider: str, api_key: str | None) -> str | None:
    """Get API key from parameter or environment variable."""
    if api_key:
        return api_key
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def create_llm(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> BaseLLM:
    """Create an LLM instance based on provider.

    This factory function creates the appropriate LLM class based on
    the provider name, handling configuration and validation.

    Args:
        provider: LLM provider ("ollama", "groq", "openai", "anthropic")
        api_key: API key (required for cloud providers)
        model: Model name (uses provider default if not specified)
        base_url: Custom API URL (for Ollama or proxies)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Configured LLM instance

    Raises:
        ConfigurationError: If provider is unknown or configuration is invalid

    Example:
        >>> # Local inference with Ollama
        >>> llm = create_llm("ollama", model="mistral:7b")
        >>>
        >>> # Cloud inference with Groq
        >>> llm = create_llm("groq", api_key="gsk_...", model="llama-3.1-8b-instant")
        >>>
        >>> # OpenAI
        >>> llm = create_llm("openai", api_key="sk-...", model="gpt-4")
    """
    provider = provider.lower()

    # Auto-resolve API key from environment if not provided
    resolved_api_key = _get_api_key(provider, api_key)

    logger.info(f"Creating LLM: provider={provider}, model={model}, has_api_key={bool(resolved_api_key)}")

    if provider == "ollama":
        from rlm.models.ollama_llm import OllamaLLM

        return OllamaLLM(
            model=model or "mistral:7b",
            base_url=base_url or "http://localhost:11434/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "groq":
        if not resolved_api_key:
            raise ConfigurationError(
                "Groq API key is required",
                details={"provider": "groq", "hint": "Set GROQ_API_KEY environment variable"},
            )
        from rlm.models.groq_llm import GroqLLM

        return GroqLLM(
            api_key=resolved_api_key,
            model=model or "llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai":
        if not resolved_api_key:
            raise ConfigurationError(
                "OpenAI API key is required",
                details={"provider": "openai", "hint": "Set OPENAI_API_KEY environment variable"},
            )
        from rlm.models.openai_llm import OpenAILLM

        return OpenAILLM(
            api_key=resolved_api_key,
            model=model or "gpt-4",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "anthropic":
        if not resolved_api_key:
            raise ConfigurationError(
                "Anthropic API key is required",
                details={
                    "provider": "anthropic",
                    "hint": "Set ANTHROPIC_API_KEY environment variable",
                },
            )
        from rlm.models.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            api_key=resolved_api_key,
            model=model or "claude-3-sonnet-20240229",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ConfigurationError(
            f"Unknown LLM provider: {provider}",
            details={
                "provider": provider,
                "supported": ["ollama", "groq", "openai", "anthropic"],
            },
        )


def create_llm_from_config(config: "RLMConfig") -> BaseLLM:  # noqa: F821
    """Create an LLM instance from an RLMConfig object.

    Args:
        config: RLMConfig instance with LLM settings

    Returns:
        Configured LLM instance

    Example:
        >>> from rlm.config import RLMConfig
        >>> config = RLMConfig()
        >>> llm = create_llm_from_config(config)
    """
    return create_llm(
        provider=config.llm.provider,
        api_key=config.get_api_key(config.llm.provider) or config.llm.api_key,
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
