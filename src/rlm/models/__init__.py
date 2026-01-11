"""LLM model integrations for the RLM system.

This module provides abstract interfaces and concrete implementations
for various LLM providers including Ollama (local), Groq, OpenAI, and Anthropic.

Example:
    >>> from rlm.models import create_llm, BaseLLM
    >>> llm = create_llm("ollama", model="mistral:7b")
    >>> response = await llm.generate("Hello, world!")
"""

from rlm.models.base import BaseLLM, LLMResponse
from rlm.models.factory import create_llm
from rlm.models.tiers import ModelSpec, ModelTier, ModelTiers

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "create_llm",
    "ModelTiers",
    "ModelTier",
    "ModelSpec",
]
