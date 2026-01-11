"""Model tier system for cost/performance optimization.

This module provides a tiered model system that allows using:
- Lightweight models for code generation (fast, cheap)
- Heavyweight models for reasoning (accurate, expensive)

The paper suggests using cheaper models for code generation and
more capable models for actual reasoning tasks.

Model Tiers:
    - LIGHTWEIGHT: Fast, cheap models for simple tasks
    - STANDARD: Balanced models for general use
    - HEAVYWEIGHT: Most capable models for complex reasoning

Example:
    >>> tiers = ModelTiers.from_provider("ollama")
    >>> code_llm = tiers.get_lightweight()
    >>> reasoning_llm = tiers.get_heavyweight()
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rlm.models.base import BaseLLM
from rlm.models.factory import create_llm


class ModelTier(Enum):
    """Model capability tiers."""

    LIGHTWEIGHT = "lightweight"  # Fast, cheap - for code generation
    STANDARD = "standard"        # Balanced - general purpose
    HEAVYWEIGHT = "heavyweight"  # Most capable - for reasoning


@dataclass
class ModelSpec:
    """Specification for a model."""

    provider: str
    model: str
    tier: ModelTier
    description: str
    context_window: int
    cost_per_1k_tokens: float  # Approximate cost in USD
    tokens_per_second: float   # Approximate speed


# Model catalog organized by provider and tier
MODEL_CATALOG: dict[str, dict[ModelTier, ModelSpec]] = {
    "ollama": {
        ModelTier.LIGHTWEIGHT: ModelSpec(
            provider="ollama",
            model="phi3:3.8b",
            tier=ModelTier.LIGHTWEIGHT,
            description="Microsoft Phi-3 Mini - fast, good at code",
            context_window=4096,
            cost_per_1k_tokens=0.0,  # Free (local)
            tokens_per_second=50.0,
        ),
        ModelTier.STANDARD: ModelSpec(
            provider="ollama",
            model="mistral:7b",
            tier=ModelTier.STANDARD,
            description="Mistral 7B - balanced performance",
            context_window=8192,
            cost_per_1k_tokens=0.0,
            tokens_per_second=30.0,
        ),
        ModelTier.HEAVYWEIGHT: ModelSpec(
            provider="ollama",
            model="llama3:8b",
            tier=ModelTier.HEAVYWEIGHT,
            description="Llama 3 8B - best local reasoning",
            context_window=8192,
            cost_per_1k_tokens=0.0,
            tokens_per_second=25.0,
        ),
    },
    "groq": {
        ModelTier.LIGHTWEIGHT: ModelSpec(
            provider="groq",
            model="llama-3.1-8b-instant",
            tier=ModelTier.LIGHTWEIGHT,
            description="Llama 3.1 8B on Groq - extremely fast",
            context_window=8192,
            cost_per_1k_tokens=0.0001,
            tokens_per_second=300.0,  # Groq is very fast
        ),
        ModelTier.STANDARD: ModelSpec(
            provider="groq",
            model="llama-3.3-70b-versatile",
            tier=ModelTier.STANDARD,
            description="Llama 3.3 70B - powerful and fast",
            context_window=8192,
            cost_per_1k_tokens=0.0005,
            tokens_per_second=150.0,
        ),
        ModelTier.HEAVYWEIGHT: ModelSpec(
            provider="groq",
            model="llama-3.3-70b-versatile",
            tier=ModelTier.HEAVYWEIGHT,
            description="Llama 3.3 70B - most capable on Groq",
            context_window=8192,
            cost_per_1k_tokens=0.0005,
            tokens_per_second=150.0,
        ),
    },
    "openai": {
        ModelTier.LIGHTWEIGHT: ModelSpec(
            provider="openai",
            model="gpt-3.5-turbo",
            tier=ModelTier.LIGHTWEIGHT,
            description="GPT-3.5 Turbo - fast and cheap",
            context_window=16385,
            cost_per_1k_tokens=0.0015,
            tokens_per_second=80.0,
        ),
        ModelTier.STANDARD: ModelSpec(
            provider="openai",
            model="gpt-4-turbo",
            tier=ModelTier.STANDARD,
            description="GPT-4 Turbo - fast GPT-4",
            context_window=128000,
            cost_per_1k_tokens=0.01,
            tokens_per_second=40.0,
        ),
        ModelTier.HEAVYWEIGHT: ModelSpec(
            provider="openai",
            model="gpt-4",
            tier=ModelTier.HEAVYWEIGHT,
            description="GPT-4 - most capable",
            context_window=8192,
            cost_per_1k_tokens=0.03,
            tokens_per_second=20.0,
        ),
    },
    "anthropic": {
        ModelTier.LIGHTWEIGHT: ModelSpec(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            tier=ModelTier.LIGHTWEIGHT,
            description="Claude 3 Haiku - fast and cheap",
            context_window=200000,
            cost_per_1k_tokens=0.00025,
            tokens_per_second=100.0,
        ),
        ModelTier.STANDARD: ModelSpec(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            tier=ModelTier.STANDARD,
            description="Claude 3 Sonnet - balanced",
            context_window=200000,
            cost_per_1k_tokens=0.003,
            tokens_per_second=50.0,
        ),
        ModelTier.HEAVYWEIGHT: ModelSpec(
            provider="anthropic",
            model="claude-3-opus-20240229",
            tier=ModelTier.HEAVYWEIGHT,
            description="Claude 3 Opus - most capable",
            context_window=200000,
            cost_per_1k_tokens=0.015,
            tokens_per_second=25.0,
        ),
    },
}


class ModelTiers:
    """Manager for tiered model access.

    Provides easy access to lightweight, standard, and heavyweight
    models from the same provider.

    Example:
        >>> tiers = ModelTiers("ollama")
        >>> code_llm = tiers.get_lightweight()
        >>> reasoning_llm = tiers.get_heavyweight()
        >>>
        >>> # Or create both at once
        >>> code_llm, reasoning_llm = tiers.get_pair()
    """

    def __init__(
        self,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize model tiers for a provider.

        Args:
            provider: LLM provider name
            api_key: API key (for cloud providers)
            base_url: Custom base URL
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url

        if self.provider not in MODEL_CATALOG:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(MODEL_CATALOG.keys())}"
            )

        self.specs = MODEL_CATALOG[self.provider]

    def get_spec(self, tier: ModelTier) -> ModelSpec:
        """Get the model specification for a tier."""
        return self.specs[tier]

    def get_model(self, tier: ModelTier) -> BaseLLM:
        """Create an LLM instance for the specified tier.

        Args:
            tier: The model tier to use

        Returns:
            Configured LLM instance
        """
        spec = self.specs[tier]
        return create_llm(
            provider=spec.provider,
            api_key=self.api_key,
            model=spec.model,
            base_url=self.base_url,
        )

    def get_lightweight(self) -> BaseLLM:
        """Get the lightweight model (for code generation)."""
        return self.get_model(ModelTier.LIGHTWEIGHT)

    def get_standard(self) -> BaseLLM:
        """Get the standard model (balanced)."""
        return self.get_model(ModelTier.STANDARD)

    def get_heavyweight(self) -> BaseLLM:
        """Get the heavyweight model (for reasoning)."""
        return self.get_model(ModelTier.HEAVYWEIGHT)

    def get_pair(self) -> tuple[BaseLLM, BaseLLM]:
        """Get a lightweight/heavyweight pair for RLM.

        Returns:
            Tuple of (code_llm, reasoning_llm)
        """
        return self.get_lightweight(), self.get_heavyweight()

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        tier: ModelTier,
    ) -> float:
        """Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tier: Model tier

        Returns:
            Estimated cost in USD
        """
        spec = self.specs[tier]
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * spec.cost_per_1k_tokens

    def estimate_time(
        self,
        output_tokens: int,
        tier: ModelTier,
    ) -> float:
        """Estimate generation time.

        Args:
            output_tokens: Number of tokens to generate
            tier: Model tier

        Returns:
            Estimated time in seconds
        """
        spec = self.specs[tier]
        return output_tokens / spec.tokens_per_second

    def print_catalog(self) -> None:
        """Print the model catalog for this provider."""
        print(f"\n{'=' * 60}")
        print(f"Model Catalog: {self.provider.upper()}")
        print(f"{'=' * 60}\n")

        for tier in ModelTier:
            spec = self.specs[tier]
            print(f"[{tier.value.upper()}] {spec.model}")
            print(f"  Description: {spec.description}")
            print(f"  Context: {spec.context_window:,} tokens")
            print(f"  Cost: ${spec.cost_per_1k_tokens:.4f}/1k tokens")
            print(f"  Speed: ~{spec.tokens_per_second:.0f} tokens/sec")
            print()


def print_all_models() -> None:
    """Print all available models across all providers."""
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS BY PROVIDER AND TIER")
    print("=" * 70)

    for provider, specs in MODEL_CATALOG.items():
        print(f"\n┌{'─' * 68}┐")
        print(f"│ {provider.upper():^66} │")
        print(f"├{'─' * 68}┤")

        for tier in ModelTier:
            spec = specs[tier]
            cost_str = f"${spec.cost_per_1k_tokens:.4f}/1k" if spec.cost_per_1k_tokens > 0 else "FREE"
            print(f"│ {tier.value:<12} │ {spec.model:<25} │ {cost_str:<10} │ {spec.tokens_per_second:>5.0f} t/s │")

        print(f"└{'─' * 68}┘")

    print("\n💡 Recommendation:")
    print("   - Use LIGHTWEIGHT for code generation (fast, cheap)")
    print("   - Use HEAVYWEIGHT for reasoning (accurate)")
    print("   - Local (Ollama) = FREE, Cloud = costs vary")
