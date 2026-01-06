"""Base interface for LLM providers.

This module defines the abstract base class that all LLM implementations
must inherit from, ensuring a consistent interface across providers.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Response from an LLM API call.

    Attributes:
        content: The generated text content
        tokens_used: Total tokens consumed (prompt + completion)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        model: The model that generated the response
        finish_reason: Why generation stopped (e.g., "stop", "length")
        raw_response: Optional raw response from the API
    """

    content: str = Field(..., description="Generated text content")
    tokens_used: int = Field(..., ge=0, description="Total tokens consumed")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens")
    model: str = Field(..., description="Model that generated response")
    finish_reason: str = Field(default="stop", description="Reason generation stopped")
    raw_response: dict[str, Any] | None = Field(
        default=None, description="Raw API response"
    )


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    All LLM implementations (Ollama, OpenAI, Anthropic, Groq) must inherit
    from this class and implement the abstract methods.

    Attributes:
        model: The model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Example:
        >>> class MyLLM(BaseLLM):
        ...     async def generate(self, prompt, **kwargs):
        ...         # Implementation
        ...         pass
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize the LLM.

        Args:
            model: Model identifier (e.g., "gpt-4", "mistral:7b")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion.

        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: If the API call fails
        """
        pass

    @abstractmethod
    async def generate_code(
        self,
        task_description: str,
        context_info: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate Python code for a task.

        This method is specialized for generating code that will be
        executed in the RLM sandbox. It includes appropriate system
        prompts for code generation.

        Args:
            task_description: Description of the code to generate
            context_info: Information about available context
            **kwargs: Additional arguments

        Returns:
            Generated Python code as a string

        Raises:
            LLMError: If code generation fails
        """
        pass

    async def health_check(self) -> bool:
        """Check if the LLM provider is available.

        Returns:
            True if the provider is healthy, False otherwise
        """
        try:
            response = await self.generate("Say 'ok'", max_tokens=10)
            return len(response.content) > 0
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
