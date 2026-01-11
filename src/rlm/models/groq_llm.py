"""Groq LLM implementation for fast cloud inference.

Groq provides extremely fast inference (300+ tokens/second) with a
generous free tier (30 requests/minute). Great for testing and
low-latency applications.

Setup:
    1. Get API key: https://console.groq.com/
    2. Set GROQ_API_KEY environment variable

Example:
    >>> from rlm.models.groq_llm import GroqLLM
    >>> llm = GroqLLM(api_key="your-key", model="llama-3.1-8b-instant")
    >>> response = await llm.generate("Hello!")
"""

from typing import Any

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from rlm.exceptions import LLMError
from rlm.models.base import BaseLLM, LLMResponse
from rlm.models.prompts import CODE_GENERATION_SYSTEM_PROMPT
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


class GroqLLM(BaseLLM):
    """Groq LLM provider for fast cloud inference.

    Groq uses custom hardware (LPUs) for extremely fast inference.
    Free tier: 30 requests/minute, 6000 tokens/minute.

    Attributes:
        api_key: Groq API key
        model: Model name (e.g., "llama-3.1-8b-instant", "mixtral-8x7b-32768")

    Recommended Models:
        - llama-3.1-8b-instant: Fast, good for code generation
        - llama-3.1-70b-versatile: More capable, slightly slower
        - mixtral-8x7b-32768: Good balance, large context

    Example:
        >>> llm = GroqLLM(api_key="your-key")
        >>> response = await llm.generate("Explain recursion")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize Groq LLM.

        Args:
            api_key: Groq API key (required)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError("Groq API key is required")

        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.client = AsyncGroq(api_key=api_key)
        logger.debug(f"Initialized GroqLLM with model={model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text completion using Groq.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional arguments

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: If the API call fails
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"Calling Groq model={self.model}, prompt_length={len(prompt)}")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content or ""
            usage = response.usage

            result = LLMResponse(
                content=content,
                tokens_used=usage.total_tokens if usage else 0,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                model=self.model,
                finish_reason=response.choices[0].finish_reason or "stop",
            )

            logger.debug(f"Groq response: tokens={result.tokens_used}")
            return result

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise LLMError(
                f"Groq API error: {e}",
                provider="groq",
                model=self.model,
                original_error=e,
            ) from e

    async def generate_code(
        self,
        task_description: str,
        context_info: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate Python code for context processing.

        Args:
            task_description: What the code should accomplish
            context_info: Information about the context
            **kwargs: Additional arguments

        Returns:
            Generated Python code

        Raises:
            LLMError: If code generation fails
        """
        prompt_parts = [f"Task: {task_description}"]
        if context_info:
            prompt_parts.append(f"\nContext Information:\n{context_info}")
        prompt_parts.append("\nWrite Python code to accomplish this task.")

        prompt = "\n".join(prompt_parts)

        response = await self.generate(
            prompt=prompt,
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
            temperature=0.3,
            **kwargs,
        )

        # Extract code from response
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()
