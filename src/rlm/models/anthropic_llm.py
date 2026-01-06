"""Anthropic Claude LLM implementation.

Provides integration with Anthropic's Claude models.
Requires an Anthropic API key.

Example:
    >>> from rlm.models.anthropic_llm import AnthropicLLM
    >>> llm = AnthropicLLM(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
    >>> response = await llm.generate("Hello!")
"""

from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from rlm.exceptions import LLMError
from rlm.models.base import BaseLLM, LLMResponse
from rlm.models.prompts import CODE_GENERATION_SYSTEM_PROMPT
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM provider.

    Supports Claude 3 Opus, Sonnet, and Haiku models.

    Attributes:
        api_key: Anthropic API key
        model: Model name (e.g., "claude-3-opus-20240229")

    Recommended Models:
        - claude-3-opus-20240229: Best quality, highest cost
        - claude-3-sonnet-20240229: Good balance of quality and speed
        - claude-3-haiku-20240307: Fast and cheap

    Example:
        >>> llm = AnthropicLLM(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
        >>> response = await llm.generate("Explain recursion")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize Anthropic LLM.

        Args:
            api_key: Anthropic API key (required)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError("Anthropic API key is required")

        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.debug(f"Initialized AnthropicLLM with model={model}")

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
        """Generate text completion using Anthropic Claude.

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
        try:
            logger.debug(
                f"Calling Anthropic model={self.model}, prompt_length={len(prompt)}"
            )

            message = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            content = message.content[0].text if message.content else ""
            usage = message.usage

            result = LLMResponse(
                content=content,
                tokens_used=usage.input_tokens + usage.output_tokens,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                model=self.model,
                finish_reason=message.stop_reason or "stop",
            )

            logger.debug(f"Anthropic response: tokens={result.tokens_used}")
            return result

        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic authentication error: {e}")
            raise LLMError(
                "Invalid Anthropic API key",
                provider="anthropic",
                model=self.model,
                original_error=e,
            ) from e
        except anthropic.RateLimitError as e:
            logger.warning(f"Anthropic rate limit: {e}")
            raise LLMError(
                "Anthropic rate limit exceeded",
                provider="anthropic",
                model=self.model,
                original_error=e,
            ) from e
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise LLMError(
                f"Anthropic API error: {e}",
                provider="anthropic",
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
