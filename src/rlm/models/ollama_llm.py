"""Ollama LLM implementation for local model inference.

Ollama provides an easy way to run LLMs locally with an OpenAI-compatible API.
This is the recommended provider for development and testing.

Setup:
    1. Install Ollama: https://ollama.ai/download
    2. Pull a model: `ollama pull mistral:7b`
    3. Start server: `ollama serve` (runs automatically on install)

Example:
    >>> from rlm.models.ollama_llm import OllamaLLM
    >>> llm = OllamaLLM(model="mistral:7b")
    >>> response = await llm.generate("Hello, world!")
"""

from typing import Any

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from rlm.exceptions import LLMError
from rlm.models.base import BaseLLM, LLMResponse
from rlm.models.prompts import CODE_GENERATION_SYSTEM_PROMPT
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM provider for local inference.

    Uses Ollama's OpenAI-compatible API for seamless integration.
    No API key required - runs entirely locally.

    Attributes:
        base_url: Ollama server URL (default: http://localhost:11434/v1)
        model: Model name (e.g., "mistral:7b", "llama3:8b")

    Example:
        >>> llm = OllamaLLM(model="mistral:7b")
        >>> response = await llm.generate("Explain recursion")
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str = "mistral:7b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize Ollama LLM.

        Args:
            model: Ollama model name (e.g., "mistral:7b")
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",  # Required but not used by Ollama
        )
        logger.debug(f"Initialized OllamaLLM with model={model}, base_url={base_url}")

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
        """Generate text completion using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional arguments passed to API

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
            logger.debug(f"Calling Ollama model={self.model}, prompt_length={len(prompt)}")

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

            logger.debug(f"Ollama response: tokens={result.tokens_used}")
            return result

        except openai.APIConnectionError as e:
            logger.error(f"Ollama connection error: {e}")
            raise LLMError(
                "Failed to connect to Ollama. Is the server running?",
                provider="ollama",
                model=self.model,
                original_error=e,
            ) from e
        except openai.APIError as e:
            logger.error(f"Ollama API error: {e}")
            raise LLMError(
                f"Ollama API error: {e}",
                provider="ollama",
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

        Uses a specialized system prompt for code generation that
        describes the available context access functions.

        Args:
            task_description: What the code should accomplish
            context_info: Information about the context (size, structure)
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
            temperature=0.3,  # Lower temperature for more deterministic code
            **kwargs,
        )

        # Extract code from response (handle markdown code blocks)
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()

    async def health_check(self) -> bool:
        """Check if Ollama server is available.

        Returns:
            True if Ollama is running and the model is available
        """
        try:
            # Try to list models
            models = await self.client.models.list()
            available_models = [m.id for m in models.data]
            logger.debug(f"Ollama available models: {available_models}")
            return self.model in available_models or any(
                self.model in m for m in available_models
            )
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
