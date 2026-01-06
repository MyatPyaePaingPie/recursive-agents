"""Recursive Language Models (RLM) - Process unlimited context through code.

This package implements the RLM architecture from the 2025 research paper,
enabling LLMs to process arbitrarily long inputs by generating and executing
code that recursively examines context.

Example:
    >>> from rlm import RecursiveInferenceEngine
    >>> from rlm.models import create_llm
    >>> from rlm.config import RLMConfig
    >>>
    >>> config = RLMConfig()
    >>> llm = create_llm("ollama", model="mistral:7b")
    >>> engine = RecursiveInferenceEngine(llm=llm, config=config)
    >>> result = await engine.process(query="Summarize this", context=long_text)
"""

from rlm.core.engine import RecursiveInferenceEngine
from rlm.core.models import InferenceResult, RecursionNode
from rlm.config import RLMConfig

__version__ = "0.1.0"
__all__ = [
    "RecursiveInferenceEngine",
    "InferenceResult",
    "RecursionNode",
    "RLMConfig",
]
