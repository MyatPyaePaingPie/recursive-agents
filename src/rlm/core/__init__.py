"""Core recursive inference engine for the RLM system.

This module contains the main RecursiveInferenceEngine that orchestrates
the recursive processing of large contexts.

Example:
    >>> from rlm.core import RecursiveInferenceEngine
    >>> from rlm.models import create_llm
    >>> from rlm.config import RLMConfig
    >>>
    >>> config = RLMConfig()
    >>> llm = create_llm("ollama")
    >>> engine = RecursiveInferenceEngine(llm=llm, config=config)
    >>> result = await engine.process("Summarize this", long_text)
"""

from rlm.core.engine import RecursiveInferenceEngine
from rlm.core.models import InferenceResult, ProcessingState, RecursionNode
from rlm.core.transparent import Event, EventType, TransparentEngine

__all__ = [
    "RecursiveInferenceEngine",
    "TransparentEngine",
    "InferenceResult",
    "RecursionNode",
    "ProcessingState",
    "Event",
    "EventType",
]
