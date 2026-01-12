"""Core recursive inference engine for the RLM system.

This module contains the RLM engines:
- RLMEngine: Paper-accurate engine with complexity detection (RECOMMENDED)
- RecursiveInferenceEngine: Original implementation
- TransparentEngine: Heavily instrumented version

Example (Paper-accurate):
    >>> from rlm.core import RLMEngine
    >>> from rlm.models import create_llm
    >>>
    >>> heavyweight = create_llm("anthropic", model="claude-3-5-sonnet")
    >>> lightweight = create_llm("groq", model="llama-3.1-8b-instant")
    >>> engine = RLMEngine(heavyweight_llm=heavyweight, lightweight_llm=lightweight)
    >>> result = await engine.process("Summarize this book", long_text)
"""

from rlm.core.engine import RecursiveInferenceEngine
from rlm.core.models import InferenceResult, ProcessingState, RecursionNode
from rlm.core.rlm_engine import (
    Event,
    EventType,
    RLMEngine,
    RLMResult,
    Trajectory,
)
from rlm.core.transparent import TransparentEngine

__all__ = [
    # Paper-accurate (recommended)
    "RLMEngine",
    "RLMResult",
    "Trajectory",
    "Event",
    "EventType",
    # Original implementations
    "RecursiveInferenceEngine",
    "TransparentEngine",
    "InferenceResult",
    "RecursionNode",
    "ProcessingState",
]
