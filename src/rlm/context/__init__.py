"""Context management for the RLM system.

This module provides context storage, chunking strategies, and access APIs
for managing large input contexts that exceed LLM context windows.

Example:
    >>> from rlm.context import ContextManager, SemanticChunking
    >>> manager = ContextManager(strategy=SemanticChunking())
    >>> manager.load_context("Very long text...")
    >>> chunk = manager.get_chunk(0)
"""

from rlm.context.chunking import (
    AdaptiveChunking,
    ChunkingStrategy,
    FixedSizeChunking,
    HierarchicalChunking,
    SemanticChunking,
)
from rlm.context.manager import ContextManager
from rlm.context.models import ChunkAccess, ContextChunk, ContextMetadata

__all__ = [
    "ContextManager",
    "ContextChunk",
    "ContextMetadata",
    "ChunkAccess",
    "ChunkingStrategy",
    "FixedSizeChunking",
    "SemanticChunking",
    "HierarchicalChunking",
    "AdaptiveChunking",
]
