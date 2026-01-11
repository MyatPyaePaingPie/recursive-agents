"""Context manager for storing and accessing large contexts.

This module provides the ContextManager class that handles loading,
chunking, and accessing large input contexts for the RLM system.

Example:
    >>> from rlm.context import ContextManager, SemanticChunking
    >>> manager = ContextManager(strategy=SemanticChunking())
    >>> manager.load_context("Very long document text...")
    >>> print(manager.get_context_length())
    >>> chunk = manager.get_chunk(0)
"""

import re
from datetime import datetime
from typing import Any

from rlm.context.chunking import (
    ChunkingStrategy,
    SemanticChunking,
    estimate_tokens,
)
from rlm.context.models import (
    ChunkAccess,
    ContextChunk,
    ContextMetadata,
    ContextSummary,
)
from rlm.exceptions import ContextError
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


class ContextManager:
    """Manages large context storage and access.

    The ContextManager is responsible for:
    - Storing arbitrarily large contexts
    - Chunking contexts using configurable strategies
    - Providing chunk-based access
    - Tracking access patterns for optimization

    Attributes:
        strategy: The chunking strategy to use
        chunks: List of context chunks
        metadata: Context metadata
        access_log: Log of chunk accesses

    Example:
        >>> manager = ContextManager()
        >>> manager.load_context(long_document)
        >>> print(f"Total tokens: {manager.get_context_length()}")
        >>> chunk = manager.get_chunk(0)
    """

    def __init__(
        self,
        strategy: ChunkingStrategy | None = None,
        max_chunk_tokens: int = 4000,
    ) -> None:
        """Initialize the context manager.

        Args:
            strategy: Chunking strategy (default: SemanticChunking)
            max_chunk_tokens: Maximum tokens per chunk
        """
        self.strategy = strategy or SemanticChunking()
        self.max_chunk_tokens = max_chunk_tokens
        self.chunks: list[ContextChunk] = []
        self.metadata: ContextMetadata | None = None
        self.access_log: list[ChunkAccess] = []
        self._raw_context: str = ""

        logger.debug(
            f"Initialized ContextManager with strategy={self.strategy.name}"
        )

    def load_context(
        self,
        content: str,
        max_chunk_tokens: int | None = None,
        **kwargs: Any,
    ) -> ContextMetadata:
        """Load and chunk a context.

        Args:
            content: The raw context text
            max_chunk_tokens: Override default max chunk size
            **kwargs: Additional arguments passed to chunking strategy

        Returns:
            ContextMetadata with information about the loaded context

        Raises:
            ContextError: If content is empty or chunking fails
        """
        if not content or not content.strip():
            raise ContextError("Cannot load empty context")

        self._raw_context = content
        chunk_size = max_chunk_tokens or self.max_chunk_tokens

        logger.info(f"Loading context: {len(content)} chars, strategy={self.strategy.name}")

        try:
            self.chunks = self.strategy.chunk(content, chunk_size, **kwargs)
        except Exception as e:
            raise ContextError(f"Chunking failed: {e}") from e

        if not self.chunks:
            raise ContextError("Chunking produced no chunks")

        # Detect structure type
        structure = self._detect_structure(content)

        self.metadata = ContextMetadata(
            total_length=len(content),
            total_tokens=sum(c.tokens for c in self.chunks),
            num_chunks=len(self.chunks),
            chunking_strategy=self.strategy.name,
            structure_type=structure,
            created_at=datetime.now(),
        )

        logger.info(
            f"Context loaded: {self.metadata.num_chunks} chunks, "
            f"{self.metadata.total_tokens} tokens"
        )

        return self.metadata

    def _detect_structure(self, text: str) -> str:
        """Detect the structure type of the text."""
        # Simple heuristics
        if re.search(r"^(#{1,6}|def |class |import )", text, re.MULTILINE):
            if re.search(r"^(def |class |import )", text, re.MULTILINE):
                return "code"
            return "markdown"
        elif re.search(r"\n\s*\n", text):
            return "prose"
        else:
            return "unknown"

    def get_context_length(self) -> int:
        """Get total context length in tokens.

        Returns:
            Total estimated token count
        """
        if not self.metadata:
            return 0
        return self.metadata.total_tokens

    def get_chunk(self, chunk_id: int, log_access: bool = True) -> ContextChunk:
        """Get a specific chunk by ID.

        Args:
            chunk_id: The chunk ID to retrieve
            log_access: Whether to log this access

        Returns:
            The ContextChunk

        Raises:
            ContextError: If chunk_id is invalid
        """
        if not self.chunks:
            raise ContextError("No context loaded")

        if chunk_id < 0 or chunk_id >= len(self.chunks):
            raise ContextError(
                f"Invalid chunk_id: {chunk_id}. Valid range: 0-{len(self.chunks)-1}"
            )

        if log_access:
            self.access_log.append(
                ChunkAccess(chunk_id=chunk_id, access_type="read")
            )

        return self.chunks[chunk_id]

    def get_chunk_content(self, chunk_id: int) -> str:
        """Get the content of a specific chunk.

        Convenience method that returns just the text content.

        Args:
            chunk_id: The chunk ID

        Returns:
            The chunk's text content
        """
        return self.get_chunk(chunk_id).content

    def get_chunk_range(self, start: int, end: int) -> str:
        """Get context content from character position range.

        Args:
            start: Starting character position
            end: Ending character position

        Returns:
            The text content in the range

        Raises:
            ContextError: If range is invalid
        """
        if not self._raw_context:
            raise ContextError("No context loaded")

        if start < 0:
            start = 0
        if end > len(self._raw_context):
            end = len(self._raw_context)
        if start >= end:
            raise ContextError(f"Invalid range: {start}-{end}")

        # Log access for relevant chunks
        for chunk in self.chunks:
            if chunk.start_pos < end and chunk.end_pos > start:
                self.access_log.append(
                    ChunkAccess(chunk_id=chunk.chunk_id, access_type="range_read")
                )

        return self._raw_context[start:end]

    def search_context(
        self,
        pattern: str,
        max_results: int = 10,
    ) -> list[tuple[int, int, str]]:
        """Search the context for a regex pattern.

        Args:
            pattern: Regex pattern to search for
            max_results: Maximum number of results

        Returns:
            List of (start, end, matched_text) tuples
        """
        if not self._raw_context:
            return []

        results = []
        try:
            for match in re.finditer(pattern, self._raw_context):
                if len(results) >= max_results:
                    break
                results.append((match.start(), match.end(), match.group()))

                # Log access
                self.access_log.append(
                    ChunkAccess(
                        chunk_id=-1,
                        access_type="search",
                        query=pattern,
                    )
                )
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {e}")

        return results

    def get_context_summary(self) -> ContextSummary:
        """Get a summary of the context for prompting.

        Returns:
            ContextSummary object
        """
        if not self.metadata or not self._raw_context:
            return ContextSummary(
                total_tokens=0,
                num_chunks=0,
                structure="empty",
                sample_start="",
                sample_end="",
            )

        sample_size = 200
        return ContextSummary(
            total_tokens=self.metadata.total_tokens,
            num_chunks=self.metadata.num_chunks,
            structure=self.metadata.structure_type,
            sample_start=self._raw_context[:sample_size],
            sample_end=self._raw_context[-sample_size:],
        )

    def get_access_statistics(self) -> dict[str, Any]:
        """Get statistics about context access patterns.

        Returns:
            Dictionary with access statistics
        """
        if not self.access_log:
            return {"total_accesses": 0, "by_type": {}, "by_chunk": {}}

        by_type: dict[str, int] = {}
        by_chunk: dict[int, int] = {}

        for access in self.access_log:
            by_type[access.access_type] = by_type.get(access.access_type, 0) + 1
            by_chunk[access.chunk_id] = by_chunk.get(access.chunk_id, 0) + 1

        return {
            "total_accesses": len(self.access_log),
            "by_type": by_type,
            "by_chunk": by_chunk,
        }

    def clear(self) -> None:
        """Clear all loaded context and reset state."""
        self.chunks = []
        self.metadata = None
        self.access_log = []
        self._raw_context = ""
        logger.debug("ContextManager cleared")

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __iter__(self):
        """Iterate over chunks."""
        return iter(self.chunks)

    def __repr__(self) -> str:
        if self.metadata:
            return (
                f"ContextManager(chunks={self.metadata.num_chunks}, "
                f"tokens={self.metadata.total_tokens})"
            )
        return "ContextManager(empty)"
