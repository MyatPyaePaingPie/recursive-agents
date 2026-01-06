"""Chunking strategies for splitting large contexts.

This module provides various strategies for splitting large text contexts
into manageable chunks that fit within LLM context windows.

Strategies:
    - FixedSizeChunking: Simple fixed-size character/token splits
    - SemanticChunking: Split on paragraph/sentence boundaries
    - HierarchicalChunking: Tree-based structure for nested documents
    - AdaptiveChunking: Dynamically adjust chunk size based on content
"""

import re
from abc import ABC, abstractmethod
from typing import Any

from rlm.context.models import ContextChunk
from rlm.utils.logging import get_logger

logger = get_logger(__name__)

# Approximate characters per token (varies by tokenizer)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    This is a simple character-based estimation. For production,
    use tiktoken or the appropriate tokenizer.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies.

    All chunking strategies must implement the `chunk` method
    which splits text into a list of ContextChunk objects.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        max_chunk_tokens: int = 4000,
        **kwargs: Any,
    ) -> list[ContextChunk]:
        """Split text into chunks.

        Args:
            text: The text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            **kwargs: Strategy-specific arguments

        Returns:
            List of ContextChunk objects
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass


class FixedSizeChunking(ChunkingStrategy):
    """Split text into fixed-size chunks.

    Simple strategy that splits text at regular intervals.
    May split mid-word or mid-sentence.

    Example:
        >>> strategy = FixedSizeChunking()
        >>> chunks = strategy.chunk("Very long text...", max_chunk_tokens=1000)
    """

    @property
    def name(self) -> str:
        return "fixed"

    def chunk(
        self,
        text: str,
        max_chunk_tokens: int = 4000,
        overlap_tokens: int = 100,
        **kwargs: Any,
    ) -> list[ContextChunk]:
        """Split text into fixed-size chunks with optional overlap.

        Args:
            text: Text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Token overlap between chunks
            **kwargs: Ignored

        Returns:
            List of ContextChunk objects
        """
        if not text:
            return []

        max_chars = max_chunk_tokens * CHARS_PER_TOKEN
        overlap_chars = overlap_tokens * CHARS_PER_TOKEN

        chunks = []
        chunk_id = 0
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end]

            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                start_pos=start,
                end_pos=end,
                tokens=estimate_tokens(chunk_text),
            )
            chunks.append(chunk)

            # Move start, accounting for overlap
            start = end - overlap_chars if end < len(text) else end
            chunk_id += 1

        logger.debug(f"FixedSizeChunking created {len(chunks)} chunks")
        return chunks


class SemanticChunking(ChunkingStrategy):
    """Split text at semantic boundaries (paragraphs, sentences).

    This strategy tries to preserve meaning by splitting at natural
    boundaries like paragraph breaks or sentence endings.

    Example:
        >>> strategy = SemanticChunking()
        >>> chunks = strategy.chunk("Paragraph 1.\\n\\nParagraph 2.", max_chunk_tokens=1000)
    """

    # Regex patterns for splitting
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(
        self,
        text: str,
        max_chunk_tokens: int = 4000,
        prefer_paragraphs: bool = True,
        **kwargs: Any,
    ) -> list[ContextChunk]:
        """Split text at semantic boundaries.

        Args:
            text: Text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            prefer_paragraphs: If True, prefer paragraph boundaries
            **kwargs: Ignored

        Returns:
            List of ContextChunk objects
        """
        if not text:
            return []

        # First, split by paragraphs
        if prefer_paragraphs:
            units = self.PARAGRAPH_PATTERN.split(text)
            separator = "\n\n"
        else:
            units = self.SENTENCE_PATTERN.split(text)
            separator = " "

        # Build chunks by accumulating units
        chunks = []
        chunk_id = 0
        current_text = ""
        start_pos = 0

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            test_text = current_text + separator + unit if current_text else unit
            test_tokens = estimate_tokens(test_text)

            if test_tokens > max_chunk_tokens and current_text:
                # Save current chunk and start new one
                end_pos = start_pos + len(current_text)
                chunk = ContextChunk(
                    chunk_id=chunk_id,
                    content=current_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    tokens=estimate_tokens(current_text),
                )
                chunks.append(chunk)

                chunk_id += 1
                start_pos = text.find(unit, end_pos)
                if start_pos == -1:
                    start_pos = end_pos
                current_text = unit
            else:
                current_text = test_text

        # Don't forget the last chunk
        if current_text:
            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=current_text,
                start_pos=start_pos,
                end_pos=start_pos + len(current_text),
                tokens=estimate_tokens(current_text),
            )
            chunks.append(chunk)

        logger.debug(f"SemanticChunking created {len(chunks)} chunks")
        return chunks


class HierarchicalChunking(ChunkingStrategy):
    """Split text based on document hierarchy.

    Useful for structured documents with headers, sections, etc.
    Preserves document structure in chunk metadata.

    Example:
        >>> strategy = HierarchicalChunking()
        >>> chunks = strategy.chunk("# Section 1\\nContent...\\n# Section 2\\nMore...")
    """

    # Patterns for detecting headers
    MARKDOWN_HEADER = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    NUMBERED_HEADER = re.compile(r"^(\d+\.)+\s+(.+)$", re.MULTILINE)

    @property
    def name(self) -> str:
        return "hierarchical"

    def chunk(
        self,
        text: str,
        max_chunk_tokens: int = 4000,
        **kwargs: Any,
    ) -> list[ContextChunk]:
        """Split text based on document structure.

        Args:
            text: Text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            **kwargs: Ignored

        Returns:
            List of ContextChunk objects with hierarchy metadata
        """
        if not text:
            return []

        # Find all headers
        headers = list(self.MARKDOWN_HEADER.finditer(text))
        if not headers:
            headers = list(self.NUMBERED_HEADER.finditer(text))

        if not headers:
            # No structure detected, fall back to semantic
            logger.debug("No headers found, falling back to semantic chunking")
            return SemanticChunking().chunk(text, max_chunk_tokens)

        # Split by headers
        chunks = []
        chunk_id = 0

        for i, header in enumerate(headers):
            start = header.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)

            section_text = text[start:end].strip()
            section_tokens = estimate_tokens(section_text)

            # If section is too large, sub-chunk it
            if section_tokens > max_chunk_tokens:
                sub_chunks = SemanticChunking().chunk(
                    section_text, max_chunk_tokens
                )
                for sub in sub_chunks:
                    sub.chunk_id = chunk_id
                    sub.start_pos = start + sub.start_pos
                    sub.end_pos = start + sub.end_pos
                    sub.metadata["header"] = header.group(2)
                    sub.metadata["level"] = len(header.group(1))
                    chunks.append(sub)
                    chunk_id += 1
            else:
                chunk = ContextChunk(
                    chunk_id=chunk_id,
                    content=section_text,
                    start_pos=start,
                    end_pos=end,
                    tokens=section_tokens,
                    metadata={
                        "header": header.group(2),
                        "level": len(header.group(1)),
                    },
                )
                chunks.append(chunk)
                chunk_id += 1

        logger.debug(f"HierarchicalChunking created {len(chunks)} chunks")
        return chunks


class AdaptiveChunking(ChunkingStrategy):
    """Dynamically adjust chunk size based on content type.

    This strategy analyzes the content to determine the best
    chunking approach (code vs prose, structured vs unstructured).

    Example:
        >>> strategy = AdaptiveChunking()
        >>> chunks = strategy.chunk(mixed_content, max_chunk_tokens=2000)
    """

    @property
    def name(self) -> str:
        return "adaptive"

    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content.

        Returns:
            One of: "code", "markdown", "prose"
        """
        # Check for code indicators
        code_patterns = [
            r"^\s*(def|class|import|from|if|for|while)\s",
            r"[{}\[\];]",
            r"^\s*#.*$",
        ]
        code_score = sum(
            len(re.findall(p, text, re.MULTILINE)) for p in code_patterns
        )

        # Check for markdown
        markdown_patterns = [r"^#+\s", r"^\*\s", r"^\d+\.\s", r"\[.*\]\(.*\)"]
        md_score = sum(
            len(re.findall(p, text, re.MULTILINE)) for p in markdown_patterns
        )

        lines = text.count("\n") + 1
        code_ratio = code_score / max(lines, 1)
        md_ratio = md_score / max(lines, 1)

        if code_ratio > 0.3:
            return "code"
        elif md_ratio > 0.1:
            return "markdown"
        else:
            return "prose"

    def chunk(
        self,
        text: str,
        max_chunk_tokens: int = 4000,
        **kwargs: Any,
    ) -> list[ContextChunk]:
        """Adaptively chunk based on content type.

        Args:
            text: Text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            **kwargs: Ignored

        Returns:
            List of ContextChunk objects
        """
        if not text:
            return []

        content_type = self._detect_content_type(text)
        logger.debug(f"AdaptiveChunking detected content type: {content_type}")

        if content_type == "code":
            # For code, use smaller chunks and fixed-size
            return FixedSizeChunking().chunk(
                text, max_chunk_tokens=max_chunk_tokens // 2
            )
        elif content_type == "markdown":
            # For markdown, use hierarchical chunking
            return HierarchicalChunking().chunk(text, max_chunk_tokens)
        else:
            # For prose, use semantic chunking
            return SemanticChunking().chunk(text, max_chunk_tokens)


def get_chunking_strategy(name: str) -> ChunkingStrategy:
    """Get a chunking strategy by name.

    Args:
        name: Strategy name ("fixed", "semantic", "hierarchical", "adaptive")

    Returns:
        ChunkingStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        "fixed": FixedSizeChunking,
        "semantic": SemanticChunking,
        "hierarchical": HierarchicalChunking,
        "adaptive": AdaptiveChunking,
    }

    if name.lower() not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[name.lower()]()
