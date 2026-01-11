"""Tests for context management module."""

import pytest

from rlm.context import (
    AdaptiveChunking,
    ContextManager,
    FixedSizeChunking,
    HierarchicalChunking,
    SemanticChunking,
)
from rlm.context.chunking import estimate_tokens
from rlm.exceptions import ContextError


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_basic_estimation(self) -> None:
        """Test basic token estimation."""
        # ~4 chars per token
        text = "This is a test string."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)

    def test_empty_string(self) -> None:
        """Test estimation for empty string."""
        assert estimate_tokens("") == 1  # minimum of 1


class TestFixedSizeChunking:
    """Tests for FixedSizeChunking strategy."""

    def test_basic_chunking(self) -> None:
        """Test basic fixed-size chunking."""
        strategy = FixedSizeChunking()
        text = "Hello world. " * 100

        chunks = strategy.chunk(text, max_chunk_tokens=50)
        assert len(chunks) > 1
        assert all(c.tokens <= 60 for c in chunks)  # Allow some tolerance

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        strategy = FixedSizeChunking()
        chunks = strategy.chunk("", max_chunk_tokens=100)
        assert len(chunks) == 0

    def test_small_text(self) -> None:
        """Test text smaller than chunk size."""
        strategy = FixedSizeChunking()
        text = "Small text."
        chunks = strategy.chunk(text, max_chunk_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0].content == text


class TestSemanticChunking:
    """Tests for SemanticChunking strategy."""

    def test_paragraph_splitting(self) -> None:
        """Test splitting by paragraphs."""
        strategy = SemanticChunking()
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = strategy.chunk(text, max_chunk_tokens=1000)
        # Should keep paragraphs together when possible
        assert len(chunks) >= 1

    def test_long_paragraph(self) -> None:
        """Test handling of paragraphs longer than chunk size."""
        strategy = SemanticChunking()
        text = "This is a very long paragraph. " * 100

        chunks = strategy.chunk(text, max_chunk_tokens=50)
        assert len(chunks) > 1


class TestHierarchicalChunking:
    """Tests for HierarchicalChunking strategy."""

    def test_markdown_headers(self) -> None:
        """Test chunking markdown with headers."""
        strategy = HierarchicalChunking()
        text = """# Section 1
Content for section 1.

# Section 2
Content for section 2.

## Subsection 2.1
More content here.
"""
        chunks = strategy.chunk(text, max_chunk_tokens=1000)
        assert len(chunks) >= 2

        # Check headers in metadata
        headers = [c.metadata.get("header") for c in chunks if c.metadata.get("header")]
        assert len(headers) > 0

    def test_no_headers(self) -> None:
        """Test fallback when no headers found."""
        strategy = HierarchicalChunking()
        text = "Just plain text without any headers or structure."

        chunks = strategy.chunk(text, max_chunk_tokens=1000)
        assert len(chunks) >= 1


class TestAdaptiveChunking:
    """Tests for AdaptiveChunking strategy."""

    def test_code_detection(self) -> None:
        """Test code content detection."""
        strategy = AdaptiveChunking()
        code = """
def hello():
    print("Hello")

class MyClass:
    pass

import os
"""
        chunks = strategy.chunk(code, max_chunk_tokens=1000)
        assert len(chunks) >= 1

    def test_prose_detection(self) -> None:
        """Test prose content detection."""
        strategy = AdaptiveChunking()
        prose = """
This is a long paragraph of prose text. It contains multiple sentences
that flow together naturally. There are no code elements or markdown
headers in this text. It should be detected as prose content.
"""
        chunks = strategy.chunk(prose, max_chunk_tokens=1000)
        assert len(chunks) >= 1


class TestContextManager:
    """Tests for ContextManager."""

    def test_load_context(self) -> None:
        """Test loading context."""
        manager = ContextManager()
        text = "Test context with some content. " * 50

        metadata = manager.load_context(text)

        assert metadata.total_length == len(text)
        assert metadata.num_chunks > 0
        assert manager.get_context_length() > 0

    def test_empty_context_error(self) -> None:
        """Test error on empty context."""
        manager = ContextManager()

        with pytest.raises(ContextError):
            manager.load_context("")

        with pytest.raises(ContextError):
            manager.load_context("   ")

    def test_get_chunk(self) -> None:
        """Test getting specific chunk."""
        manager = ContextManager()
        manager.load_context("Test " * 1000)

        chunk = manager.get_chunk(0)
        assert chunk.chunk_id == 0
        assert len(chunk.content) > 0

    def test_invalid_chunk_id(self) -> None:
        """Test error on invalid chunk ID."""
        manager = ContextManager()
        manager.load_context("Test " * 100)

        with pytest.raises(ContextError):
            manager.get_chunk(999)

        with pytest.raises(ContextError):
            manager.get_chunk(-1)

    def test_get_chunk_range(self) -> None:
        """Test getting context by character range."""
        manager = ContextManager()
        text = "ABCDEFGHIJ" * 100
        manager.load_context(text)

        result = manager.get_chunk_range(0, 10)
        assert result == "ABCDEFGHIJ"

    def test_search_context(self) -> None:
        """Test searching context."""
        manager = ContextManager()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        manager.load_context(text)

        results = manager.search_context(r"fox")
        assert len(results) > 0
        assert all("fox" in r[2] for r in results)

    def test_context_summary(self) -> None:
        """Test getting context summary."""
        manager = ContextManager()
        manager.load_context("Test content here. " * 100)

        summary = manager.get_context_summary()
        assert summary.total_tokens > 0
        assert summary.num_chunks > 0
        assert len(summary.sample_start) > 0

    def test_access_statistics(self) -> None:
        """Test access statistics tracking."""
        manager = ContextManager()
        manager.load_context("Test " * 100)

        # Make some accesses
        manager.get_chunk(0)
        manager.get_chunk(0)
        manager.search_context("Test")

        stats = manager.get_access_statistics()
        assert stats["total_accesses"] >= 3

    def test_clear(self) -> None:
        """Test clearing context."""
        manager = ContextManager()
        manager.load_context("Test " * 100)

        manager.clear()

        assert len(manager.chunks) == 0
        assert manager.metadata is None
