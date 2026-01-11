"""Data models for context management.

This module defines the Pydantic models used for representing
context chunks, access logs, and metadata.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    """Represents a chunk of the input context.

    Attributes:
        chunk_id: Unique identifier for this chunk
        content: The actual text content
        start_pos: Starting character position in original context
        end_pos: Ending character position in original context
        tokens: Estimated token count for this chunk
        metadata: Additional metadata (e.g., section headers)
    """

    chunk_id: int = Field(..., ge=0, description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    start_pos: int = Field(..., ge=0, description="Start position in original")
    end_pos: int = Field(..., ge=0, description="End position in original")
    tokens: int = Field(default=0, ge=0, description="Estimated token count")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def length(self) -> int:
        """Get the character length of this chunk."""
        return len(self.content)

    def __repr__(self) -> str:
        return f"ContextChunk(id={self.chunk_id}, tokens={self.tokens}, pos={self.start_pos}-{self.end_pos})"


class ChunkAccess(BaseModel):
    """Records an access to a context chunk.

    Used for tracking which chunks have been accessed and how,
    useful for optimization and debugging.

    Attributes:
        chunk_id: ID of the accessed chunk
        timestamp: When the access occurred
        access_type: Type of access ("read", "search", "submodel")
        query: Optional query associated with the access
    """

    chunk_id: int = Field(..., description="Accessed chunk ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Access time")
    access_type: str = Field(default="read", description="Type of access")
    query: str | None = Field(default=None, description="Associated query")


class ContextMetadata(BaseModel):
    """Metadata about the loaded context.

    Attributes:
        total_length: Total character length
        total_tokens: Estimated total tokens
        num_chunks: Number of chunks created
        chunking_strategy: Strategy used for chunking
        structure_type: Detected structure (e.g., "prose", "code", "mixed")
        created_at: When the context was loaded
    """

    total_length: int = Field(..., ge=0, description="Total character length")
    total_tokens: int = Field(..., ge=0, description="Estimated total tokens")
    num_chunks: int = Field(..., ge=0, description="Number of chunks")
    chunking_strategy: str = Field(..., description="Chunking strategy used")
    structure_type: str = Field(default="unknown", description="Detected structure type")
    created_at: datetime = Field(default_factory=datetime.now, description="Load timestamp")


class ContextSummary(BaseModel):
    """Summary of context for prompting.

    Provides a concise summary of the context that can be included
    in prompts to help the LLM understand what's available.

    Attributes:
        total_tokens: Total tokens in context
        num_chunks: Number of chunks
        structure: Description of structure
        sample_start: Sample from beginning
        sample_end: Sample from end
    """

    total_tokens: int = Field(..., description="Total tokens")
    num_chunks: int = Field(..., description="Number of chunks")
    structure: str = Field(..., description="Structure description")
    sample_start: str = Field(default="", description="Sample from start")
    sample_end: str = Field(default="", description="Sample from end")

    def to_prompt_string(self) -> str:
        """Convert to a string suitable for prompting."""
        return f"""Context Summary:
- Total tokens: {self.total_tokens:,}
- Chunks: {self.num_chunks}
- Structure: {self.structure}
- Start: "{self.sample_start[:100]}..."
- End: "...{self.sample_end[-100:]}" """
