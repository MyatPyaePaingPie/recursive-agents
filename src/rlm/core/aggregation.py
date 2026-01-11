"""Result aggregation strategies for combining recursive call results.

This module provides various strategies for aggregating results from
multiple recursive LLM calls into a single coherent response.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from rlm.models.base import BaseLLM
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


class AggregationMethod(Enum):
    """Available aggregation methods."""

    CONCATENATE = "concatenate"
    SUMMARIZE = "summarize"
    VOTE = "vote"
    WEIGHTED = "weighted"
    FIRST = "first"
    LAST = "last"


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies."""

    @abstractmethod
    async def aggregate(
        self,
        results: list[str],
        query: str,
        **kwargs: Any,
    ) -> str:
        """Aggregate multiple results into one.

        Args:
            results: List of results to aggregate
            query: Original query for context
            **kwargs: Strategy-specific arguments

        Returns:
            Aggregated result
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class ConcatenateStrategy(AggregationStrategy):
    """Simple concatenation of results.

    Joins all results with a separator. Fast but may produce
    redundant or incoherent output.
    """

    def __init__(self, separator: str = "\n\n---\n\n") -> None:
        """Initialize with separator."""
        self.separator = separator

    @property
    def name(self) -> str:
        return "concatenate"

    async def aggregate(
        self,
        results: list[str],
        query: str,
        **kwargs: Any,
    ) -> str:
        """Concatenate results with separator."""
        if not results:
            return ""
        return self.separator.join(r for r in results if r)


class SummarizeStrategy(AggregationStrategy):
    """Use LLM to summarize and combine results.

    Most coherent output but requires an additional LLM call.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """Initialize with LLM for summarization."""
        self.llm = llm

    @property
    def name(self) -> str:
        return "summarize"

    async def aggregate(
        self,
        results: list[str],
        query: str,
        **kwargs: Any,
    ) -> str:
        """Use LLM to synthesize results."""
        if not results:
            return ""

        if len(results) == 1:
            return results[0]

        # Build aggregation prompt
        results_text = "\n\n---\n\n".join(
            f"Result {i+1}:\n{r}" for i, r in enumerate(results)
        )

        prompt = f"""You are combining results from multiple context chunks to answer a query.

Original Query: {query}

Results from chunks:
{results_text}

Instructions:
1. Synthesize the information from all results
2. Remove redundancy while preserving important details
3. Create a coherent, well-organized answer
4. If results conflict, note the discrepancy
5. Focus on answering the original query

Combined Response:"""

        response = await self.llm.generate(prompt)
        return response.content


class VoteStrategy(AggregationStrategy):
    """Majority voting for classification tasks.

    Best for queries with discrete answers where multiple chunks
    might give the same or similar answers.
    """

    @property
    def name(self) -> str:
        return "vote"

    async def aggregate(
        self,
        results: list[str],
        query: str,
        **kwargs: Any,
    ) -> str:
        """Return most common result (simple voting)."""
        if not results:
            return ""

        # Count occurrences (normalized)
        normalized = [r.strip().lower() for r in results if r]
        if not normalized:
            return ""

        # Find most common
        from collections import Counter

        counts = Counter(normalized)
        most_common = counts.most_common(1)[0][0]

        # Return original (non-normalized) version
        for r in results:
            if r.strip().lower() == most_common:
                return r

        return results[0]


class WeightedStrategy(AggregationStrategy):
    """Weighted aggregation based on confidence or relevance.

    Requires results to include confidence scores.
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        """Initialize with optional LLM for confidence scoring."""
        self.llm = llm

    @property
    def name(self) -> str:
        return "weighted"

    async def aggregate(
        self,
        results: list[str],
        query: str,
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> str:
        """Aggregate with weights.

        If weights not provided, falls back to summarize or concatenate.
        """
        if not results:
            return ""

        if len(results) == 1:
            return results[0]

        if weights and len(weights) == len(results):
            # Sort by weight and take top results
            weighted = sorted(
                zip(results, weights), key=lambda x: x[1], reverse=True
            )
            top_results = [r for r, w in weighted[:3]]

            # Concatenate top results
            return "\n\n".join(top_results)

        # No weights - fall back to concatenation
        return "\n\n".join(results[:5])


class FirstStrategy(AggregationStrategy):
    """Return first non-empty result.

    Useful when order matters and first result is most relevant.
    """

    @property
    def name(self) -> str:
        return "first"

    async def aggregate(
        self,
        results: list[str],
        query: str,
        **kwargs: Any,
    ) -> str:
        """Return first non-empty result."""
        for r in results:
            if r and r.strip():
                return r
        return ""


class LastStrategy(AggregationStrategy):
    """Return last non-empty result.

    Useful when later results build on earlier ones.
    """

    @property
    def name(self) -> str:
        return "last"

    async def aggregate(
        self,
        results: list[str],
        query: str,
        **kwargs: Any,
    ) -> str:
        """Return last non-empty result."""
        for r in reversed(results):
            if r and r.strip():
                return r
        return ""


def get_aggregation_strategy(
    method: AggregationMethod | str,
    llm: BaseLLM | None = None,
) -> AggregationStrategy:
    """Get an aggregation strategy by name.

    Args:
        method: Strategy method name or enum
        llm: LLM instance (required for summarize strategy)

    Returns:
        AggregationStrategy instance

    Raises:
        ValueError: If method is unknown or LLM missing for summarize
    """
    if isinstance(method, str):
        try:
            method = AggregationMethod(method.lower())
        except ValueError:
            raise ValueError(f"Unknown aggregation method: {method}")

    strategies = {
        AggregationMethod.CONCATENATE: ConcatenateStrategy,
        AggregationMethod.VOTE: VoteStrategy,
        AggregationMethod.FIRST: FirstStrategy,
        AggregationMethod.LAST: LastStrategy,
    }

    if method == AggregationMethod.SUMMARIZE:
        if not llm:
            raise ValueError("LLM required for summarize aggregation")
        return SummarizeStrategy(llm)

    if method == AggregationMethod.WEIGHTED:
        return WeightedStrategy(llm)

    if method in strategies:
        return strategies[method]()

    raise ValueError(f"Unknown aggregation method: {method}")
