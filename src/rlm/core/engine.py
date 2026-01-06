"""The core Recursive Inference Engine.

This is the heart of the RLM system. It orchestrates the recursive
processing of large contexts by:

1. Loading context into the ContextManager
2. Generating code to examine the context
3. Executing code safely in the sandbox
4. Making recursive calls as needed
5. Aggregating results

Based on the 2025 research paper "Recursive Language Models".
"""

import time
from typing import Any

from rlm.config import RLMConfig
from rlm.context import ContextManager, SemanticChunking
from rlm.context.chunking import get_chunking_strategy
from rlm.core.models import InferenceResult, ProcessingState, RecursionNode
from rlm.exceptions import (
    CodeExecutionError,
    LLMError,
    MaxDepthExceededError,
    RLMException,
)
from rlm.execution import SandboxEnvironment
from rlm.execution.sandbox import ContextAPIBuilder
from rlm.models.base import BaseLLM
from rlm.models.prompts import (
    AGGREGATION_PROMPT,
    CODE_GENERATION_SYSTEM_PROMPT,
    CODE_GENERATION_USER_TEMPLATE,
    DIRECT_PROCESSING_PROMPT,
)
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


class RecursiveInferenceEngine:
    """Main engine for recursive inference.

    This class implements the RLM algorithm from the research paper:
    1. Accept user query and large context
    2. Generate code to examine and process context
    3. Execute code in sandbox
    4. Make recursive calls as needed (depth = 1 per paper)
    5. Aggregate and return results

    Attributes:
        llm: LLM instance for generation
        config: RLM configuration
        context_manager: Manager for context storage
        sandbox: Sandbox for code execution

    Example:
        >>> engine = RecursiveInferenceEngine(llm, config)
        >>> result = await engine.process("Summarize this book", long_text)
        >>> print(result.answer)
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: RLMConfig | None = None,
        context_manager: ContextManager | None = None,
        sandbox: SandboxEnvironment | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            llm: LLM instance for text generation
            config: Configuration (uses defaults if None)
            context_manager: Context manager (creates default if None)
            sandbox: Sandbox environment (creates default if None)
        """
        self.llm = llm
        self.config = config or RLMConfig()

        # Create or use provided context manager
        if context_manager:
            self.context_manager = context_manager
        else:
            strategy = get_chunking_strategy(self.config.chunking_strategy)
            self.context_manager = ContextManager(
                strategy=strategy,
                max_chunk_tokens=self.config.default_chunk_size,
            )

        # Create or use provided sandbox
        self.sandbox = sandbox or SandboxEnvironment(
            timeout=self.config.execution.timeout,
            memory_limit_mb=self.config.execution.memory_limit_mb,
        )

        logger.info(
            f"RecursiveInferenceEngine initialized: "
            f"max_depth={self.config.max_recursion_depth}, "
            f"llm={self.llm}"
        )

    async def process(
        self,
        query: str,
        context: str,
        max_depth: int | None = None,
    ) -> InferenceResult:
        """Process a query with potentially unlimited context.

        This is the main entry point for recursive inference.

        Args:
            query: User's question or task
            context: Input context (can be arbitrarily long)
            max_depth: Override max recursion depth

        Returns:
            InferenceResult with answer and processing details

        Raises:
            RLMException: If processing fails
        """
        start_time = time.time()
        max_depth = max_depth or self.config.max_recursion_depth

        logger.info(f"Processing query: '{query[:50]}...' (context: {len(context)} chars)")

        # Step 1: Load context
        try:
            self.context_manager.load_context(
                context,
                max_chunk_tokens=self.config.default_chunk_size,
            )
        except Exception as e:
            raise RLMException(f"Failed to load context: {e}") from e

        context_summary = self.context_manager.get_context_summary()
        logger.info(f"Context loaded: {context_summary.total_tokens} tokens")

        # Step 2: Initialize processing state
        state = ProcessingState(
            query=query,
            max_depth=max_depth,
        )

        # Step 3: Process recursively
        try:
            root_node = await self._recursive_step(
                query=query,
                depth=0,
                state=state,
            )
        except MaxDepthExceededError:
            raise
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise RLMException(f"Processing failed: {e}") from e

        # Step 4: Build result
        total_time = time.time() - start_time

        result = InferenceResult(
            answer=root_node.result or "No result generated",
            recursion_tree=root_node,
            total_tokens=state.total_tokens,
            execution_time=total_time,
            num_recursive_calls=len(state.nodes),
            metadata={
                "context_tokens": context_summary.total_tokens,
                "context_chunks": context_summary.num_chunks,
                "max_depth_used": self._get_max_depth(root_node),
            },
        )

        logger.info(
            f"Processing complete: {result.num_recursive_calls} calls, "
            f"{result.total_tokens} tokens, {result.execution_time:.2f}s"
        )

        return result

    async def _recursive_step(
        self,
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> RecursionNode:
        """Execute one step of recursive processing.

        Args:
            query: Query for this level
            depth: Current recursion depth
            state: Processing state

        Returns:
            RecursionNode with results

        Raises:
            MaxDepthExceededError: If depth exceeds limit
        """
        step_start = time.time()

        # Check depth limit
        if depth > state.max_depth:
            raise MaxDepthExceededError(depth, state.max_depth)

        logger.debug(f"Recursive step: depth={depth}, query='{query[:30]}...'")

        # Create node
        node = RecursionNode(
            depth=depth,
            query=query,
            context_summary=self.context_manager.get_context_summary().to_prompt_string(),
        )

        try:
            # Check if context is small enough for direct processing
            context_length = self.context_manager.get_context_length()

            if context_length <= self.config.default_chunk_size and depth == 0:
                # Direct processing - context fits in one call
                logger.debug("Using direct processing (context fits)")
                result = await self._process_direct(query, state)
                node.result = result
            else:
                # Generate and execute code for recursive processing
                code = await self._generate_code(query, depth, state)
                node.generated_code = code

                # Execute in sandbox
                result = await self._execute_code(code, depth, state)

                # If result contains pending submodel calls, process them
                if isinstance(result, str) and "[PENDING_SUBMODEL:" in result:
                    result = await self._process_submodel_calls(
                        result, query, depth, state, node
                    )

                node.result = str(result) if result else "No result"

        except Exception as e:
            logger.error(f"Step failed at depth {depth}: {e}")
            node.error = str(e)
            # Try fallback
            node.result = await self._fallback_processing(query, depth, state)

        node.execution_time = time.time() - step_start
        state.nodes.append(node)

        return node

    async def _process_direct(
        self,
        query: str,
        state: ProcessingState,
    ) -> str:
        """Process query directly without code generation.

        Used when context is small enough to fit in one LLM call.
        """
        # Get full context
        full_context = self.context_manager.get_chunk_range(
            0, self.context_manager.metadata.total_length if self.context_manager.metadata else 0
        )

        prompt = DIRECT_PROCESSING_PROMPT.format(
            query=query,
            context=full_context[:100000],  # Limit for safety
        )

        response = await self.llm.generate(prompt)
        state.add_tokens(response.tokens_used)

        return response.content

    async def _generate_code(
        self,
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> str:
        """Generate Python code for processing.

        Args:
            query: Current query
            depth: Current depth
            state: Processing state

        Returns:
            Generated Python code
        """
        context_summary = self.context_manager.get_context_summary()

        user_prompt = CODE_GENERATION_USER_TEMPLATE.format(
            query=query,
            context_length=context_summary.total_tokens,
            context_structure=context_summary.structure,
        )

        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
            temperature=0.3,  # Lower for more deterministic code
        )

        state.add_tokens(response.tokens_used)

        # Extract code from response
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        logger.debug(f"Generated code ({len(code)} chars)")
        return code.strip()

    async def _execute_code(
        self,
        code: str,
        depth: int,
        state: ProcessingState,
    ) -> Any:
        """Execute generated code in sandbox.

        Args:
            code: Python code to execute
            depth: Current depth
            state: Processing state

        Returns:
            Execution result
        """
        # Build context API
        api_builder = ContextAPIBuilder(
            context_manager=self.context_manager,
            engine=self,
            current_depth=depth,
            max_depth=state.max_depth,
        )
        context_api = api_builder.build()

        # Execute in sandbox
        result = await self.sandbox.execute(code, context_api)

        return result

    async def _process_submodel_calls(
        self,
        result: str,
        query: str,
        depth: int,
        state: ProcessingState,
        parent_node: RecursionNode,
    ) -> str:
        """Process pending submodel calls from code execution.

        When the executed code calls call_submodel(), we need to
        actually make those recursive calls.
        """
        # This is where we would implement actual recursive calls
        # For now, use the LLM directly for sub-queries

        # Simple approach: make direct LLM call for each pending call
        if depth < state.max_depth:
            # Extract pending calls and process them
            submodel_results = []
            chunks = self.context_manager.chunks

            for i, chunk in enumerate(chunks[:5]):  # Limit chunks
                sub_query = f"Based on this chunk, answer: {query}"
                sub_prompt = f"Chunk {i+1}:\n{chunk.content[:4000]}\n\nQuery: {sub_query}"

                response = await self.llm.generate(sub_prompt)
                state.add_tokens(response.tokens_used)
                submodel_results.append(response.content)

                # Create child node
                child_node = RecursionNode(
                    depth=depth + 1,
                    query=sub_query,
                    result=response.content,
                    tokens_used=response.tokens_used,
                )
                parent_node.sub_calls.append(child_node)

            # Aggregate results
            if submodel_results:
                return await self._aggregate_results(submodel_results, query, state)

        return result

    async def _aggregate_results(
        self,
        results: list[str],
        query: str,
        state: ProcessingState,
    ) -> str:
        """Aggregate results from multiple recursive calls.

        Args:
            results: List of results to aggregate
            query: Original query
            state: Processing state

        Returns:
            Aggregated result
        """
        if not results:
            return ""

        if len(results) == 1:
            return results[0]

        # Use LLM to aggregate
        chunk_results = "\n\n---\n\n".join(
            f"Result {i+1}:\n{r}" for i, r in enumerate(results)
        )

        prompt = AGGREGATION_PROMPT.format(
            query=query,
            chunk_results=chunk_results,
        )

        response = await self.llm.generate(prompt)
        state.add_tokens(response.tokens_used)

        return response.content

    async def _fallback_processing(
        self,
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> str:
        """Fallback when code execution fails.

        Uses simple chunking and direct LLM calls.
        """
        logger.warning(f"Using fallback processing at depth {depth}")

        # Get first few chunks
        results = []
        for chunk in self.context_manager.chunks[:3]:
            prompt = f"Context:\n{chunk.content[:4000]}\n\nQuery: {query}"
            response = await self.llm.generate(prompt)
            state.add_tokens(response.tokens_used)
            results.append(response.content)

        return await self._aggregate_results(results, query, state)

    def _get_max_depth(self, node: RecursionNode) -> int:
        """Get maximum depth reached in recursion tree."""
        if not node.sub_calls:
            return node.depth
        return max(self._get_max_depth(sub) for sub in node.sub_calls)
