"""Transparent execution mode for full visibility into RLM processing.

This module provides complete transparency into every step of the recursive
inference process, showing:
- Context loading and chunking
- Code generation (with the actual prompt and response)
- Code validation results
- Sandbox execution
- Recursive calls
- Result aggregation
- Token usage and timing

Usage:
    engine = TransparentEngine(llm, config, callback=print_callback)
    result = await engine.process(query, context)
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from rlm.config import RLMConfig
from rlm.context import ContextManager
from rlm.context.chunking import get_chunking_strategy
from rlm.core.models import InferenceResult, ProcessingState, RecursionNode
from rlm.exceptions import MaxDepthExceededError, RLMException
from rlm.execution import CodeValidator, SandboxEnvironment
from rlm.execution.sandbox import ContextAPIBuilder
from rlm.models.base import BaseLLM, LLMResponse
from rlm.models.prompts import (
    AGGREGATION_PROMPT,
    CODE_GENERATION_SYSTEM_PROMPT,
    CODE_GENERATION_USER_TEMPLATE,
    DIRECT_PROCESSING_PROMPT,
)


class EventType(Enum):
    """Types of events during RLM processing."""

    # Lifecycle events
    PROCESS_START = "process_start"
    PROCESS_END = "process_end"

    # Context events
    CONTEXT_LOADING = "context_loading"
    CONTEXT_LOADED = "context_loaded"
    CONTEXT_CHUNKED = "context_chunked"

    # LLM events
    LLM_PROMPT_PREPARED = "llm_prompt_prepared"
    LLM_THINKING = "llm_thinking"
    LLM_RESPONSE_RECEIVED = "llm_response_received"

    # Code events
    CODE_GENERATED = "code_generated"
    CODE_VALIDATING = "code_validating"
    CODE_VALIDATION_RESULT = "code_validation_result"
    CODE_EXECUTING = "code_executing"
    CODE_EXECUTION_RESULT = "code_execution_result"

    # Recursion events
    RECURSION_START = "recursion_start"
    RECURSION_STEP = "recursion_step"
    RECURSION_SUBMODEL_CALL = "recursion_submodel_call"
    RECURSION_END = "recursion_end"

    # Aggregation events
    AGGREGATION_START = "aggregation_start"
    AGGREGATION_END = "aggregation_end"

    # Error events
    ERROR = "error"
    FALLBACK_TRIGGERED = "fallback_triggered"


@dataclass
class Event:
    """An event during RLM processing.

    Attributes:
        type: The type of event
        timestamp: When the event occurred
        depth: Current recursion depth
        data: Event-specific data
        duration_ms: Duration in milliseconds (for completed operations)
    """
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    depth: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def __str__(self) -> str:
        indent = "  " * self.depth
        time_str = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        duration = f" ({self.duration_ms:.1f}ms)" if self.duration_ms else ""
        return f"[{time_str}]{indent} {self.type.value}{duration}"


# Type alias for callback functions
EventCallback = Callable[[Event], None]


def default_callback(event: Event) -> None:
    """Default callback that prints events to console with formatting."""
    indent = "  " * event.depth
    time_str = event.timestamp.strftime("%H:%M:%S.%f")[:-3]

    # Color codes for different event types
    colors = {
        EventType.PROCESS_START: "\033[1;32m",      # Bold Green
        EventType.PROCESS_END: "\033[1;32m",        # Bold Green
        EventType.CONTEXT_LOADING: "\033[34m",      # Blue
        EventType.CONTEXT_LOADED: "\033[34m",       # Blue
        EventType.CONTEXT_CHUNKED: "\033[34m",      # Blue
        EventType.LLM_PROMPT_PREPARED: "\033[33m",  # Yellow
        EventType.LLM_THINKING: "\033[33m",         # Yellow
        EventType.LLM_RESPONSE_RECEIVED: "\033[33m",# Yellow
        EventType.CODE_GENERATED: "\033[35m",       # Magenta
        EventType.CODE_VALIDATING: "\033[35m",      # Magenta
        EventType.CODE_VALIDATION_RESULT: "\033[35m",# Magenta
        EventType.CODE_EXECUTING: "\033[36m",       # Cyan
        EventType.CODE_EXECUTION_RESULT: "\033[36m",# Cyan
        EventType.RECURSION_START: "\033[1;34m",    # Bold Blue
        EventType.RECURSION_STEP: "\033[1;34m",     # Bold Blue
        EventType.RECURSION_SUBMODEL_CALL: "\033[1;34m", # Bold Blue
        EventType.RECURSION_END: "\033[1;34m",      # Bold Blue
        EventType.AGGREGATION_START: "\033[32m",    # Green
        EventType.AGGREGATION_END: "\033[32m",      # Green
        EventType.ERROR: "\033[1;31m",              # Bold Red
        EventType.FALLBACK_TRIGGERED: "\033[31m",   # Red
    }
    reset = "\033[0m"
    color = colors.get(event.type, "")

    # Format duration if present
    duration = f" ({event.duration_ms:.1f}ms)" if event.duration_ms else ""

    # Print header
    print(f"{color}[{time_str}]{indent} ═══ {event.type.value.upper()}{duration} ═══{reset}")

    # Print relevant data based on event type
    data = event.data

    if event.type == EventType.PROCESS_START:
        print(f"{indent}   Query: {data.get('query', '')[:100]}...")
        print(f"{indent}   Context length: {data.get('context_length', 0):,} chars")

    elif event.type == EventType.CONTEXT_CHUNKED:
        print(f"{indent}   Chunks: {data.get('num_chunks', 0)}")
        print(f"{indent}   Total tokens: {data.get('total_tokens', 0):,}")
        print(f"{indent}   Strategy: {data.get('strategy', 'unknown')}")

    elif event.type == EventType.LLM_PROMPT_PREPARED:
        print(f"{indent}   Model: {data.get('model', 'unknown')}")
        print(f"{indent}   Prompt length: {data.get('prompt_length', 0):,} chars")
        if data.get('system_prompt_preview'):
            print(f"{indent}   System prompt: {data['system_prompt_preview'][:80]}...")
        if data.get('user_prompt_preview'):
            print(f"{indent}   User prompt: {data['user_prompt_preview'][:80]}...")

    elif event.type == EventType.LLM_THINKING:
        print(f"{indent}   ⏳ Model is generating response...")
        print(f"{indent}   Model: {data.get('model', 'unknown')}")

    elif event.type == EventType.LLM_RESPONSE_RECEIVED:
        print(f"{indent}   Tokens used: {data.get('tokens_used', 0)}")
        print(f"{indent}   Finish reason: {data.get('finish_reason', 'unknown')}")
        if data.get('response_preview'):
            print(f"{indent}   Response: {data['response_preview'][:100]}...")

    elif event.type == EventType.CODE_GENERATED:
        code = data.get('code', '')
        print(f"{indent}   Code length: {len(code)} chars")
        print(f"{indent}   ┌{'─' * 60}")
        for line in code.split('\n')[:15]:
            print(f"{indent}   │ {line}")
        if code.count('\n') > 15:
            print(f"{indent}   │ ... ({code.count(chr(10)) - 15} more lines)")
        print(f"{indent}   └{'─' * 60}")

    elif event.type == EventType.CODE_VALIDATION_RESULT:
        is_valid = data.get('is_valid', False)
        status = "✅ PASSED" if is_valid else "❌ FAILED"
        print(f"{indent}   Validation: {status}")
        if data.get('errors'):
            for err in data['errors'][:5]:
                print(f"{indent}   ⚠️  {err}")
        if data.get('warnings'):
            for warn in data['warnings'][:3]:
                print(f"{indent}   ℹ️  {warn}")

    elif event.type == EventType.CODE_EXECUTING:
        print(f"{indent}   ⚙️  Executing in sandbox...")
        print(f"{indent}   Timeout: {data.get('timeout', 5)}s")

    elif event.type == EventType.CODE_EXECUTION_RESULT:
        success = data.get('success', False)
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{indent}   Execution: {status}")
        if data.get('result_preview'):
            print(f"{indent}   Result: {data['result_preview'][:100]}...")
        if data.get('error'):
            print(f"{indent}   Error: {data['error']}")

    elif event.type == EventType.RECURSION_STEP:
        print(f"{indent}   Depth: {data.get('current_depth', 0)} / {data.get('max_depth', 1)}")
        print(f"{indent}   Query: {data.get('query', '')[:80]}...")

    elif event.type == EventType.RECURSION_SUBMODEL_CALL:
        print(f"{indent}   📞 Calling sub-model...")
        print(f"{indent}   Chunk size: {data.get('chunk_size', 0):,} chars")
        print(f"{indent}   Sub-query: {data.get('sub_query', '')[:60]}...")

    elif event.type == EventType.AGGREGATION_START:
        print(f"{indent}   Aggregating {data.get('num_results', 0)} results...")

    elif event.type == EventType.AGGREGATION_END:
        print(f"{indent}   Final result length: {data.get('result_length', 0):,} chars")

    elif event.type == EventType.ERROR:
        print(f"{indent}   ❌ ERROR: {data.get('error', 'Unknown error')}")
        if data.get('traceback'):
            print(f"{indent}   Traceback: {data['traceback'][:200]}...")

    elif event.type == EventType.PROCESS_END:
        print(f"{indent}   Total tokens: {data.get('total_tokens', 0):,}")
        print(f"{indent}   Total time: {data.get('total_time', 0):.2f}s")
        print(f"{indent}   Recursive calls: {data.get('num_calls', 0)}")
        if data.get('answer_preview'):
            print(f"{indent}   Answer: {data['answer_preview'][:150]}...")

    print()  # Blank line for readability


class TransparentEngine:
    """RLM Engine with full transparency and visibility.

    This engine wraps the standard RecursiveInferenceEngine but fires
    events at every step, allowing complete visibility into the process.

    Attributes:
        llm: Primary LLM for reasoning
        code_llm: LLM for code generation (can be lighter weight)
        config: RLM configuration
        callback: Function called for each event

    Example:
        >>> def my_callback(event):
        ...     print(f"{event.type}: {event.data}")
        >>>
        >>> engine = TransparentEngine(llm, config, callback=my_callback)
        >>> result = await engine.process("Summarize", long_text)
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: RLMConfig | None = None,
        code_llm: BaseLLM | None = None,
        callback: EventCallback | None = None,
    ) -> None:
        """Initialize transparent engine.

        Args:
            llm: Primary LLM for reasoning (heavyweight recommended)
            config: RLM configuration
            code_llm: LLM for code generation (lightweight OK)
            callback: Function to call for each event
        """
        self.llm = llm
        self.code_llm = code_llm or llm  # Use same LLM if not specified
        self.config = config or RLMConfig()
        self.callback = callback or default_callback

        # Initialize components
        strategy = get_chunking_strategy(self.config.chunking_strategy)
        self.context_manager = ContextManager(
            strategy=strategy,
            max_chunk_tokens=self.config.default_chunk_size,
        )
        self.sandbox = SandboxEnvironment(
            timeout=self.config.execution.timeout,
            memory_limit_mb=self.config.execution.memory_limit_mb,
        )
        self.validator = CodeValidator()

        # Event log
        self.events: list[Event] = []

    def _emit(self, event_type: EventType, depth: int = 0, **data: Any) -> Event:
        """Emit an event and call the callback."""
        event = Event(type=event_type, depth=depth, data=data)
        self.events.append(event)
        if self.callback:
            self.callback(event)
        return event

    def _emit_with_duration(
        self,
        event_type: EventType,
        start_time: float,
        depth: int = 0,
        **data: Any
    ) -> Event:
        """Emit an event with duration calculated from start time."""
        duration_ms = (time.time() - start_time) * 1000
        event = Event(
            type=event_type,
            depth=depth,
            data=data,
            duration_ms=duration_ms
        )
        self.events.append(event)
        if self.callback:
            self.callback(event)
        return event

    async def process(
        self,
        query: str,
        context: str,
        max_depth: int | None = None,
    ) -> InferenceResult:
        """Process a query with full transparency.

        Args:
            query: User's question
            context: Large input context
            max_depth: Override max recursion depth

        Returns:
            InferenceResult with answer and metadata
        """
        process_start = time.time()
        max_depth = max_depth or self.config.max_recursion_depth
        self.events = []  # Reset event log

        # ═══ PROCESS START ═══
        self._emit(
            EventType.PROCESS_START,
            query=query,
            context_length=len(context),
            max_depth=max_depth,
        )

        # ═══ LOAD CONTEXT ═══
        self._emit(EventType.CONTEXT_LOADING, context_length=len(context))

        context_start = time.time()
        try:
            metadata = self.context_manager.load_context(context)
        except Exception as e:
            self._emit(EventType.ERROR, error=str(e))
            raise

        self._emit_with_duration(
            EventType.CONTEXT_CHUNKED,
            context_start,
            num_chunks=metadata.num_chunks,
            total_tokens=metadata.total_tokens,
            strategy=metadata.chunking_strategy,
        )

        # ═══ RECURSIVE PROCESSING ═══
        state = ProcessingState(query=query, max_depth=max_depth)

        try:
            root_node = await self._recursive_step(query, 0, state)
        except Exception as e:
            self._emit(EventType.ERROR, error=str(e))
            raise

        # ═══ PROCESS END ═══
        total_time = time.time() - process_start

        result = InferenceResult(
            answer=root_node.result or "No result generated",
            recursion_tree=root_node,
            total_tokens=state.total_tokens,
            execution_time=total_time,
            num_recursive_calls=len(state.nodes),
            metadata={
                "events": len(self.events),
                "context_tokens": metadata.total_tokens,
            },
        )

        self._emit(
            EventType.PROCESS_END,
            total_tokens=state.total_tokens,
            total_time=total_time,
            num_calls=len(state.nodes),
            answer_preview=result.answer[:200] if result.answer else "",
        )

        return result

    async def _recursive_step(
        self,
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> RecursionNode:
        """Execute one recursive step with full visibility."""
        step_start = time.time()

        # ═══ RECURSION STEP ═══
        self._emit(
            EventType.RECURSION_STEP,
            depth=depth,
            current_depth=depth,
            max_depth=state.max_depth,
            query=query,
        )

        if depth > state.max_depth:
            raise MaxDepthExceededError(depth, state.max_depth)

        node = RecursionNode(
            depth=depth,
            query=query,
            context_summary=self.context_manager.get_context_summary().to_prompt_string(),
        )

        try:
            context_length = self.context_manager.get_context_length()

            if context_length <= self.config.default_chunk_size and depth == 0:
                # Direct processing
                result = await self._process_direct(query, depth, state)
                node.result = result
            else:
                # Generate and execute code
                code = await self._generate_code(query, depth, state)
                node.generated_code = code

                # Validate code
                validation_result = await self._validate_code(code, depth)

                if validation_result:
                    # Execute code
                    exec_result = await self._execute_code(code, depth, state)

                    # Process any submodel calls
                    if isinstance(exec_result, str) and "[PENDING_SUBMODEL:" in exec_result:
                        exec_result = await self._process_submodel_calls(
                            exec_result, query, depth, state, node
                        )

                    node.result = str(exec_result) if exec_result else "No result"
                else:
                    # Fallback
                    self._emit(EventType.FALLBACK_TRIGGERED, depth=depth, reason="validation_failed")
                    node.result = await self._fallback_processing(query, depth, state)

        except Exception as e:
            self._emit(EventType.ERROR, depth=depth, error=str(e))
            node.error = str(e)
            node.result = await self._fallback_processing(query, depth, state)

        node.execution_time = time.time() - step_start
        state.nodes.append(node)

        return node

    async def _process_direct(
        self,
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> str:
        """Process directly without code generation."""
        full_context = self.context_manager.get_chunk_range(
            0,
            self.context_manager.metadata.total_length if self.context_manager.metadata else 0
        )

        prompt = DIRECT_PROCESSING_PROMPT.format(
            query=query,
            context=full_context[:50000],
        )

        # ═══ LLM CALL ═══
        self._emit(
            EventType.LLM_PROMPT_PREPARED,
            depth=depth,
            model=self.llm.model,
            prompt_length=len(prompt),
            user_prompt_preview=prompt[:200],
        )

        self._emit(EventType.LLM_THINKING, depth=depth, model=self.llm.model)

        llm_start = time.time()
        response = await self.llm.generate(prompt)

        self._emit_with_duration(
            EventType.LLM_RESPONSE_RECEIVED,
            llm_start,
            depth=depth,
            tokens_used=response.tokens_used,
            finish_reason=response.finish_reason,
            response_preview=response.content[:300],
        )

        state.add_tokens(response.tokens_used)
        return response.content

    async def _generate_code(
        self,
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> str:
        """Generate code with full visibility."""
        context_summary = self.context_manager.get_context_summary()

        user_prompt = CODE_GENERATION_USER_TEMPLATE.format(
            query=query,
            context_length=context_summary.total_tokens,
            context_structure=context_summary.structure,
        )

        # ═══ CODE GENERATION LLM CALL ═══
        self._emit(
            EventType.LLM_PROMPT_PREPARED,
            depth=depth,
            model=self.code_llm.model,
            prompt_length=len(user_prompt) + len(CODE_GENERATION_SYSTEM_PROMPT),
            system_prompt_preview=CODE_GENERATION_SYSTEM_PROMPT[:150],
            user_prompt_preview=user_prompt[:150],
        )

        self._emit(
            EventType.LLM_THINKING,
            depth=depth,
            model=self.code_llm.model,
            task="code_generation",
        )

        llm_start = time.time()
        response = await self.code_llm.generate(
            prompt=user_prompt,
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
            temperature=0.3,
        )

        self._emit_with_duration(
            EventType.LLM_RESPONSE_RECEIVED,
            llm_start,
            depth=depth,
            tokens_used=response.tokens_used,
            finish_reason=response.finish_reason,
            response_preview=response.content[:200],
        )

        state.add_tokens(response.tokens_used)

        # Extract code
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        code = code.strip()

        # ═══ CODE GENERATED ═══
        self._emit(EventType.CODE_GENERATED, depth=depth, code=code)

        return code

    async def _validate_code(self, code: str, depth: int) -> bool:
        """Validate code with visibility."""
        self._emit(EventType.CODE_VALIDATING, depth=depth, code_length=len(code))

        result = self.validator.validate(code)

        self._emit(
            EventType.CODE_VALIDATION_RESULT,
            depth=depth,
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
        )

        return result.is_valid

    async def _execute_code(
        self,
        code: str,
        depth: int,
        state: ProcessingState,
    ) -> Any:
        """Execute code with visibility."""
        self._emit(
            EventType.CODE_EXECUTING,
            depth=depth,
            timeout=self.config.execution.timeout,
        )

        # Build context API
        api_builder = ContextAPIBuilder(
            context_manager=self.context_manager,
            engine=self,
            current_depth=depth,
            max_depth=state.max_depth,
        )
        context_api = api_builder.build()

        exec_start = time.time()
        try:
            result = await self.sandbox.execute(code, context_api, validate=False)

            self._emit_with_duration(
                EventType.CODE_EXECUTION_RESULT,
                exec_start,
                depth=depth,
                success=True,
                result_preview=str(result)[:200] if result else "",
            )

            return result

        except Exception as e:
            self._emit_with_duration(
                EventType.CODE_EXECUTION_RESULT,
                exec_start,
                depth=depth,
                success=False,
                error=str(e),
            )
            raise

    async def _process_submodel_calls(
        self,
        result: str,
        query: str,
        depth: int,
        state: ProcessingState,
        parent_node: RecursionNode,
    ) -> str:
        """Process submodel calls with visibility."""
        if depth >= state.max_depth:
            return result

        submodel_results = []
        chunks = self.context_manager.chunks

        for i, chunk in enumerate(chunks[:5]):
            # ═══ SUBMODEL CALL ═══
            self._emit(
                EventType.RECURSION_SUBMODEL_CALL,
                depth=depth,
                chunk_index=i,
                chunk_size=len(chunk.content),
                sub_query=query[:60],
            )

            sub_prompt = f"Chunk {i+1}:\n{chunk.content[:4000]}\n\nQuery: {query}"

            self._emit(
                EventType.LLM_THINKING,
                depth=depth + 1,
                model=self.llm.model,
                task=f"submodel_chunk_{i}",
            )

            llm_start = time.time()
            response = await self.llm.generate(sub_prompt)

            self._emit_with_duration(
                EventType.LLM_RESPONSE_RECEIVED,
                llm_start,
                depth=depth + 1,
                tokens_used=response.tokens_used,
                response_preview=response.content[:100],
            )

            state.add_tokens(response.tokens_used)
            submodel_results.append(response.content)

            # Add child node
            child_node = RecursionNode(
                depth=depth + 1,
                query=f"Process chunk {i+1}",
                result=response.content,
                tokens_used=response.tokens_used,
            )
            parent_node.sub_calls.append(child_node)

        # Aggregate
        if submodel_results:
            return await self._aggregate_results(submodel_results, query, depth, state)

        return result

    async def _aggregate_results(
        self,
        results: list[str],
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> str:
        """Aggregate results with visibility."""
        self._emit(
            EventType.AGGREGATION_START,
            depth=depth,
            num_results=len(results),
        )

        if len(results) == 1:
            return results[0]

        chunk_results = "\n\n---\n\n".join(
            f"Result {i+1}:\n{r}" for i, r in enumerate(results)
        )

        prompt = AGGREGATION_PROMPT.format(query=query, chunk_results=chunk_results)

        self._emit(EventType.LLM_THINKING, depth=depth, model=self.llm.model, task="aggregation")

        agg_start = time.time()
        response = await self.llm.generate(prompt)

        self._emit_with_duration(
            EventType.LLM_RESPONSE_RECEIVED,
            agg_start,
            depth=depth,
            tokens_used=response.tokens_used,
        )

        state.add_tokens(response.tokens_used)

        self._emit(
            EventType.AGGREGATION_END,
            depth=depth,
            result_length=len(response.content),
        )

        return response.content

    async def _fallback_processing(
        self,
        query: str,
        depth: int,
        state: ProcessingState,
    ) -> str:
        """Fallback with visibility."""
        self._emit(EventType.FALLBACK_TRIGGERED, depth=depth, reason="error_recovery")

        results = []
        for chunk in self.context_manager.chunks[:3]:
            prompt = f"Context:\n{chunk.content[:4000]}\n\nQuery: {query}"
            response = await self.llm.generate(prompt)
            state.add_tokens(response.tokens_used)
            results.append(response.content)

        return await self._aggregate_results(results, query, depth, state)

    def get_event_summary(self) -> dict[str, Any]:
        """Get a summary of all events."""
        by_type: dict[str, int] = {}
        total_llm_time = 0.0
        total_exec_time = 0.0

        for event in self.events:
            by_type[event.type.value] = by_type.get(event.type.value, 0) + 1
            if event.duration_ms:
                if "llm" in event.type.value.lower():
                    total_llm_time += event.duration_ms
                elif "exec" in event.type.value.lower():
                    total_exec_time += event.duration_ms

        return {
            "total_events": len(self.events),
            "by_type": by_type,
            "total_llm_time_ms": total_llm_time,
            "total_exec_time_ms": total_exec_time,
        }
