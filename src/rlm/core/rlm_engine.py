"""Paper-accurate RLM Engine with visibility.

This engine follows the 2025 RLM paper more closely:
1. Model gets RAW context string (discovers chunking via code)
2. FINAL() detection for completion signaling
3. Complexity detection (heavyweight decides if RLM is needed)
4. Simple event tracking for visibility

Key difference from original engine.py:
- NO pre-chunking - model writes code to decompose context
- Heavyweight LLM can skip RLM for simple tasks
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from rlm.config import RLMConfig
from rlm.exceptions import CodeExecutionError, MaxDepthExceededError, RLMException
from rlm.execution import SandboxEnvironment
from rlm.execution.validator import CodeValidator
from rlm.models.base import BaseLLM
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# EVENTS - Simple, 6 types only
# =============================================================================

class EventType(Enum):
    """Simple event types for visibility."""
    PROCESS_START = "process_start"
    COMPLEXITY_CHECK = "complexity_check"
    CODE_GENERATED = "code_generated"
    CODE_EXECUTED = "code_executed"
    LLM_CALL = "llm_call"
    PROCESS_END = "process_end"


@dataclass
class Event:
    """A single event in the processing pipeline."""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.type.value}: {self.data}"


@dataclass
class Trajectory:
    """Paper-style trajectory tracking."""
    query: str
    context_length: int
    events: list[Event] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    total_cost: float = 0.0
    total_tokens: int = 0
    execution_time: float = 0.0
    used_rlm: bool = False

    def add_event(self, event_type: EventType, **data: Any) -> None:
        """Add an event to the trajectory."""
        event = Event(type=event_type, data=data)
        self.events.append(event)
        logger.debug(str(event))

    def add_turn(self, llm_output: str, code: str | None, result: Any) -> None:
        """Add a turn (LLM output + execution result)."""
        self.turns.append({
            "step": len(self.turns) + 1,
            "llm_output": llm_output[:500] + "..." if len(llm_output) > 500 else llm_output,
            "code": code,
            "result": str(result)[:500] if result else None,
        })


@dataclass
class RLMResult:
    """Result from RLM processing."""
    answer: str
    trajectory: Trajectory
    used_rlm: bool

    @property
    def total_tokens(self) -> int:
        return self.trajectory.total_tokens

    @property
    def execution_time(self) -> float:
        return self.trajectory.execution_time


# =============================================================================
# PROMPTS - Paper-accurate with FINAL() convention
# =============================================================================

COMPLEXITY_CHECK_PROMPT = """Assess if this task requires recursive context processing (RLM).

Query: {query}
Context length: {context_length:,} characters (~{token_estimate:,} tokens)

RLM is NEEDED when:
- Context is too large to process at once (>30K tokens)
- Task requires analyzing/comparing multiple sections
- Task requires aggregating information across the document

RLM is NOT needed when:
- Context fits in one pass
- Simple lookup or extraction
- Query can be answered from small portion

Respond with exactly one line:
USE_RLM: <brief reason>
or
DIRECT: <brief reason>"""

CODE_GEN_SYSTEM_PROMPT = """You are an RLM (Recursive Language Model). You write Python code to process large contexts.

## Available Variables and Functions

```python
context      # The FULL context as a string - you decide how to decompose it
llm_query(text, question) -> str   # Call sub-model on a chunk
len(context) # Get context length
```

## Your Job
Write Python code that:
1. Examines the context structure
2. Decides how to split it (YOU discover the chunking strategy)
3. Calls llm_query() on relevant chunks
4. Aggregates results

## Completion Signal
When you have the final answer, use:
```python
FINAL("Your answer here")
```
Or reference a variable:
```python
answer = "computed result"
FINAL_VAR(answer)
```

## Example - Summarizing a long document
```python
# Discover structure
paragraphs = context.split('\\n\\n')

if len(paragraphs) < 5:
    # Small enough to process directly
    result = llm_query(context, "Summarize this document")
    FINAL(result)
else:
    # Split and process
    chunk_size = len(paragraphs) // 3
    chunks = [
        '\\n\\n'.join(paragraphs[i:i+chunk_size])
        for i in range(0, len(paragraphs), chunk_size)
    ]

    summaries = []
    for i, chunk in enumerate(chunks):
        summary = llm_query(chunk, f"Summarize section {i+1}")
        summaries.append(summary)

    # Aggregate
    combined = '\\n---\\n'.join(summaries)
    final = llm_query(combined, "Combine these summaries into one coherent summary")
    FINAL(final)
```

## Rules
- You MUST use FINAL() or FINAL_VAR() to signal completion
- Do NOT use imports
- Do NOT use print()
- Keep code simple and focused"""

CODE_GEN_USER_PROMPT = """Query: {query}

Context length: {context_length:,} characters
Context preview (first 500 chars):
{context_preview}

Write Python code to answer the query. Remember to use FINAL() when done."""


# =============================================================================
# API BUILDER - Paper-accurate (raw context + llm_query)
# =============================================================================

class PaperAccurateAPI:
    """Builds the API that generated code has access to.

    Paper-accurate: gives raw context string, not pre-chunked.
    """

    def __init__(
        self,
        raw_context: str,
        sub_llm: BaseLLM,
        current_depth: int = 0,
        max_depth: int = 1,
        on_llm_call: Callable[[str, str, str], None] | None = None,
        trajectory: "Trajectory | None" = None,
    ) -> None:
        self.raw_context = raw_context
        self.sub_llm = sub_llm
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.on_llm_call = on_llm_call
        self.trajectory = trajectory
        self.call_count = 0
        self.tokens_used = 0

    def _llm_query(self, text: str, question: str) -> str:
        """Make a sub-model call synchronously.

        This executes the LLM call immediately so generated code can use
        the result in subsequent operations (like aggregation).
        """
        import asyncio

        if self.current_depth >= self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached")
            return f"[Depth limit reached. Text preview: {text[:200]}...]"

        # Truncate if too long
        if len(text) > 50000:
            text = text[:50000] + "\n[Truncated...]"

        prompt = f"""Answer this question based on the provided text.

Question: {question}

Text:
{text}

Answer:"""

        self.call_count += 1
        logger.debug(f"llm_query call #{self.call_count}: {question[:50]}...")

        # Make the call synchronously (we're in a thread pool)
        try:
            response = asyncio.run(self.sub_llm.generate(prompt))
            self.tokens_used += response.tokens_used
            if self.trajectory:
                self.trajectory.total_tokens += response.tokens_used
            return response.content
        except Exception as e:
            logger.error(f"llm_query failed: {e}")
            return f"[Error: {e}]"

    def build_globals(self) -> dict[str, Any]:
        """Build the globals dict for code execution."""
        # FINAL() detection helpers
        final_result = {"value": None, "is_set": False}

        def FINAL(value: str) -> None:
            final_result["value"] = value
            final_result["is_set"] = True

        def FINAL_VAR(var_name: str) -> None:
            # This gets replaced at runtime
            final_result["var_name"] = var_name
            final_result["is_set"] = True

        return {
            # Raw context - model discovers chunking
            "context": self.raw_context,

            # Sub-model calls
            "llm_query": self._llm_query,

            # Completion signals
            "FINAL": FINAL,
            "FINAL_VAR": FINAL_VAR,
            "_final_result": final_result,

            # Basic Python builtins for string manipulation
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "zip": zip,

            # String operations - model needs these to discover chunking
            "re": __import__("re"),
        }


# =============================================================================
# MAIN ENGINE
# =============================================================================

class RLMEngine:
    """Paper-accurate RLM engine with complexity detection.

    Key features:
    - Heavyweight LLM decides if RLM is needed (complexity check)
    - Model gets RAW context string (discovers chunking)
    - FINAL() detection for completion
    - Simple event tracking

    Args:
        heavyweight_llm: For complexity check and main reasoning (Claude Opus/Sonnet)
        lightweight_llm: For sub-model calls (cheaper/faster model)
        config: RLM configuration
    """

    def __init__(
        self,
        heavyweight_llm: BaseLLM,
        lightweight_llm: BaseLLM | None = None,
        config: RLMConfig | None = None,
    ) -> None:
        self.heavyweight = heavyweight_llm
        self.lightweight = lightweight_llm or heavyweight_llm
        self.config = config or RLMConfig()
        # Use longer timeout for RLM (LLM calls are synchronous)
        rlm_timeout = max(self.config.execution.timeout, 300)
        self.sandbox = SandboxEnvironment(
            timeout=rlm_timeout,
            memory_limit_mb=self.config.execution.memory_limit_mb,
            validator=CodeValidator(),
        )

        logger.info(
            f"RLMEngine initialized: heavyweight={heavyweight_llm}, "
            f"lightweight={lightweight_llm or 'same'}"
        )

    async def process(
        self,
        query: str,
        context: str,
        force_rlm: bool = False,
        on_event: Callable[[Event], None] | None = None,
    ) -> RLMResult:
        """Process a query with context.

        Args:
            query: The user's question
            context: The full context (can be arbitrarily long)
            force_rlm: Skip complexity check, always use RLM
            on_event: Callback for events (for UI visibility)

        Returns:
            RLMResult with answer and trajectory
        """
        start_time = time.time()

        # Initialize trajectory
        trajectory = Trajectory(
            query=query,
            context_length=len(context),
        )

        def emit(event_type: EventType, **data: Any) -> None:
            trajectory.add_event(event_type, **data)
            if on_event:
                on_event(trajectory.events[-1])

        emit(EventType.PROCESS_START, query=query[:100], context_chars=len(context))

        try:
            # Step 1: Complexity check (unless forced)
            use_rlm = force_rlm
            if not force_rlm:
                use_rlm = await self._check_complexity(query, context, emit, trajectory)

            trajectory.used_rlm = use_rlm

            # Step 2: Process
            if use_rlm:
                answer = await self._run_rlm(query, context, emit, trajectory)
            else:
                answer = await self._direct_answer(query, context, emit, trajectory)

            trajectory.final_answer = answer
            trajectory.execution_time = time.time() - start_time

            emit(
                EventType.PROCESS_END,
                answer_preview=answer[:200],
                total_tokens=trajectory.total_tokens,
                execution_time=trajectory.execution_time,
                used_rlm=use_rlm,
            )

            return RLMResult(
                answer=answer,
                trajectory=trajectory,
                used_rlm=use_rlm,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            trajectory.execution_time = time.time() - start_time
            emit(EventType.PROCESS_END, error=str(e))
            raise

    async def _check_complexity(
        self,
        query: str,
        context: str,
        emit: Callable,
        trajectory: Trajectory,
    ) -> bool:
        """Use heavyweight LLM to decide if RLM is needed."""
        # Quick heuristic first
        token_estimate = len(context) // 4  # Rough estimate

        if token_estimate < 5000:
            emit(EventType.COMPLEXITY_CHECK, decision="DIRECT", reason="Small context")
            return False

        if token_estimate > 100000:
            emit(EventType.COMPLEXITY_CHECK, decision="USE_RLM", reason="Very large context")
            return True

        # Ask heavyweight to decide
        prompt = COMPLEXITY_CHECK_PROMPT.format(
            query=query,
            context_length=len(context),
            token_estimate=token_estimate,
        )

        response = await self.heavyweight.generate(prompt, temperature=0.1)
        trajectory.total_tokens += response.tokens_used

        decision = response.content.strip().upper()
        use_rlm = decision.startswith("USE_RLM")
        reason = response.content.split(":", 1)[-1].strip() if ":" in response.content else ""

        emit(
            EventType.COMPLEXITY_CHECK,
            decision="USE_RLM" if use_rlm else "DIRECT",
            reason=reason,
            tokens=response.tokens_used,
        )

        return use_rlm

    async def _direct_answer(
        self,
        query: str,
        context: str,
        emit: Callable,
        trajectory: Trajectory,
    ) -> str:
        """Answer directly without RLM (context fits in one call)."""
        # Truncate if needed
        max_context = 100000  # ~25K tokens
        if len(context) > max_context:
            context = context[:max_context] + "\n\n[Context truncated...]"

        prompt = f"""Answer this query based on the context.

Query: {query}

Context:
{context}

Provide a clear, accurate answer."""

        emit(EventType.LLM_CALL, type="direct", context_chars=len(context))

        response = await self.heavyweight.generate(prompt)
        trajectory.total_tokens += response.tokens_used
        trajectory.add_turn(llm_output=response.content, code=None, result=None)

        return response.content

    async def _run_rlm(
        self,
        query: str,
        context: str,
        emit: Callable,
        trajectory: Trajectory,
    ) -> str:
        """Run the RLM pipeline: generate code, execute with sync LLM calls."""

        # Step 1: Generate code
        code = await self._generate_code(query, context, emit, trajectory)

        # Step 2: Execute code (llm_query calls are made synchronously during execution)
        result, api = await self._execute_code(code, context, emit, trajectory)

        # Step 3: Check for FINAL() result
        if isinstance(result, dict) and result.get("is_set"):
            return str(result.get("value", result.get("var_name", "")))

        return str(result) if result else "No result generated"

    async def _generate_code(
        self,
        query: str,
        context: str,
        emit: Callable,
        trajectory: Trajectory,
    ) -> str:
        """Generate Python code for context processing."""
        user_prompt = CODE_GEN_USER_PROMPT.format(
            query=query,
            context_length=len(context),
            context_preview=context[:500],
        )

        response = await self.heavyweight.generate(
            prompt=user_prompt,
            system_prompt=CODE_GEN_SYSTEM_PROMPT,
            temperature=0.3,
        )
        trajectory.total_tokens += response.tokens_used

        # Extract code from response
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        code = code.strip()

        emit(
            EventType.CODE_GENERATED,
            code_preview=code[:300],
            code_length=len(code),
            tokens=response.tokens_used,
        )

        trajectory.add_turn(llm_output=response.content, code=code, result=None)

        return code

    async def _execute_code(
        self,
        code: str,
        context: str,
        emit: Callable,
        trajectory: Trajectory,
    ) -> tuple[Any, PaperAccurateAPI]:
        """Execute generated code in sandbox."""
        api = PaperAccurateAPI(
            raw_context=context,
            sub_llm=self.lightweight,
            current_depth=0,
            max_depth=self.config.max_recursion_depth,
            trajectory=trajectory,  # For token tracking
        )

        # Build the API that code will have access to
        context_api = api.build_globals()

        try:
            # Execute in sandbox with our API
            # llm_query calls are made synchronously during execution
            await self.sandbox.execute(code, context_api, validate=True)

            # Get FINAL() result from the context_api dict
            result = context_api.get("_final_result", {})

            emit(
                EventType.CODE_EXECUTED,
                success=True,
                llm_calls=api.call_count,
                has_final=result.get("is_set", False),
            )

            return result, api

        except Exception as e:
            emit(EventType.CODE_EXECUTED, success=False, error=str(e))
            raise CodeExecutionError(f"Execution failed: {e}", code=code) from e

    async def _process_pending_calls(
        self,
        current_result: Any,
        pending_calls: list[tuple[str, str]],
        emit: Callable,
        trajectory: Trajectory,
    ) -> Any:
        """Process pending llm_query() calls from executed code."""
        results = {}

        for i, (text, question) in enumerate(pending_calls):
            # Truncate text if too long
            if len(text) > 50000:
                text = text[:50000] + "\n[Truncated...]"

            prompt = f"""Answer this question based on the provided text.

Question: {question}

Text:
{text}

Answer:"""

            emit(EventType.LLM_CALL, type="sub_call", call_id=i, text_chars=len(text))

            response = await self.lightweight.generate(prompt)
            trajectory.total_tokens += response.tokens_used
            results[f"__PENDING_LLM_CALL_{i}__"] = response.content

        # Replace placeholders in result
        if isinstance(current_result, dict) and current_result.get("value"):
            value = str(current_result["value"])
            for placeholder, replacement in results.items():
                value = value.replace(placeholder, replacement)
            current_result["value"] = value

        return current_result
