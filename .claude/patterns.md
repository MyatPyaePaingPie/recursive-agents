# Coding Patterns for RLM Project

## Project-Specific Patterns

### Pattern 1: Async-First Design
All I/O operations should be async by default.

**Good**:
```python
async def process_query(query: str, context: str) -> str:
    result = await llm.generate(query, context)
    return result

# Usage
result = await process_query("question", "context")
```

**Bad**:
```python
def process_query(query: str, context: str) -> str:
    result = llm.generate(query, context)  # Blocking!
    return result
```

### Pattern 2: Pydantic for Data Validation
Use Pydantic models for all data structures.

**Good**:
```python
from pydantic import BaseModel, Field, validator

class RecursionConfig(BaseModel):
    max_depth: int = Field(default=10, ge=1, le=50)
    timeout: int = Field(default=5, gt=0)
    
    @validator('max_depth')
    def validate_depth(cls, v):
        if v > 20:
            logger.warning(f"High recursion depth: {v}")
        return v
```

**Bad**:
```python
# Using plain dict - no validation!
config = {
    'max_depth': 10,
    'timeout': 5
}
```

### Pattern 3: Dependency Injection
Pass dependencies explicitly, don't use globals.

**Good**:
```python
class RecursiveEngine:
    def __init__(
        self,
        llm: BaseLLM,
        executor: CodeExecutor,
        context_manager: ContextManager
    ):
        self.llm = llm
        self.executor = executor
        self.context_manager = context_manager
    
    async def process(self, query: str) -> str:
        code = await self.llm.generate_code(query)
        result = await self.executor.execute(code)
        return result
```

**Bad**:
```python
# Global instances - hard to test!
llm = OpenAI()
executor = CodeExecutor()

class RecursiveEngine:
    def process(self, query: str) -> str:
        code = llm.generate_code(query)
        result = executor.execute(code)
        return result
```

### Pattern 4: Structured Logging
Use structured logging with context.

**Good**:
```python
from loguru import logger

async def execute_code(code: str, context: str) -> ExecutionResult:
    logger.info(
        "Executing generated code",
        code_length=len(code),
        context_length=len(context),
        extra={"code_hash": hash(code)}
    )
    try:
        result = await _execute(code, context)
        logger.info("Execution successful", execution_time=result.time)
        return result
    except Exception as e:
        logger.error("Execution failed", error=str(e), code=code)
        raise
```

**Bad**:
```python
async def execute_code(code: str, context: str) -> ExecutionResult:
    print(f"Executing code: {code}")  # Unstructured!
    result = await _execute(code, context)
    return result
```

### Pattern 5: Custom Exceptions
Define custom exceptions for different error types.

**Good**:
```python
# src/rlm/exceptions.py
class RLMException(Exception):
    """Base exception for RLM system"""
    pass

class MaxDepthExceededError(RLMException):
    """Raised when recursion depth limit exceeded"""
    def __init__(self, depth: int, max_depth: int):
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"Depth {depth} exceeds max {max_depth}")

class CodeExecutionError(RLMException):
    """Raised when generated code fails"""
    def __init__(self, code: str, error: str):
        self.code = code
        self.error = error
        super().__init__(f"Code execution failed: {error}")

# Usage
if depth > max_depth:
    raise MaxDepthExceededError(depth, max_depth)
```

**Bad**:
```python
# Using generic exceptions
if depth > max_depth:
    raise Exception(f"Too deep: {depth}")  # Not descriptive!
```

### Pattern 6: Context Managers for Resources
Use context managers for cleanup.

**Good**:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def sandbox_environment(timeout: int = 5):
    """Create isolated execution environment"""
    sandbox = Sandbox()
    await sandbox.setup()
    try:
        yield sandbox
    finally:
        await sandbox.cleanup()

# Usage
async with sandbox_environment(timeout=10) as sandbox:
    result = await sandbox.execute(code)
```

**Bad**:
```python
async def execute_in_sandbox(code: str):
    sandbox = Sandbox()
    await sandbox.setup()
    result = await sandbox.execute(code)
    # Forgot to cleanup!
    return result
```

### Pattern 7: Type Hints Everywhere
Use comprehensive type hints for better IDE support and catch errors.

**Good**:
```python
from typing import List, Optional, Dict, Any, Callable
from collections.abc import Awaitable

async def aggregate_results(
    results: List[str],
    strategy: str = "concatenate",
    aggregator: Optional[Callable[[List[str]], Awaitable[str]]] = None
) -> str:
    if aggregator:
        return await aggregator(results)
    return "\n".join(results)
```

**Bad**:
```python
async def aggregate_results(results, strategy="concatenate", aggregator=None):
    # No type hints - hard to use!
    if aggregator:
        return await aggregator(results)
    return "\n".join(results)
```

### Pattern 8: Builder Pattern for Complex Objects
Use builders for objects with many optional parameters.

**Good**:
```python
class InferenceRequestBuilder:
    def __init__(self):
        self._query: Optional[str] = None
        self._context: Optional[str] = None
        self._config = RLMConfig()
    
    def with_query(self, query: str) -> 'InferenceRequestBuilder':
        self._query = query
        return self
    
    def with_context(self, context: str) -> 'InferenceRequestBuilder':
        self._context = context
        return self
    
    def with_max_depth(self, depth: int) -> 'InferenceRequestBuilder':
        self._config.max_depth = depth
        return self
    
    def build(self) -> InferenceRequest:
        if not self._query or not self._context:
            raise ValueError("Query and context required")
        return InferenceRequest(
            query=self._query,
            context=self._context,
            config=self._config
        )

# Usage
request = (InferenceRequestBuilder()
    .with_query("Summarize this")
    .with_context(long_text)
    .with_max_depth(15)
    .build())
```

### Pattern 9: Strategy Pattern for Algorithms
Use strategy pattern for swappable algorithms.

**Good**:
```python
from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, max_size: int) -> List[str]:
        pass

class FixedSizeChunking(ChunkingStrategy):
    def chunk(self, text: str, max_size: int) -> List[str]:
        return [text[i:i+max_size] for i in range(0, len(text), max_size)]

class SemanticChunking(ChunkingStrategy):
    def chunk(self, text: str, max_size: int) -> List[str]:
        # Split on paragraph boundaries
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_size:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class ContextManager:
    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: ChunkingStrategy):
        self.strategy = strategy
    
    def chunk_context(self, text: str, max_size: int) -> List[str]:
        return self.strategy.chunk(text, max_size)
```

### Pattern 10: Decorator Pattern for Monitoring
Use decorators to add cross-cutting concerns.

**Good**:
```python
import functools
import time
from typing import TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def monitor_execution(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        logger.info(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"Completed {func.__name__}", duration=duration)
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Failed {func.__name__}", duration=duration, error=str(e))
            raise
    return wrapper

# Usage
@monitor_execution
async def process_recursive_call(query: str, context: str) -> str:
    # Implementation
    pass
```

## Testing Patterns

### Pattern 11: Fixture-Based Testing
Use pytest fixtures for common test setup.

**Good**:
```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=BaseLLM)
    llm.generate.return_value = "mocked response"
    return llm

@pytest.fixture
def sample_context():
    return "This is a sample context for testing" * 100

@pytest.fixture
async def engine(mock_llm):
    executor = MockExecutor()
    context_mgr = ContextManager()
    return RecursiveEngine(mock_llm, executor, context_mgr)

@pytest.mark.asyncio
async def test_simple_query(engine, sample_context):
    result = await engine.process("test query", sample_context)
    assert result is not None
```

### Pattern 12: Parametrized Tests
Test multiple scenarios with parametrize.

**Good**:
```python
@pytest.mark.parametrize("depth,expected", [
    (1, True),
    (10, True),
    (50, True),
    (51, False),  # Exceeds max
])
def test_depth_validation(depth, expected):
    if expected:
        config = RLMConfig(max_depth=depth)
        assert config.max_depth == depth
    else:
        with pytest.raises(ValidationError):
            RLMConfig(max_depth=depth)
```

### Pattern 13: Mock External Services
Always mock external API calls.

**Good**:
```python
@pytest.fixture
def mock_openai_api(monkeypatch):
    async def mock_completion(*args, **kwargs):
        return {
            "choices": [{"message": {"content": "mocked response"}}],
            "usage": {"total_tokens": 100}
        }
    
    monkeypatch.setattr(
        "openai.ChatCompletion.acreate",
        mock_completion
    )

@pytest.mark.asyncio
async def test_with_mocked_api(mock_openai_api):
    llm = OpenAILLM(api_key="fake-key")
    response = await llm.generate("test")
    assert response == "mocked response"
```

## Security Patterns

### Pattern 14: Whitelist, Not Blacklist
For code execution, use whitelists.

**Good**:
```python
ALLOWED_BUILTINS = {
    'len', 'str', 'int', 'float', 'list', 'dict',
    'enumerate', 'range', 'zip', 'min', 'max', 'sum'
}

ALLOWED_MODULES = {
    'math', 're', 'json'
}

def validate_code(code: str) -> bool:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    raise SecurityError(f"Module not allowed: {alias.name}")
        # More validation...
    return True
```

**Bad**:
```python
FORBIDDEN = ['os', 'sys', 'subprocess']  # Easy to bypass!

def validate_code(code: str) -> bool:
    for forbidden in FORBIDDEN:
        if forbidden in code:
            return False
    return True
```

### Pattern 15: Defense in Depth
Layer multiple security checks.

**Good**:
```python
async def execute_code(code: str) -> ExecutionResult:
    # Layer 1: Static analysis
    if not validate_code_syntax(code):
        raise SecurityError("Invalid syntax")
    
    # Layer 2: Security scan
    if not security_scan(code):
        raise SecurityError("Security violation detected")
    
    # Layer 3: Sandboxed execution
    async with sandbox_environment(timeout=5) as sandbox:
        # Layer 4: Resource limits
        sandbox.set_memory_limit(512 * 1024 * 1024)  # 512MB
        sandbox.set_cpu_limit(1.0)  # 1 CPU second
        
        try:
            result = await sandbox.execute(code)
            return result
        except TimeoutError:
            raise ExecutionError("Code execution timeout")
```

## Performance Patterns

### Pattern 16: Caching with TTL
Cache expensive operations.

**Good**:
```python
from functools import lru_cache
import hashlib
import json

class LLMCache:
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
    
    def _hash_request(self, prompt: str, **kwargs) -> str:
        key = json.dumps({"prompt": prompt, **kwargs}, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def get_or_generate(
        self,
        llm: BaseLLM,
        prompt: str,
        **kwargs
    ) -> str:
        cache_key = self._hash_request(prompt, **kwargs)
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired():
                logger.info("Cache hit", key=cache_key)
                return entry.value
        
        logger.info("Cache miss", key=cache_key)
        result = await llm.generate(prompt, **kwargs)
        self._cache[cache_key] = CacheEntry(result, ttl=3600)
        return result
```

### Pattern 17: Parallel Execution
Execute independent operations in parallel.

**Good**:
```python
import asyncio

async def process_chunks_parallel(chunks: List[str]) -> List[str]:
    # Process all chunks concurrently
    tasks = [process_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return results

async def process_chunk(chunk: str) -> str:
    # Process single chunk
    result = await llm.generate(f"Summarize: {chunk}")
    return result
```

**Bad**:
```python
async def process_chunks_sequential(chunks: List[str]) -> List[str]:
    # Sequential - slow!
    results = []
    for chunk in chunks:
        result = await process_chunk(chunk)
        results.append(result)
    return results
```

These patterns will help maintain consistency and quality throughout the RLM project implementation.


