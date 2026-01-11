# Known Issues and Gotchas

## Code Execution Security

### ⚠️ CRITICAL: RestrictedPython Limitations
**Issue**: RestrictedPython can be bypassed by determined attackers.

**Example**:
```python
# This might be blocked:
import os

# But this might work:
__import__('os').system('malicious_command')

# Or this:
getattr(__builtins__, 'exec')('malicious_code')
```

**Mitigation**:
- Always validate code with AST inspection first
- Remove or restrict `__builtins__`
- Set aggressive timeouts
- Consider Docker for production
- Never trust generated code

**Testing**:
```python
def test_bypass_attempts():
    malicious_codes = [
        "__import__('os').system('ls')",
        "getattr(__builtins__, 'eval')('1+1')",
        "[c for c in ().__class__.__bases__[0].__subclasses__()]",
    ]
    for code in malicious_codes:
        with pytest.raises(SecurityError):
            execute_code(code)
```

## Async/Await Pitfalls

### Forgetting `await`
**Issue**: Easy to forget `await` and get a coroutine object instead of result.

**Bad**:
```python
result = llm.generate("prompt")  # result is coroutine, not string!
print(result)  # Prints: <coroutine object ...>
```

**Good**:
```python
result = await llm.generate("prompt")  # Now result is string
print(result)  # Prints actual result
```

**Detection**: Enable MyPy with strict async checking:
```python
# mypy.ini
[mypy]
warn_unused_coroutines = True
```

### Mixing Sync and Async
**Issue**: Can't call async functions from sync code without event loop.

**Bad**:
```python
def sync_function():
    result = await async_function()  # SyntaxError!
```

**Good**:
```python
import asyncio

def sync_function():
    result = asyncio.run(async_function())  # Creates event loop
```

**But Better**: Keep everything async or provide sync wrappers explicitly.

## LLM API Rate Limits

### OpenAI Rate Limits
**Issue**: OpenAI enforces RPM (requests per minute) and TPM (tokens per minute) limits.

**Symptoms**:
- 429 status code
- "Rate limit exceeded" errors
- Intermittent failures during high recursion

**Mitigation**:
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def call_with_retry(llm, prompt):
    return await llm.generate(prompt)
```

**Configuration**:
```python
class OpenAILLM:
    def __init__(self, rpm_limit: int = 60):
        self.rate_limiter = AsyncLimiter(rpm_limit, 60)  # 60 per minute
    
    async def generate(self, prompt: str):
        async with self.rate_limiter:
            return await self._generate(prompt)
```

## Context Window Token Counting

### Token Count Mismatch
**Issue**: Simple `len(text)` doesn't equal token count.

**Problem**:
```python
text = "Hello world"
print(len(text))  # 11 characters
# But OpenAI tokenizer: ~2 tokens
```

**Solution**: Use proper tokenizer
```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

**Gotcha**: Different models use different tokenizers!
- GPT-4: cl100k_base
- GPT-3.5: cl100k_base  
- Claude: Different (use Anthropic's method)

## Recursion Depth Issues

### Stack Overflow in Python
**Issue**: Python has default recursion limit of 1000.

**Problem**:
```python
import sys
print(sys.getrecursionlimit())  # Usually 1000

def deep_recursion(n):
    if n == 0:
        return
    deep_recursion(n - 1)

deep_recursion(2000)  # RecursionError!
```

**Solution**: Don't use Python recursion for RLM recursion!
```python
# Bad: Python recursion
def rlm_recursive(context, depth):
    if depth > 10:
        return
    result = process(context)
    return rlm_recursive(result, depth + 1)

# Good: Iteration with explicit stack
def rlm_iterative(context, max_depth):
    stack = [(context, 0)]
    results = []
    
    while stack:
        ctx, depth = stack.pop()
        if depth >= max_depth:
            continue
        result = process(ctx)
        if needs_recursion(result):
            stack.append((result, depth + 1))
        else:
            results.append(result)
    
    return aggregate(results)
```

## Pydantic Validation Gotchas

### Mutable Default Arguments
**Issue**: Mutable defaults are shared across instances.

**Bad**:
```python
from pydantic import BaseModel

class Config(BaseModel):
    options: list = []  # WRONG! Shared across instances

c1 = Config()
c1.options.append('a')
c2 = Config()
print(c2.options)  # ['a'] - unexpected!
```

**Good**:
```python
from pydantic import BaseModel, Field

class Config(BaseModel):
    options: list = Field(default_factory=list)  # New list per instance

c1 = Config()
c1.options.append('a')
c2 = Config()
print(c2.options)  # [] - correct!
```

### Validation Order
**Issue**: Validators run in order, dependencies matter.

```python
class RecursionConfig(BaseModel):
    max_depth: int = 10
    timeout_per_level: int = 5
    total_timeout: int = 50
    
    @validator('total_timeout')
    def validate_total_timeout(cls, v, values):
        # values only has fields defined BEFORE this one
        max_depth = values.get('max_depth')
        timeout_per_level = values.get('timeout_per_level')
        if max_depth and timeout_per_level:
            min_timeout = max_depth * timeout_per_level
            if v < min_timeout:
                raise ValueError(f"Total timeout too small: need >= {min_timeout}")
        return v
```

## Caching Issues

### Cache Invalidation
**Issue**: Cached results may become stale if LLM behavior changes.

**Problem**:
- Model updated by provider
- Temperature > 0 gives different results
- Prompt engineering improvements not reflected

**Mitigation**:
```python
def cache_key(prompt: str, model: str, temperature: float, version: str) -> str:
    # Include version to invalidate on updates
    return hashlib.sha256(
        f"{prompt}|{model}|{temperature}|{version}".encode()
    ).hexdigest()

# Bump version to invalidate all caches
CACHE_VERSION = "v2"
```

### Memory Leaks
**Issue**: Unbounded cache grows forever.

**Solution**: Use TTL and size limits
```python
from cachetools import TTLCache

class LLMCache:
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
```

## Testing Gotchas

### Async Test Fixtures
**Issue**: Need special handling for async fixtures.

**Wrong**:
```python
@pytest.fixture
def engine():  # Not async!
    return RecursiveEngine()

@pytest.mark.asyncio
async def test_something(engine):
    await engine.process("test")  # May fail!
```

**Right**:
```python
@pytest.fixture
async def engine():  # Async fixture
    eng = RecursiveEngine()
    await eng.initialize()
    yield eng
    await eng.cleanup()

@pytest.mark.asyncio
async def test_something(engine):
    await engine.process("test")  # Works!
```

### Mocking Async Functions
**Issue**: Regular Mock doesn't work with async.

**Wrong**:
```python
from unittest.mock import Mock

mock_llm = Mock()
mock_llm.generate.return_value = "result"

await mock_llm.generate("test")  # RuntimeWarning: coroutine never awaited
```

**Right**:
```python
from unittest.mock import AsyncMock

mock_llm = AsyncMock()
mock_llm.generate.return_value = "result"

result = await mock_llm.generate("test")  # Works!
```

## Windows-Specific Issues

### Path Separators
**Issue**: Windows uses backslashes, Unix uses forward slashes.

**Solution**: Always use `pathlib`
```python
from pathlib import Path

# Bad
config_path = "config/settings.json"  # Breaks on Windows

# Good
config_path = Path("config") / "settings.json"  # Works everywhere
```

### Line Endings
**Issue**: Windows uses CRLF (`\r\n`), Unix uses LF (`\n`).

**Solution**: Normalize when reading
```python
def read_file(path: Path) -> str:
    return path.read_text().replace('\r\n', '\n')
```

### Subprocess on Windows
**Issue**: Different shell syntax and executables.

**Solution**: Use `shell=True` cautiously or avoid shell
```python
# Cross-platform
subprocess.run([sys.executable, "-m", "pytest"], check=True)

# Not cross-platform
subprocess.run("pytest", shell=True, check=True)  # May fail on Windows
```

## Performance Gotchas

### Blocking I/O in Async Code
**Issue**: Blocking calls block entire event loop.

**Bad**:
```python
async def process():
    data = open('file.txt').read()  # Blocks entire event loop!
    return data
```

**Good**:
```python
import aiofiles

async def process():
    async with aiofiles.open('file.txt', 'r') as f:
        data = await f.read()  # Non-blocking
    return data
```

### Too Many Concurrent Tasks
**Issue**: Creating 10,000 tasks at once can exhaust resources.

**Solution**: Use semaphore to limit concurrency
```python
import asyncio

async def process_with_limit(items: List[str]):
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
    
    async def process_one(item):
        async with semaphore:
            return await expensive_operation(item)
    
    tasks = [process_one(item) for item in items]
    return await asyncio.gather(*tasks)
```

## Documentation Gotchas

### Outdated Examples
**Issue**: Code examples in docstrings get outdated.

**Solution**: Test docstring examples with doctest
```python
def process(text: str) -> str:
    """Process text.
    
    Example:
        >>> process("hello")
        'HELLO'
    """
    return text.upper()

# Run with: python -m doctest myfile.py
```

## Environment Variables

### Missing API Keys
**Issue**: Forgetting to set API keys causes cryptic errors.

**Solution**: Validate early with clear errors
```python
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    @validator('openai_api_key', 'anthropic_api_key')
    def check_at_least_one_key(cls, v, values):
        if not any(values.values()) and not v:
            raise ValueError(
                "At least one API key must be set. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
            )
        return v
    
    class Config:
        env_file = '.env'
```

## Logging Gotchas

### Logging in Async Context
**Issue**: Multiple tasks logging simultaneously can interleave.

**Solution**: Use structured logging with context
```python
from loguru import logger
import contextvars

request_id = contextvars.ContextVar('request_id', default='unknown')

logger.add(
    "logs/app.log",
    format="{time} | {level} | {extra[request_id]} | {message}"
)

async def process_request(req_id: str):
    request_id.set(req_id)
    logger.bind(request_id=req_id).info("Processing started")
```

Remember: When in doubt, check this file first!


