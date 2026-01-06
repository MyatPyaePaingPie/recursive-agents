# Architecture Documentation

## System Overview

The Recursive Language Models (RLM) system enables LLMs to process arbitrarily long inputs by generating and executing code that recursively examines context. This document describes the system architecture and component interactions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                           │
│                  (Query + Large Context)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Recursive Inference Engine                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Orchestrate recursive processing                   │  │
│  │  • Manage recursion depth and state                  │  │
│  │  • Coordinate components                             │  │
│  └──────────────────────────────────────────────────────┘  │
└───┬────────────────┬───────────────────┬───────────────┬───┘
    │                │                   │               │
    ▼                ▼                   ▼               ▼
┌────────┐    ┌──────────┐    ┌──────────────┐   ┌──────────┐
│Context │    │   LLM    │    │    Code      │   │ Result   │
│Manager │    │  Models  │    │  Executor    │   │Aggregator│
└────────┘    └──────────┘    └──────────────┘   └──────────┘
    │              │                   │               │
    │         ┌────┴────┐              │               │
    │         │         │              │               │
    ▼         ▼         ▼              ▼               ▼
┌────────┐ ┌────┐  ┌────────┐  ┌──────────┐   ┌──────────┐
│Chunking│ │OpenAI│ │Anthropic│ │ Sandbox  │   │Strategies│
└────────┘ └────┘  └────────┘  └──────────┘   └──────────┘
```

## Component Details

### 1. Recursive Inference Engine
**Location**: `src/rlm/core/engine.py`

The central orchestrator that manages the recursive processing workflow.

**Responsibilities**:
- Accept user queries and large contexts
- Manage recursion depth tracking
- Coordinate between components
- Handle errors and implement fallback strategies
- Return aggregated results

**Key Methods**:
```python
class RecursiveInferenceEngine:
    async def process(
        self,
        query: str,
        context: str,
        max_depth: int = 10
    ) -> InferenceResult:
        """Main entry point for recursive processing"""

    async def _recursive_step(
        self,
        query: str,
        context: str,
        depth: int
    ) -> RecursionNode:
        """Execute one step of recursion"""

    async def _aggregate_results(
        self,
        nodes: List[RecursionNode]
    ) -> str:
        """Aggregate results from recursive calls"""
```

**Workflow**:
1. Initialize context manager with input
2. Generate code to examine context
3. Execute code safely
4. If code requests recursion:
   - Create sub-problems
   - Recursively process each
   - Aggregate results
5. Return final result

### 2. Context Manager
**Location**: `src/rlm/context/manager.py`

Manages large context storage, chunking, and access.

**Responsibilities**:
- Store arbitrarily large contexts efficiently
- Provide chunk-based access
- Track context access patterns
- Implement various chunking strategies

**Data Model**:
```python
class ContextManager:
    chunks: List[ContextChunk]
    chunk_map: Dict[int, ContextChunk]
    access_log: List[ChunkAccess]
    strategy: ChunkingStrategy
```

**Chunking Strategies**:
- **Fixed-size**: Simple token-based splitting
- **Semantic**: Split on paragraph/section boundaries
- **Hierarchical**: Tree-based for nested documents
- **Adaptive**: Adjust based on recursion depth

### 3. LLM Models
**Location**: `src/rlm/models/`

Abstraction layer for different LLM providers.

**Interface**:
```python
class BaseLLM(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate text completion"""

    @abstractmethod
    async def generate_code(
        self,
        task_description: str
    ) -> str:
        """Generate Python code for task"""
```

**Implementations**:
- `OpenAILLM`: GPT-4, GPT-3.5
- `AnthropicLLM`: Claude 3 (Opus, Sonnet, Haiku)
- `LocalLLM`: HuggingFace models (Llama, Mistral)

**Caching**:
- Content-hash-based caching
- TTL expiration
- Size limits
- Cache invalidation on version changes

### 4. Code Executor
**Location**: `src/rlm/execution/`

Safely executes LLM-generated code.

**Security Layers**:
1. **AST Validation**: Parse and validate syntax
2. **Security Scanning**: Check for forbidden operations
3. **Sandboxed Execution**: RestrictedPython
4. **Resource Limits**: Timeout, memory, CPU

**Available Functions** (for generated code):
```python
def get_context_length() -> int:
    """Get total context length in tokens"""

def get_context_chunk(start: int, end: int) -> str:
    """Get specific chunk of context"""

def call_submodel(chunk: str, query: str) -> str:
    """Make recursive LLM call on chunk"""

def aggregate_results(results: List[str]) -> str:
    """Aggregate multiple results"""
```

**Execution Environment**:
```python
class SandboxEnvironment:
    timeout: int = 5  # seconds
    memory_limit: int = 512 * 1024 * 1024  # 512MB
    allowed_builtins: Set[str] = {'len', 'str', 'int', ...}
    allowed_modules: Set[str] = {'math', 're', 'json'}
```

### 5. Result Aggregator
**Location**: `src/rlm/core/aggregation.py`

Combines results from recursive calls.

**Strategies**:
- **Concatenate**: Simple string concatenation
- **Summarize**: LLM-based summarization of results
- **Vote**: Majority voting for classification tasks
- **Weighted**: Confidence-weighted aggregation

## Data Flow

### Typical Request Flow

```
1. User submits query + large context
   │
   ▼
2. Engine initializes ContextManager
   │
   ▼
3. Engine prompts LLM to generate code
   │
   ▼
4. LLM returns Python code
   │
   ▼
5. Validator checks code safety
   │
   ▼
6. Sandbox executes code
   │
   ├─ Code examines context structure
   │  (via get_context_length, etc.)
   │
   ├─ Code decides recursion needed
   │
   └─ Code calls call_submodel() multiple times
      │
      ▼
7. Engine creates recursive sub-problems
   │
   ▼
8. For each sub-problem:
   │  ├─ Repeat steps 3-7 (recursive)
   │  └─ Depth tracked, max enforced
   │
   ▼
9. Aggregate results from all sub-calls
   │
   ▼
10. Return final result to user
```

### Example Code Generation

**User Query**: "Summarize this book"
**Context**: 500-page book (~200K tokens)

**Generated Code**:
```python
# LLM generates this code
def process_book():
    total_length = get_context_length()
    
    # Check if recursion needed
    if total_length > 4000:  # Larger than comfortable window
        # Split into chapters (assume 10 chapters)
        chapter_size = total_length // 10
        summaries = []
        
        for i in range(10):
            start = i * chapter_size
            end = (i + 1) * chapter_size
            chapter = get_context_chunk(start, end)
            
            # Recursive call
            summary = call_submodel(
                chunk=chapter,
                query="Summarize this chapter concisely"
            )
            summaries.append(f"Chapter {i+1}: {summary}")
        
        # Final aggregation
        all_summaries = "\n".join(summaries)
        final = call_submodel(
            chunk=all_summaries,
            query="Create a cohesive summary from these chapter summaries"
        )
        return final
    else:
        # Direct processing
        return "This book discusses..."  # LLM processes directly
```

## Error Handling

### Error Types & Strategies

1. **Configuration Errors**
   - Validation: Pydantic models
   - Strategy: Fail fast with clear message

2. **API Errors** (rate limits, timeouts)
   - Detection: HTTP status codes
   - Strategy: Exponential backoff retry

3. **Code Execution Errors**
   - Detection: Exception in sandbox
   - Strategy: Log and use fallback

4. **Recursion Errors** (max depth)
   - Detection: Depth counter
   - Strategy: Return partial result with warning

5. **Security Errors** (malicious code)
   - Detection: AST validation
   - Strategy: Block and alert

### Fallback Hierarchy

```
Try: Generated code execution
  ├─ Success → Return result
  │
  └─ Failure
     │
     └─ Try: Simpler chunking strategy (no code gen)
        ├─ Success → Return result
        │
        └─ Failure
           │
           └─ Try: Truncate context to fit
              ├─ Success → Return result with warning
              │
              └─ Failure → Return error + partial results
```

## Performance Considerations

### Optimization Strategies

1. **Caching**
   - Cache LLM responses by content hash
   - Cache context chunks
   - TTL-based expiration

2. **Parallelization**
   - Execute independent recursive calls concurrently
   - Use asyncio.gather for parallel LLM calls

3. **Early Stopping**
   - Stop recursion when answer found
   - Skip irrelevant context sections

4. **Model Selection**
   - Use cheaper models (GPT-3.5) for code generation
   - Use expensive models (GPT-4) only for reasoning

5. **Token Optimization**
   - Compress prompts
   - Use efficient chunking
   - Remove redundant context

### Expected Performance

- **Simple queries** (no recursion): 2-5 seconds
- **Single-level recursion**: 10-20 seconds
- **Multi-level recursion**: 30-60 seconds

*Times assume OpenAI GPT-4 with typical API latency*

## Security Architecture

### Defense in Depth

```
Layer 1: Input Validation
  └─ Pydantic models, size limits

Layer 2: Code Analysis
  └─ AST parsing, forbidden pattern detection

Layer 3: Sandboxing
  └─ RestrictedPython, whitelist operations

Layer 4: Resource Limits
  └─ Timeout, memory, CPU limits

Layer 5: Monitoring
  └─ Log all executions, alert on suspicious patterns
```

### Threat Model

**Threats**:
- Malicious code in LLM output
- Resource exhaustion (infinite loops)
- Data exfiltration attempts
- Privilege escalation

**Mitigations**:
- See Defense in Depth above
- Regular security audits
- Update RestrictedPython regularly
- Consider Docker for production

## Scalability

### Current Limitations
- Single machine, in-memory storage
- Sequential recursion in some cases
- LLM API rate limits

### Future Enhancements
1. **Distributed Processing**
   - Redis for shared context storage
   - Worker pool for parallel execution

2. **Persistent Storage**
   - S3/Object storage for large contexts
   - PostgreSQL for metadata

3. **Advanced Caching**
   - Shared cache across instances
   - Semantic similarity for cache hits

4. **Load Balancing**
   - Multiple LLM API keys
   - Fallback to different providers

## Testing Strategy

### Test Pyramid

```
        /\
       /  \       E2E Tests
      /────\      (Full workflows)
     /      \
    /────────\    Integration Tests
   /          \   (Component interactions)
  /────────────\  
 /              \  Unit Tests
/────────────────\ (Individual functions)
```

### Test Categories

1. **Unit Tests** (80% coverage target)
   - Each component in isolation
   - Mocked dependencies

2. **Integration Tests**
   - Component interactions
   - Real LLM calls (with mocks in CI)

3. **Security Tests**
   - Malicious code blocking
   - Resource limit enforcement

4. **Performance Tests**
   - Benchmarks vs baseline
   - Token efficiency
   - Latency measurements

## Monitoring & Observability

### Metrics to Track

1. **Business Metrics**
   - Requests per day
   - Success rate
   - Average recursion depth

2. **Performance Metrics**
   - Latency (p50, p95, p99)
   - Token usage per query
   - Cache hit rate

3. **Cost Metrics**
   - API costs per query
   - Cost by model
   - Cost by recursion depth

4. **Security Metrics**
   - Blocked code executions
   - Timeout rate
   - Security alerts

### Logging Structure

```python
{
    "timestamp": "2025-01-05T10:30:00Z",
    "level": "INFO",
    "request_id": "abc123",
    "component": "engine",
    "event": "recursion_step",
    "depth": 2,
    "tokens_used": 1500,
    "execution_time_ms": 234
}
```

## Deployment Architecture

### Development
- Local machine
- In-memory storage
- RestrictedPython sandbox

### Production (Future)
- Container orchestration (K8s)
- Redis for caching
- Docker for sandboxing
- Load balancer
- Monitoring stack (Prometheus + Grafana)

```
                  ┌─────────────┐
                  │Load Balancer│
                  └──────┬──────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼───┐       ┌───▼───┐       ┌───▼───┐
    │RLM Pod│       │RLM Pod│       │RLM Pod│
    └───┬───┘       └───┬───┘       └───┬───┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                ┌───────┴────────┐
                │                │
           ┌────▼────┐      ┌───▼────┐
           │  Redis  │      │Postgres│
           │(Caching)│      │(Metrics)│
           └─────────┘      └────────┘
```

## Next Steps

1. Implement core components (Phases 2-5 in TODO.md)
2. Write comprehensive tests
3. Create example implementations
4. Benchmark against baselines
5. Gather feedback and iterate


