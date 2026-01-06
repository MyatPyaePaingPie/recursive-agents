# Architecture Decision Records (ADR)

## ADR-001: Use Python as Primary Language

**Status**: Accepted

**Context**: Need to choose primary implementation language for RLM system.

**Decision**: Use Python 3.10+

**Rationale**:
- Excellent LLM API libraries (OpenAI, Anthropic, HuggingFace)
- Rich ecosystem for data processing
- RestrictedPython available for sandboxing
- Fast prototyping for research implementation
- Strong async support with asyncio

**Consequences**:
- Good: Rapid development, many libraries
- Bad: Performance overhead vs compiled languages
- Mitigation: Use async for I/O, profile critical paths

## ADR-002: Async-First Architecture

**Status**: Accepted

**Context**: System makes many external API calls to LLMs.

**Decision**: Use async/await throughout, all I/O operations async

**Rationale**:
- LLM API calls are I/O bound (100-1000ms each)
- Recursive calls can be parallelized
- Better resource utilization
- Modern Python idiom

**Consequences**:
- Good: Better performance, can parallelize
- Bad: More complex than synchronous code
- Mitigation: Provide sync wrappers where needed

## ADR-003: RestrictedPython for Code Sandboxing

**Status**: Accepted

**Context**: Need to execute untrusted generated code safely.

**Decision**: Use RestrictedPython as primary sandboxing mechanism, with Docker as optional stronger isolation.

**Rationale**:
- RestrictedPython: Lightweight, in-process
- Validates code before execution
- Restricts dangerous operations
- Docker: Heavier but stronger isolation

**Consequences**:
- Good: Multiple layers of security
- Bad: RestrictedPython can be bypassed by clever exploits
- Mitigation: Regular security audits, keep RestrictedPython updated

**Alternatives Considered**:
- PyPy sandbox: Outdated, unmaintained
- Web Assembly: Too restrictive for LLM code
- Full VM per execution: Too slow/expensive

## ADR-004: Pydantic for Data Validation

**Status**: Accepted

**Context**: Need runtime validation of configurations and data models.

**Decision**: Use Pydantic for all data models

**Rationale**:
- Runtime type checking
- Automatic JSON serialization
- Clear error messages
- Good IDE support via type hints
- Fast (implemented in Rust)

**Consequences**:
- Good: Catch errors early, great DX
- Bad: Adds dependency
- Mitigation: Pydantic is well-maintained and stable

## ADR-005: Support Multiple LLM Providers

**Status**: Accepted

**Context**: Different use cases need different LLMs.

**Decision**: Abstract LLM interface, support OpenAI, Anthropic, and local models.

**Rationale**:
- OpenAI: Best for complex reasoning
- Anthropic: Long context for baseline comparison
- Local: Privacy, cost, experimentation
- Users have different preferences/requirements

**Consequences**:
- Good: Flexibility, not locked to one provider
- Bad: More code to maintain
- Mitigation: Share common interface and test suite

## ADR-006: Semantic Chunking by Default

**Status**: Accepted

**Context**: Need to chunk large contexts for processing.

**Decision**: Default to semantic chunking (split on paragraph/section boundaries) over fixed-size chunking.

**Rationale**:
- Preserves meaning across chunks
- Better for LLM understanding
- More natural for hierarchical documents
- Research shows better quality

**Consequences**:
- Good: Better results
- Bad: Slightly more complex than fixed-size
- Mitigation: Keep fixed-size as fallback option

## ADR-007: Use Loguru for Logging

**Status**: Accepted

**Context**: Need structured logging for debugging and monitoring.

**Decision**: Use Loguru instead of standard logging module.

**Rationale**:
- Better API, easier to use
- Structured logging out of the box
- Async-safe
- Better formatting
- Exception tracing

**Consequences**:
- Good: Better logging experience
- Bad: Non-standard library
- Mitigation: Loguru is stable and popular

## ADR-008: Explicit Recursion Depth Limits

**Status**: Accepted

**Context**: Recursive calls could theoretically be infinite.

**Decision**: Enforce strict max recursion depth (default 10, configurable up to 50).

**Rationale**:
- Prevents runaway costs
- Protects against infinite loops
- Forces more efficient solutions
- Practical limit on complexity

**Consequences**:
- Good: Safety, cost control
- Bad: May limit some valid use cases
- Mitigation: Make configurable, provide guidance

## ADR-009: Cache LLM Responses

**Status**: Accepted

**Context**: Identical prompts may be sent multiple times.

**Decision**: Implement content-hash-based caching with TTL.

**Rationale**:
- LLM calls are expensive (cost and time)
- Identical inputs → identical outputs (with temp=0)
- Significant cost savings potential
- Faster responses

**Consequences**:
- Good: Performance, cost savings
- Bad: Memory usage, stale cache risk
- Mitigation: Configurable TTL, cache eviction policy

**Implementation**:
```python
cache_key = hash(prompt + str(temperature) + str(model))
```

## ADR-010: Fail-Safe Fallback Strategies

**Status**: Accepted

**Context**: Code generation or execution can fail.

**Decision**: Implement fallback strategies when recursion fails.

**Rationale**:
- LLMs don't always generate valid code
- Execution can fail for various reasons
- Better to degrade gracefully than fail completely

**Fallback Hierarchy**:
1. Try generated code execution
2. If fails, try simpler chunking strategy
3. If fails, truncate context to fit
4. If fails, return error with partial results

**Consequences**:
- Good: Robustness, better UX
- Bad: More complex error handling
- Mitigation: Clear logging of fallback usage

## ADR-011: Separate Code Generation from Reasoning

**Status**: Accepted

**Context**: Different models have different strengths.

**Decision**: Allow using different models for code generation vs recursive reasoning.

**Rationale**:
- Cheaper models (GPT-3.5) can generate code
- More expensive models (GPT-4) for reasoning
- Optimize cost vs quality
- Experiment with specialized models

**Consequences**:
- Good: Cost optimization flexibility
- Bad: More configuration complexity
- Mitigation: Sensible defaults, clear documentation

## ADR-012: In-Memory Context Storage Initially

**Status**: Accepted

**Context**: Need to store large contexts efficiently.

**Decision**: Use in-memory storage for MVP, design for pluggable backends.

**Rationale**:
- Simple for initial implementation
- Fast access
- Good for prototype/research
- Can add persistence later

**Future Options**:
- Redis for distributed systems
- S3/Object storage for very large contexts
- Vector databases for semantic search

**Consequences**:
- Good: Simple, fast
- Bad: Limited by RAM
- Mitigation: Document limits, design for future backends

## ADR-013: Pytest for Testing

**Status**: Accepted

**Context**: Need testing framework.

**Decision**: Use pytest with pytest-asyncio for testing.

**Rationale**:
- De facto standard for Python
- Excellent async support
- Rich plugin ecosystem
- Clear assertions
- Good fixtures system

**Consequences**:
- Good: Industry standard, well-supported
- Bad: None significant
- Mitigation: N/A

## ADR-014: Type Checking with MyPy

**Status**: Accepted

**Context**: Python is dynamically typed but we use type hints.

**Decision**: Enforce type checking with MyPy in CI.

**Rationale**:
- Catch type errors before runtime
- Better IDE support
- Documentation through types
- Matches Pydantic usage

**Consequences**:
- Good: Fewer bugs, better docs
- Bad: Slower development initially
- Mitigation: Gradual adoption, use `# type: ignore` sparingly

## ADR-015: Monorepo Structure

**Status**: Accepted

**Context**: Project has multiple components that evolve together.

**Decision**: Keep everything in single repository.

**Rationale**:
- Components tightly coupled
- Easier to ensure compatibility
- Simpler for research/experimentation
- Easier for users to get started

**Consequences**:
- Good: Simplicity, atomic changes
- Bad: Harder to version components independently
- Mitigation: Use semantic versioning, clear changelogs

## Questions / Open Decisions

### Q1: Should we support streaming responses?
**Status**: Under consideration

**Pros**: Better UX for long operations
**Cons**: More complex implementation
**Decision**: Defer to v2

### Q2: Should we implement learning/optimization?
**Status**: Under consideration

**Context**: System could learn better recursion strategies over time
**Decision**: Defer to future research phase

### Q3: Multi-modal support?
**Status**: Under consideration

**Context**: Process images, PDFs, etc.
**Decision**: Focus on text first, design for extensibility


