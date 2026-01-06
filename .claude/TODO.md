# Development TODO

## Phase 1: Core Infrastructure ⚡
- [ ] Set up Python project structure
  - [ ] Create src/rlm package structure
  - [ ] Set up virtual environment
  - [ ] Create requirements.txt and requirements-dev.txt
  - [ ] Configure Makefile for common tasks
- [ ] Configure development tools
  - [ ] Set up pytest configuration
  - [ ] Configure ruff for linting
  - [ ] Configure black for formatting
  - [ ] Set up mypy for type checking
  - [ ] Configure pre-commit hooks
- [ ] Create basic exceptions module (src/rlm/exceptions.py)
- [ ] Set up logging configuration (src/rlm/logging.py)
- [ ] Create configuration management (src/rlm/config.py)

## Phase 2: LLM Abstractions 🤖
- [ ] Create base LLM interface (src/rlm/models/base.py)
  - [ ] Define BaseLLM abstract class
  - [ ] Define LLMResponse model
  - [ ] Define common methods (generate, generate_async, generate_code)
- [ ] **LOCAL MODEL PRIORITY** ⭐
  - [ ] Implement Ollama integration (src/rlm/models/ollama_llm.py)
    - [ ] Use OpenAI-compatible API
    - [ ] Async generate method
    - [ ] Code generation method
    - [ ] Unit tests with local model
  - [ ] Implement Groq integration (src/rlm/models/groq_llm.py)
    - [ ] Fast cloud alternative (free tier)
    - [ ] Async methods
    - [ ] Unit tests with mocked API
- [ ] OPTIONAL: Cloud providers (if needed)
  - [ ] Implement OpenAI integration (src/rlm/models/openai_llm.py)
  - [ ] Implement Anthropic integration (src/rlm/models/anthropic_llm.py)
- [ ] Create LLM factory (src/rlm/models/factory.py)
  - [ ] Support ollama, groq, openai, anthropic
- [ ] Add response caching (src/rlm/models/cache.py)

## Phase 3: Context Management 📚
- [ ] Design context chunk data model
- [ ] Implement ContextManager (src/rlm/context/manager.py)
  - [ ] In-memory storage
  - [ ] Chunk management
  - [ ] Access tracking
- [ ] Implement chunking strategies (src/rlm/context/chunking.py)
  - [ ] Fixed-size chunking
  - [ ] Semantic chunking (paragraph-based)
  - [ ] Hierarchical chunking
  - [ ] Strategy interface
- [ ] Implement context analysis utilities
  - [ ] Token counting
  - [ ] Structure detection
- [ ] Unit tests for all chunking strategies

## Phase 4: Safe Code Execution 🔒
- [ ] Design code execution sandbox (src/rlm/execution/sandbox.py)
  - [ ] RestrictedPython integration
  - [ ] AST validation
  - [ ] Whitelist allowed operations
  - [ ] Resource limits (timeout, memory)
- [ ] Create context access API for generated code
  - [ ] get_context_length()
  - [ ] get_context_chunk()
  - [ ] call_submodel()
- [ ] Implement execution environment (src/rlm/execution/environment.py)
  - [ ] Isolated namespace
  - [ ] Function whitelisting
  - [ ] Error handling
- [ ] Security validation (src/rlm/execution/validator.py)
  - [ ] AST-based security checks
  - [ ] Forbidden pattern detection
- [ ] Comprehensive security tests
  - [ ] Test malicious code is blocked
  - [ ] Test resource limits work
  - [ ] Test timeout enforcement

## Phase 5: Recursive Inference Engine 🔄
- [ ] Design recursion tree data structure
- [ ] Implement RecursiveInferenceEngine (src/rlm/core/engine.py)
  - [ ] Main process() method
  - [ ] Recursion depth tracking
  - [ ] State management
  - [ ] Error handling with fallbacks
- [ ] Implement prompt templates (src/rlm/core/prompts.py)
  - [ ] System prompt for code generation
  - [ ] User prompt templates
  - [ ] Few-shot examples
- [ ] Create result aggregation strategies (src/rlm/core/aggregation.py)
  - [ ] Concatenation
  - [ ] LLM-based summarization
  - [ ] Voting/consensus
- [ ] Implement recursion strategies (src/rlm/core/strategies.py)
  - [ ] Auto-recursive (LLM decides)
  - [ ] Fixed-depth
  - [ ] Adaptive
- [ ] Integration tests for full recursive flow

## Phase 6: Testing & Quality 🧪
- [ ] Unit tests for all modules (80%+ coverage)
- [ ] Integration tests
  - [ ] Test with small contexts (no recursion)
  - [ ] Test with medium contexts (1-level recursion)
  - [ ] Test with large contexts (multi-level recursion)
- [ ] Security tests
  - [ ] Malicious code blocking
  - [ ] Resource limit enforcement
  - [ ] API key security
- [ ] Performance benchmarks (benchmarks/)
  - [ ] Compare to baseline (truncated context)
  - [ ] Measure token efficiency
  - [ ] Measure latency
  - [ ] Cost analysis
- [ ] Set up CI/CD (GitHub Actions)
  - [ ] Run tests on PR
  - [ ] Run linting
  - [ ] Type checking
  - [ ] Coverage reporting

## Phase 7: Examples & Documentation 📖
- [ ] Create basic example (examples/basic_rlm_demo.py)
  - [ ] Simple recursive processing
  - [ ] With comments explaining each step
- [ ] Create advanced examples
  - [ ] Book summarization (examples/book_summary.py)
  - [ ] Codebase analysis (examples/code_analysis.py)
  - [ ] Multi-document QA (examples/multi_doc_qa.py)
- [ ] Write detailed architecture documentation (docs/ARCHITECTURE.md)
- [ ] Create usage guide (docs/USAGE.md)
- [ ] Write API reference (docs/API.md)
- [ ] Add troubleshooting guide (docs/TROUBLESHOOTING.md)

## Phase 8: Optional Enhancements 🚀
- [ ] Web API (FastAPI)
  - [ ] POST /api/v1/inference endpoint
  - [ ] WebSocket for streaming
  - [ ] API documentation (OpenAPI)
- [ ] Docker support
  - [ ] Dockerfile for sandboxed execution
  - [ ] Docker Compose for full stack
- [ ] Monitoring & observability
  - [ ] Prometheus metrics
  - [ ] Structured logging
  - [ ] Tracing (OpenTelemetry)
- [ ] Advanced features
  - [ ] Streaming responses
  - [ ] Parallel recursion optimization
  - [ ] Learned chunking strategies
  - [ ] Multi-modal support (images, PDFs)

## Blocked / Questions ❓
- [ ] Which LLM should we use for initial testing? (OpenAI GPT-4 recommended)
- [ ] Do we need Docker for MVP or is RestrictedPython sufficient?
- [ ] What's the priority: completeness or speed to first demo?

## Current Focus 🎯
**Start with Phase 1**: Set up the project structure, development tools, and basic configuration.

Once Phase 1 is complete, we can begin implementing the core components in Phases 2-5.

