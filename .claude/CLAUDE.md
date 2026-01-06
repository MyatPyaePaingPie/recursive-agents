# Instructions for Claude Code

## Project Context
This is a **research implementation project** that builds a Recursive Language Model (RLM) system based on the 2025 research paper. The goal is to enable LLMs to process unlimited context by writing code that recursively examines and processes input data.

## What You Need to Know

### Research Paper Key Concepts
1. **Fixed Context Window Problem**: Traditional LLMs have hard limits (e.g., 128K tokens)
2. **RLM Solution**: Treat the prompt as an external environment variable that the model accesses via generated code
3. **Recursive Inference**: The model writes code to:
   - Examine the input structure
   - Decompose it into manageable chunks
   - Recursively call itself on those chunks
   - Aggregate results

### Core Components to Implement
- **Recursive Inference Engine**: The orchestrator that manages recursive calls
- **Code Generation Module**: Generates safe Python code to access/process context
- **Execution Sandbox**: Safely runs generated code with restricted permissions
- **Context Manager**: Handles input chunking, state management, and memory
- **Sub-model Handler**: Manages LLM API calls and prompt construction

## Code Style Preferences

### General
- Use **type hints** everywhere (Python 3.10+ syntax)
- Write **docstrings** for all classes and functions (Google style)
- Use **async/await** for all I/O operations (API calls, file operations)
- Prefer **composition over inheritance**
- Use **Pydantic** for data validation and settings management

### Naming Conventions
- Classes: `PascalCase` (e.g., `RecursiveInferenceEngine`)
- Functions/methods: `snake_case` (e.g., `process_recursive_call`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RECURSION_DEPTH`)
- Private methods: `_leading_underscore` (e.g., `_internal_helper`)

### Error Handling
- Use custom exceptions defined in `src/rlm/exceptions.py`
- Always log errors with context
- Fail fast for configuration errors
- Gracefully handle API errors with retries

### Code Organization
```python
# Standard imports
import os
from typing import List, Optional

# Third-party imports
import torch
from pydantic import BaseModel

# Local imports
from rlm.core import RecursiveEngine
from rlm.utils import logger
```

## Architecture Decisions

### Why Async/Await?
RLM systems make multiple LLM API calls that can be parallelized. Async enables:
- Concurrent sub-model invocations
- Non-blocking I/O for better performance
- Easier timeout and cancellation handling

### Why Pydantic?
- Runtime validation of configuration and data models
- Clear schema definitions for prompts and responses
- Easy serialization for logging and debugging

### Why Sandboxed Execution?
Generated code could be malicious or buggy. We use:
- **RestrictedPython** for basic sandboxing
- **Docker containers** (optional) for stronger isolation
- **Resource limits** (CPU, memory, time) for all code execution

### Why Multiple LLM Backends?
Different models have different strengths:
- GPT-4 for complex reasoning
- Claude for long-context baseline comparisons
- Local models (Llama) for privacy/cost-sensitive use cases

## Files to Pay Attention To

- `src/rlm/core/engine.py`: Main recursive inference engine - the heart of the system
- `src/rlm/execution/sandbox.py`: Code execution safety - CRITICAL for security
- `src/rlm/context/manager.py`: Context chunking and state management
- `src/rlm/models/base.py`: Base LLM interface - all models implement this
- `docs/ARCHITECTURE.md`: System design and component interaction
- `context/patterns.md`: Common coding patterns for this project

## Current Focus
**Phase 1**: Build the core recursive inference engine with basic code generation and safe execution.

**Immediate Goals**:
1. Implement `RecursiveInferenceEngine` class
2. Create safe code execution environment
3. Build context chunking system
4. Integrate with at least one LLM API (OpenAI or Anthropic)
5. Write comprehensive tests

## Testing Requirements
- **Unit tests**: For each module in isolation
- **Integration tests**: For component interactions
- **Safety tests**: Ensure malicious code is blocked
- **Performance tests**: Measure recursion overhead vs traditional approaches
- **Coverage target**: 80%+ for core modules

Use pytest with these fixtures:
- `mock_llm`: Mocked LLM responses
- `temp_context`: Temporary test context data
- `sandbox_env`: Isolated execution environment

## Common Tasks

### When implementing a new module:
1. Create the module file with class/function stubs
2. Write the test file with test cases (TDD approach)
3. Implement the actual logic
4. Add docstrings and type hints
5. Run tests and linter
6. Update relevant documentation

### When adding LLM integration:
1. Inherit from `BaseLLM` in `src/rlm/models/base.py`
2. Implement `generate()` and `generate_async()` methods
3. Add configuration to `src/rlm/config.py`
4. Add tests with mocked API responses
5. Add example to `examples/`

### When implementing recursive logic:
1. Always track recursion depth
2. Implement max depth limits
3. Handle base cases explicitly
4. Log each recursive call for debugging
5. Test with various recursion depths

### When working with code execution:
1. ALWAYS use the sandbox
2. Set resource limits (time, memory)
3. Validate generated code before execution
4. Log all executed code for auditing
5. Test with malicious code examples

## Security Considerations
This project generates and executes code - security is CRITICAL:

- **Never execute untrusted code without sandboxing**
- **Always set execution timeouts** (default: 5 seconds)
- **Restrict file system access** in sandbox
- **Restrict network access** in sandbox
- **Log all code execution** for audit trails
- **Validate all user inputs** with Pydantic

## Performance Considerations
- Cache LLM responses when possible
- Use async for parallel sub-model calls
- Implement early stopping for recursion
- Monitor token usage across recursive calls
- Profile code execution overhead

## Documentation Standards
Every module should have:
1. Module-level docstring explaining its purpose
2. Class docstrings with usage examples
3. Method docstrings with parameters and return types
4. Inline comments for complex logic only

Example:
```python
def process_recursive_call(
    context: str,
    depth: int,
    max_depth: int = 10
) -> RecursionResult:
    """Process a single recursive inference step.
    
    Args:
        context: The input context to process
        depth: Current recursion depth (0-indexed)
        max_depth: Maximum allowed recursion depth
        
    Returns:
        RecursionResult containing the processed output
        
    Raises:
        MaxDepthExceededError: If depth > max_depth
        ExecutionError: If code execution fails
        
    Example:
        >>> result = process_recursive_call("long text...", depth=0)
        >>> print(result.output)
    """
    pass
```

## Questions to Ask Me
If you're unsure about:
- **Architecture decisions**: Check `docs/ARCHITECTURE.md` first, then ask
- **Research paper details**: Check `context/research.md` or ask
- **Security concerns**: Always ask - security is critical
- **Performance trade-offs**: Ask for guidance on optimization priorities

## Working Style
- **Proactive**: Anticipate edge cases and handle them
- **Thorough**: Write tests alongside implementation
- **Clear**: Prefer readable code over clever code
- **Safe**: Default to more restrictive/safe approaches
- **Documented**: Keep documentation in sync with code

## Dependencies
Prefer these libraries:
- **LLM APIs**: `openai`, `anthropic`, `transformers`
- **Validation**: `pydantic`
- **Async**: `asyncio`, `aiohttp`
- **Testing**: `pytest`, `pytest-asyncio`, `pytest-mock`
- **Code execution**: `RestrictedPython`
- **Logging**: `loguru` (structured logging)

Avoid:
- `eval()` or `exec()` without sandboxing
- Synchronous blocking calls for I/O
- Global state (use dependency injection)

## Helpful Commands
See `Makefile` for:
- `make test`: Run all tests
- `make lint`: Run linter (ruff)
- `make format`: Format code (black)
- `make type-check`: Run mypy type checker
- `make run-example`: Run basic example
- `make benchmark`: Run performance benchmarks

