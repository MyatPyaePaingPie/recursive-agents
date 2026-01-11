# Project State

**Last updated:** 2026-01-11T16:10:00Z
**Updated by:** KERNEL /discover

---

## Project Identity

**Name:** Recursive Language Models (RLM)
**Domain:** AI/ML Research Library
**Tier:** 2 (Production-grade research implementation)
**Version:** 0.1.0

---

## Tech Stack

### Languages
- Python 3.10+ (strict type hints, asyncio throughout)

### Frameworks & Libraries
| Category | Library | Purpose |
|----------|---------|---------|
| Validation | pydantic >= 2.0 | Config & data models |
| Testing | pytest + pytest-asyncio | Async test framework |
| Sandboxing | RestrictedPython >= 6.0 | Code execution safety |
| Logging | loguru >= 0.7 | Structured logging |
| HTTP | aiohttp >= 3.9 | Async HTTP client |
| Tokens | tiktoken >= 0.5 | Token estimation |
| Caching | cachetools >= 5.3 | In-memory caching |

### LLM Integrations (4 providers)
- **Ollama** - Local models (mistral:7b, phi3.5, tinyllama)
- **OpenAI** - GPT-4, GPT-3.5-turbo
- **Anthropic** - Claude models
- **Groq** - Fast cloud inference (free tier)

### Tooling
| Tool | Command | Config |
|------|---------|--------|
| Formatter | `black` | line-length=100 |
| Linter | `ruff` | pyproject.toml |
| Type checker | `mypy` | strict mode |
| Import sorter | `isort` | black profile |
| Pre-commit | `pre-commit` | .pre-commit-config.yaml |

---

## Repo Map

### Entry Points
- **Examples:** `examples/basic_rlm_demo.py` - Main demo script
- **Package:** `src/rlm/__init__.py` - Library exports
- No CLI entry point (library-only usage)

### Core Directories
```
src/rlm/                    # 5,596 lines
├── core/                   # Recursive inference engine (1,728 lines)
│   ├── engine.py           # Main orchestrator (458 lines)
│   ├── models.py           # Data models (149 lines)
│   ├── aggregation.py      # Result aggregation (307 lines)
│   └── transparent.py      # Event-driven engine (784 lines)
├── execution/              # Code sandbox (785 lines)
│   ├── sandbox.py          # RestrictedPython execution (405 lines)
│   └── validator.py        # Security validation (380 lines)
├── context/                # Context management (~256 lines)
│   ├── manager.py          # Storage & coordination
│   ├── chunking.py         # 4 chunking strategies
│   └── models.py           # Chunk data models
├── models/                 # LLM integrations (1,122 lines)
│   ├── base.py             # Abstract BaseLLM (137 lines)
│   ├── factory.py          # Provider auto-detection
│   ├── ollama_llm.py       # Local Ollama
│   ├── openai_llm.py       # OpenAI API
│   ├── anthropic_llm.py    # Anthropic API
│   ├── groq_llm.py         # Groq cloud
│   ├── prompts.py          # Prompt templates
│   └── tiers.py            # Model tiering
├── utils/
│   └── logging.py          # Loguru wrapper (137 lines)
├── config.py               # Pydantic settings (208 lines)
└── exceptions.py           # Exception hierarchy (204 lines)
```

### Test Structure
```
tests/                      # 1,057 lines
├── conftest.py             # Fixtures: mock_llm, sandbox, validator
├── unit/                   # 471 lines
│   ├── test_config.py
│   ├── test_context.py
│   ├── test_exceptions.py
│   └── test_validator.py
├── security/               # 203 lines
│   └── test_sandbox_security.py
└── integration/            # EMPTY - not yet implemented
```

### Documentation
```
docs/                       # 9 guides (~97KB)
├── ARCHITECTURE.md         # System design
├── SPEC.md                 # Technical spec
├── IMPLEMENTATION_GUIDE.md # Build guide
├── QUICKSTART_LOCAL.md     # 5-min Ollama setup
├── LOCAL_MODELS_GUIDE.md   # GPU optimization
└── TESTING.md              # Test docs
```

---

## Tooling Inventory

| Tool | Command | Status |
|------|---------|--------|
| Formatter | `make format` | black + isort |
| Linter | `make lint` | ruff |
| Type check | `make type-check` | mypy strict |
| Tests | `make test` | pytest + coverage |
| Unit tests | `make test-unit` | tests/unit/ |
| Security tests | `make test-security` | tests/security/ |
| All checks | `make check-all` | lint + format + type + test |
| Run demo | `make run-example` | basic_rlm_demo.py |
| Docs | `make docs-serve` | mkdocs |
| Clean | `make clean` | Remove artifacts |

---

## Conventions

### Naming
- **Functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private methods:** `_leading_underscore`
- **Files:** `snake_case.py`

### Error Handling
- Custom exception hierarchy rooted at `RLMException`
- All exceptions include `message` and `details` dict
- Exception types: `ConfigurationError`, `MaxDepthExceededError`, `CodeExecutionError`, `SecurityError`, `LLMError`, `ContextError`, `ValidationError`, `TimeoutError`
- Pattern: Catch specific, re-raise with context

### Logging
- Using `loguru` via `rlm.utils.logging`
- Pattern: `logger = get_logger(__name__)`
- Format: `{time} | {level} | {name}:{function}:{line} | {message}`
- Supports JSON output and log rotation

### Configuration
- Environment variables via `python-dotenv`
- Pydantic Settings for type-safe config
- Config file: `.env` (template: `env.example`)
- Key vars: `OPENAI_API_KEY`, `GROQ_API_KEY`, `RLM_LLM_PROVIDER`, `RLM_LLM_MODEL`

### Async Patterns
- All I/O is async (`async def`, `await`)
- LLM calls: `await llm.generate(prompt)`
- Sandbox: `await sandbox.execute(code, context_api)`
- Tests: `@pytest.mark.asyncio` or `asyncio_mode = "auto"`

### Test Fixtures (conftest.py)
- `mock_llm()` - MockLLM with configurable responses
- `config()` - Test RLMConfig
- `context_manager()` - ContextManager with SemanticChunking
- `sandbox()` - SandboxEnvironment (5s timeout)
- `validator()` - CodeValidator
- `sample_context()` - Large document fixture
- `sample_code_context()` - Code sample fixture

### Test Markers
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.security` - Security-focused
- `@pytest.mark.requires_api` - Needs API keys

---

## Do Not Touch (Critical Paths)

### Security-Critical
- `src/rlm/execution/sandbox.py` - **CRITICAL** - All code execution goes through here
  - `SAFE_BUILTINS` whitelist - Adding items = security risk
  - `SandboxEnvironment.execute()` - Main entry point
  - RestrictedPython guards configuration
- `src/rlm/execution/validator.py` - Code validation before execution
  - `SecurityAuditor` - Audit logging
  - Dangerous pattern detection

### Core Logic
- `src/rlm/core/engine.py` - Main recursive inference orchestrator
- `src/rlm/models/base.py` - Abstract LLM interface (all providers inherit)
- `src/rlm/exceptions.py` - Exception hierarchy (breaking changes affect all)

### Configuration
- `src/rlm/config.py` - Pydantic settings (breaking = config migration needed)
- `pyproject.toml` - Build, test, lint configuration
- `.env` files - Contains API keys (never commit)

---

## Known Issues

### Not Yet Implemented
- `tests/integration/` - Integration tests directory is empty
- `benchmarks/` - Performance benchmarks not implemented
- `web_dashboard/` - Partial implementation only

### Technical Debt
- `examples/transparent_demo.py` - 23K lines (likely autogenerated, needs cleanup)
- RestrictedPython fallback warning when not installed
- No CLI entry point (library-only)

---

## Dependencies Graph

```
                    ┌─────────────────┐
                    │   exceptions    │ (no deps)
                    │   utils/logging │ (no deps)
                    │   config        │ (no deps)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │ models/  │   │ context/ │   │execution/│
       │ base.py  │   │ manager  │   │ sandbox  │
       └────┬─────┘   └────┬─────┘   └────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    ┌─────────────────┐
                    │  core/engine    │ (orchestrator)
                    └─────────────────┘
```

Safe to modify independently:
- `utils/` - No internal dependencies
- `exceptions.py` - No internal dependencies
- `models/[provider]_llm.py` - Independent implementations

---

## Recent Changes

- 2026-01-11: KERNEL initialized
- 2026-01-11: Full discovery completed

---

## Quick Commands

```bash
# Development
make install-dev          # Install all deps
make test                 # Run tests with coverage
make check-all           # Full CI check

# Running
make run-example         # Basic demo
python examples/demo_recursive_processing.py

# Code quality
make lint-fix            # Auto-fix lint issues
make format              # Format code
make type-check          # Type validation
```

---

## Notes

This project implements the "Recursive Language Models" research paper (2025). Key insight: LLMs can overcome fixed context limits by writing code that recursively processes arbitrarily long inputs.

**Security is paramount** - all code execution is sandboxed with:
- RestrictedPython compile-time restrictions
- SAFE_BUILTINS whitelist (no file/network access)
- Execution timeouts (default 5s)
- Memory limits
- Full audit logging
