# Project Scaffolding Summary

## What Has Been Created

This repository now contains a complete scaffolding for implementing and testing the **Recursive Language Models (RLM)** research paper from 2025. The project structure follows best practices for working with Claude Code and provides everything you need to get started.

## Research Paper Overview

The RLM paper (https://www.tekta.ai/ai-research-papers/recursive-language-models-2025) introduces a novel approach to overcome LLM context window limitations:

**Core Innovation**: Instead of trying to fit large inputs into fixed context windows, treat the prompt as an external data structure that the LLM can programmatically access by writing code.

**How It Works**:
1. LLM generates Python code to examine and process large contexts
2. Code executes safely in a sandbox with access to context via API functions
3. Code can recursively call the LLM on smaller chunks
4. Results are aggregated to produce the final answer

**Benefits**:
- Process arbitrarily long inputs (unlimited context)
- More efficient token usage (only process relevant parts)
- Structured, debuggable reasoning
- Works with any LLM

## What's Been Set Up

### 📁 Core Documentation Files

1. **README.md** - Project overview, setup instructions, architecture
2. **CLAUDE.md** - Detailed instructions specifically for Claude Code
3. **SPEC.md** - Complete technical specification
4. **TODO.md** - Phased development roadmap
5. **IMPLEMENTATION_GUIDE.md** - Step-by-step implementation instructions

### 📁 Context Files (for Claude Code)

Located in `context/`:
- **research.md** - Research paper summary and key concepts
- **patterns.md** - Code patterns and best practices for this project
- **decisions.md** - Architecture Decision Records (ADRs)
- **gotchas.md** - Known issues and common pitfalls

### 📁 Configuration Files

1. **pyproject.toml** - Project metadata, tool configuration (pytest, black, ruff, mypy)
2. **requirements.txt** - Production dependencies
3. **requirements-dev.txt** - Development dependencies
4. **env.example** - Environment variables template
5. **.gitignore** - Files to ignore in git
6. **.cursorrules** - Cursor-specific rules and conventions
7. **.pre-commit-config.yaml** - Pre-commit hooks configuration
8. **Makefile** - Common development commands

### 📁 Documentation

Located in `docs/`:
- **ARCHITECTURE.md** - Detailed system architecture and design
- **GETTING_STARTED.md** - Setup and getting started guide

### 📂 Project Structure

The following directory structure is recommended (you'll create these):

```
recursive-agents/
├── src/
│   └── rlm/                      # Main package
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── exceptions.py         # Custom exceptions
│       │
│       ├── core/                 # Core recursive engine
│       │   ├── __init__.py
│       │   ├── engine.py         # RecursiveInferenceEngine
│       │   ├── prompts.py        # Prompt templates
│       │   ├── models.py         # Data models
│       │   └── aggregation.py   # Result aggregation
│       │
│       ├── models/               # LLM integrations
│       │   ├── __init__.py
│       │   ├── base.py           # BaseLLM interface
│       │   ├── openai_llm.py    # OpenAI implementation
│       │   ├── anthropic_llm.py # Anthropic implementation
│       │   ├── factory.py        # LLM factory
│       │   └── cache.py          # Response caching
│       │
│       ├── context/              # Context management
│       │   ├── __init__.py
│       │   ├── manager.py        # ContextManager
│       │   ├── chunking.py       # Chunking strategies
│       │   └── models.py         # Data models
│       │
│       ├── execution/            # Safe code execution
│       │   ├── __init__.py
│       │   ├── sandbox.py        # Sandbox environment
│       │   ├── validator.py      # Code validation
│       │   ├── context_api.py    # API for generated code
│       │   └── environment.py    # Execution environment
│       │
│       └── utils/                # Utilities
│           ├── __init__.py
│           └── logging.py        # Logging setup
│
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── security/                 # Security tests
│
├── examples/
│   ├── basic_rlm_demo.py        # Basic demo
│   ├── book_summary.py          # Book summarization
│   └── code_analysis.py         # Codebase analysis
│
├── benchmarks/                   # Performance benchmarks
└── logs/                         # Log files
```

## How to Use This Scaffolding

### Option 1: Manual Implementation (Learning Path)

Follow the **IMPLEMENTATION_GUIDE.md** step by step:
1. Set up environment (Phase 1)
2. Implement LLM abstractions (Phase 2)
3. Build context management (Phase 3)
4. Create safe code execution (Phase 4)
5. Implement recursive engine (Phase 5)
6. Write comprehensive tests (Phase 6)
7. Create examples (Phase 7)

### Option 2: Use Claude Code (Recommended)

Let Claude Code help you implement the system:

1. **Start with a clear prompt**:
   ```
   I want to implement the Recursive Inference Engine. Please read:
   - CLAUDE.md for project guidelines
   - SPEC.md Section "Component 1: Recursive Inference Engine"
   - context/patterns.md for coding patterns
   - IMPLEMENTATION_GUIDE.md Phase 5
   
   Then implement src/rlm/core/engine.py with the RecursiveInferenceEngine class.
   ```

2. **Point Claude to relevant context**:
   ```
   I'm implementing code execution security. Read context/gotchas.md section on
   "Code Execution Security" and implement src/rlm/execution/sandbox.py with
   maximum security as the priority.
   ```

3. **Ask for tests alongside implementation**:
   ```
   Review the RecursiveInferenceEngine implementation and create comprehensive
   unit tests in tests/unit/test_engine.py. Include edge cases and error scenarios.
   ```

## Key Features of This Scaffolding

### ✅ Claude Code Optimized
- **CLAUDE.md**: Direct instructions for Claude Code
- **Context files**: Background information Claude can reference
- **Clear patterns**: Consistent coding patterns throughout
- **Type hints**: Comprehensive type information for better AI assistance

### ✅ Development Tools Ready
- **Testing**: pytest with async support, coverage reporting
- **Linting**: ruff for fast Python linting
- **Formatting**: black and isort for consistent code style
- **Type checking**: mypy for static type analysis
- **Pre-commit hooks**: Automatic checks before commits

### ✅ Security First
- Multiple security layers for code execution
- RestrictedPython for sandboxing
- Comprehensive security tests
- Clear security guidelines in documentation

### ✅ Production Ready Architecture
- Async-first design for performance
- Pydantic for data validation
- Structured logging with loguru
- Error handling with custom exceptions
- Caching for LLM responses
- Configuration management

### ✅ Comprehensive Documentation
- System architecture
- Implementation guide
- Code patterns
- Architecture decisions
- Known issues and gotchas

## Quick Start Commands

### For Local Models (Recommended) 🚀

```powershell
# 1. Install Ollama (easiest local model setup)
winget install Ollama.Ollama

# 2. Download a model
ollama pull mistral:7b      # Best overall (5.4GB VRAM)
ollama pull phi3.5:3.8b     # Fastest, 128K context (2.8GB VRAM)
ollama pull tinyllama:1.1b  # For testing (0.8GB VRAM)

# 3. Set up Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure for local models
copy env.example .env
# Edit .env: Set USE_LOCAL_MODEL=true, OLLAMA_MODEL=mistral:7b

# 5. Test your setup (after implementation)
python test_local.py
```

### For Cloud APIs (Optional)

```bash
# Copy environment template
cp env.example .env
# Edit .env and add API keys:
# - OPENAI_API_KEY (if using OpenAI)
# - GROQ_API_KEY (free tier, very fast!)
# - ANTHROPIC_API_KEY (if using Claude)

# Install dependencies
make install-dev
```

### Development Commands

```bash
# Run tests
make test

# Format and lint code
make format
make lint

# Run example
make run-example

# View all commands
make help
```

## Development Workflow

1. **Read documentation**: Start with README.md, then SPEC.md
2. **Understand research**: Read context/research.md
3. **Follow implementation guide**: IMPLEMENTATION_GUIDE.md
4. **Use Claude Code**: Reference CLAUDE.md for best practices
5. **Write tests**: TDD approach recommended
6. **Check security**: Security is critical for this project
7. **Run benchmarks**: Compare to baseline approaches

## Next Steps

### Immediate Actions

1. **Set up environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Copy and configure .env**:
   ```bash
   copy env.example .env  # Windows
   # Edit .env and add API keys
   ```

3. **Create package structure**:
   ```bash
   mkdir -p src/rlm/{core,models,context,execution,utils}
   mkdir -p tests/{unit,integration,security}
   mkdir -p examples benchmarks logs
   ```

4. **Start implementing**: Follow IMPLEMENTATION_GUIDE.md Phase 1

### Suggested Implementation Order

1. ✅ **Phase 1**: Foundation (config, exceptions, logging)
2. ✅ **Phase 2**: LLM abstractions (OpenAI, Anthropic)
3. ✅ **Phase 3**: Context management (chunking, storage)
4. ✅ **Phase 4**: Safe code execution (sandbox, validation)
5. ✅ **Phase 5**: Recursive engine (core logic)
6. ✅ **Phase 6**: Testing (unit, integration, security)
7. ✅ **Phase 7**: Examples and docs

Each phase builds on the previous one.

## Resources

- **Research Paper**: https://www.tekta.ai/ai-research-papers/recursive-language-models-2025
- **OpenAI Docs**: https://platform.openai.com/docs
- **Anthropic Docs**: https://docs.anthropic.com
- **RestrictedPython**: https://restrictedpython.readthedocs.io/
- **Pydantic**: https://docs.pydantic.dev/

## Success Criteria

You'll know you've successfully implemented RLM when:
- ✅ Can process contexts 10x larger than LLM window
- ✅ All tests pass (especially security tests)
- ✅ Examples run end-to-end
- ✅ Security: malicious code is blocked
- ✅ Performance: reasonable latency (<60s for typical queries)
- ✅ Code quality: passes linting and type checking

## Support

If you get stuck:
1. Check **context/gotchas.md** for known issues
2. Review **SPEC.md** for technical details
3. Check **docs/ARCHITECTURE.md** for design decisions
4. Use Claude Code with specific questions referencing these docs

## License

Add your preferred license (MIT recommended for research projects).

## Contributing

This is a research implementation. Contributions welcome! Please:
1. Follow patterns in context/patterns.md
2. Write tests for new features
3. Update documentation
4. Run `make check-all` before submitting

---

**You now have everything you need to implement and test the Recursive Language Models research paper!**

Start with `docs/GETTING_STARTED.md` and then follow `IMPLEMENTATION_GUIDE.md`.

Happy coding! 🚀

