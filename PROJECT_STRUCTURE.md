# Project Structure

## Overview

```
recursive-agents/
│
├── 📄 README.md                    Main project documentation
├── 📄 Makefile                     Build commands and shortcuts
├── 📄 pyproject.toml               Project configuration
├── 📄 requirements.txt             Production dependencies
├── 📄 requirements-dev.txt         Development dependencies
├── 📄 .gitignore                   Git ignore rules
├── 📄 .pre-commit-config.yaml      Pre-commit hooks
├── 📄 env.example                  Environment variables template
│
├── 📁 .claude/                     AI assistant context
│   ├── CLAUDE.md                   Instructions for Claude Code
│   ├── TODO.md                     Development tasks
│   ├── PROJECT_SUMMARY.md          Project overview
│   ├── decisions.md                Architecture decisions
│   ├── gotchas.md                  Known issues
│   ├── patterns.md                 Code patterns
│   └── research.md                 Research paper summary
│
├── 📁 docs/                        Documentation
│   ├── GETTING_STARTED.md          Quick start guide
│   ├── QUICKSTART_LOCAL.md         5-minute local setup
│   ├── INSTALL_OLLAMA.md           Ollama installation
│   ├── IMPLEMENTATION_GUIDE.md     Step-by-step build guide
│   ├── ARCHITECTURE.md             System architecture
│   ├── SPEC.md                     Technical specification
│   ├── LOCAL_MODELS_GUIDE.md       Local models guide
│   └── HARDWARE_OPTIMIZED_SETUP.md GPU-specific setup
│
├── 📁 src/rlm/                     Source code
│   ├── __init__.py                 Package initialization
│   ├── config.py                   Configuration management
│   ├── exceptions.py               Custom exceptions
│   │
│   ├── 📁 core/                    Recursive inference engine
│   │   ├── engine.py               Main recursive engine
│   │   ├── models.py               Data models
│   │   └── aggregation.py          Result aggregation
│   │
│   ├── 📁 models/                  LLM integrations
│   │   ├── base.py                 Base LLM interface
│   │   ├── ollama_llm.py           Ollama (local) integration
│   │   ├── groq_llm.py             Groq cloud integration
│   │   ├── openai_llm.py           OpenAI integration
│   │   ├── anthropic_llm.py        Anthropic integration
│   │   ├── factory.py              LLM factory
│   │   └── prompts.py              Prompt templates
│   │
│   ├── 📁 context/                 Context management
│   │   ├── manager.py              Context manager
│   │   ├── chunking.py             Chunking strategies
│   │   └── models.py               Data models
│   │
│   ├── 📁 execution/               Safe code execution
│   │   ├── sandbox.py              Sandboxed execution
│   │   └── validator.py            Code validation
│   │
│   └── 📁 utils/                   Utilities
│       └── logging.py              Logging configuration
│
├── 📁 tests/                       Test suite
│   ├── conftest.py                 Pytest configuration
│   ├── 📁 unit/                    Unit tests
│   ├── 📁 integration/             Integration tests
│   └── 📁 security/                Security tests
│
├── 📁 examples/                    Example scripts
│   ├── basic_rlm_demo.py           Basic RLM demo
│   └── test_ollama_integration.py  Ollama integration test
│
└── 📁 benchmarks/                  Performance benchmarks
```

## Quick Navigation

### 🚀 **Getting Started**
1. Read [`README.md`](README.md)
2. Follow [`docs/QUICKSTART_LOCAL.md`](docs/QUICKSTART_LOCAL.md)
3. Run `examples/test_ollama_integration.py`

### 📚 **Documentation**
- **User Guides**: `docs/GETTING_STARTED.md`, `docs/QUICKSTART_LOCAL.md`
- **Installation**: `docs/INSTALL_OLLAMA.md`
- **Architecture**: `docs/ARCHITECTURE.md`, `docs/SPEC.md`
- **Development**: `docs/IMPLEMENTATION_GUIDE.md`
- **Local Models**: `docs/LOCAL_MODELS_GUIDE.md`

### 🤖 **For AI Assistants**
- **Start here**: `.claude/CLAUDE.md`
- **Tasks**: `.claude/TODO.md`
- **Patterns**: `.claude/patterns.md`
- **Decisions**: `.claude/decisions.md`

### 💻 **Development**
- **Source code**: `src/rlm/`
- **Tests**: `tests/`
- **Examples**: `examples/`
- **Configuration**: `pyproject.toml`, `requirements.txt`

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `docs/QUICKSTART_LOCAL.md` | 5-minute setup with local models |
| `docs/IMPLEMENTATION_GUIDE.md` | Step-by-step build instructions |
| `.claude/CLAUDE.md` | Instructions for AI assistants |
| `src/rlm/core/engine.py` | Main recursive inference engine |
| `examples/test_ollama_integration.py` | Test local model integration |

## Development Workflow

1. **Setup**: Follow `docs/QUICKSTART_LOCAL.md`
2. **Development**: Refer to `.claude/patterns.md` for code patterns
3. **Testing**: Run `pytest tests/`
4. **Examples**: Check `examples/` for usage
5. **Documentation**: Update relevant docs when changing features

## Ignored Files/Folders

These are automatically ignored by Git (see `.gitignore`):
- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `.env` - Environment secrets
- `logs/` - Log files
- `*.egg-info/` - Build artifacts
- `.cursor/` - IDE-specific files

