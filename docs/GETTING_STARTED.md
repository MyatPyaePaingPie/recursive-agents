# Getting Started with Recursive Language Models

This guide will help you get up and running with the RLM project.

## Prerequisites

- **Python 3.10 or higher**
- **pip** and **venv** (usually included with Python)
- **Git** (for cloning the repository)
- **API Key** for OpenAI or Anthropic (optional for initial setup, required for running)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/recursive-language-models.git
cd recursive-language-models
```

### 2. Create Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies (production + development)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Or use the Makefile:
```bash
make install-dev
```

### 4. Configure Environment Variables

Copy the example environment file:
```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

Edit `.env` and add your API keys:
```bash
OPENAI_API_KEY=sk-your-key-here
# or
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Note**: You need at least one API key to run the examples.

### 5. Verify Installation

Run the tests to make sure everything is set up correctly:
```bash
make test-fast
```

If you see all tests passing, you're good to go!

## Project Structure Overview

```
recursive-agents/
├── README.md              # Project overview
├── CLAUDE.md              # Instructions for Claude Code
├── SPEC.md                # Technical specification
├── TODO.md                # Development roadmap
│
├── context/               # Context for Claude Code
│   ├── research.md        # Research paper summary
│   ├── patterns.md        # Coding patterns
│   ├── decisions.md       # Architecture decisions
│   └── gotchas.md         # Known issues
│
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # System architecture
│   └── GETTING_STARTED.md # This file
│
├── src/                   # Source code
│   └── rlm/              # Main package
│       ├── core/         # Core recursive engine
│       ├── models/       # LLM integrations
│       ├── context/      # Context management
│       ├── execution/    # Code execution
│       └── utils/        # Utilities
│
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── security/        # Security tests
│
├── examples/            # Example implementations
│   ├── basic_rlm_demo.py
│   ├── book_summary.py
│   └── code_analysis.py
│
└── benchmarks/          # Performance benchmarks
```

## Quick Start: Running Examples

Once you have the core components implemented (see TODO.md), you can run examples:

### Basic Demo
```bash
python examples/basic_rlm_demo.py
```

### Book Summarization
```bash
python examples/book_summary.py
```

### All Examples
```bash
make run-example
```

## Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code** following the patterns in `context/patterns.md`

3. **Write tests** alongside your implementation:
   ```bash
   # Create test file
   touch tests/unit/test_your_module.py
   ```

4. **Run tests**:
   ```bash
   make test
   ```

5. **Format and lint**:
   ```bash
   make format
   make lint
   ```

6. **Type check**:
   ```bash
   make type-check
   ```

7. **Run all checks**:
   ```bash
   make check-all
   ```

### Using Pre-commit Hooks

Install pre-commit hooks to automatically check your code:
```bash
pre-commit install
```

Now checks will run automatically before each commit.

## Working with Claude Code

This project is optimized for use with Claude Code (Cursor). Here's how to get the most out of it:

### Key Files for Claude

1. **CLAUDE.md**: Instructions specifically for Claude Code
2. **SPEC.md**: Technical specifications
3. **context/**: Background information and patterns

### Starting a Conversation with Claude

**Good prompts**:
```
"I need to implement the RecursiveInferenceEngine class.
Review CLAUDE.md and SPEC.md, then create the initial implementation
in src/rlm/core/engine.py with comprehensive docstrings."
```

```
"Review context/patterns.md and implement the OpenAILLM class
following Pattern 2 (Pydantic validation) and Pattern 3 (dependency injection).
Include unit tests."
```

**Bad prompts**:
```
"Write code"  # Too vague
"Create engine"  # Needs more context
```

### Pointing Claude to Context

When asking Claude to work on something specific:
```
"I'm implementing code execution. Read context/gotchas.md section on
Code Execution Security, then implement the Sandbox class with
security as the top priority."
```

## Common Tasks

### Running Tests
```bash
# All tests with coverage
make test

# Fast (no coverage report)
make test-fast

# Only unit tests
make test-unit

# Only integration tests
make test-integration

# Only security tests
make test-security
```

### Code Quality
```bash
# Format code
make format

# Check formatting (no changes)
make format-check

# Lint
make lint

# Lint with auto-fix
make lint-fix

# Type check
make type-check

# Everything
make check-all
```

### Cleaning Up
```bash
# Remove generated files
make clean
```

### Documentation
```bash
# Serve docs locally
make docs-serve

# Build docs
make docs-build
```

## Development Phases

The project is organized into phases (see TODO.md):

1. **Phase 1**: Core Infrastructure ⚡
   - Project structure, dependencies, config

2. **Phase 2**: LLM Abstractions 🤖
   - Base LLM interface, OpenAI/Anthropic integrations

3. **Phase 3**: Context Management 📚
   - Chunking strategies, context storage

4. **Phase 4**: Safe Code Execution 🔒
   - Sandboxing, validation, security

5. **Phase 5**: Recursive Inference Engine 🔄
   - Core recursive logic, orchestration

6. **Phase 6**: Testing & Quality 🧪
   - Comprehensive test suite, benchmarks

7. **Phase 7**: Examples & Documentation 📖
   - Working examples, detailed docs

8. **Phase 8**: Optional Enhancements 🚀
   - Web API, Docker, monitoring

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'rlm'`
**Solution**: Make sure you're in the virtual environment and have installed dependencies:
```bash
pip install -r requirements.txt
```

**Issue**: `OpenAI API key not set`
**Solution**: Set your API key in `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
```

**Issue**: Tests failing with async errors
**Solution**: Make sure you have pytest-asyncio installed:
```bash
pip install pytest-asyncio
```

**Issue**: Import errors in VS Code/Cursor
**Solution**: Set Python interpreter to your venv:
- Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
- Type "Python: Select Interpreter"
- Choose the interpreter from `venv/`

### Getting Help

1. Check `context/gotchas.md` for known issues
2. Review `SPEC.md` for technical details
3. Check `docs/ARCHITECTURE.md` for design decisions
4. Search existing issues on GitHub
5. Create a new issue with:
   - What you were trying to do
   - What you expected
   - What actually happened
   - Error messages (full traceback)

## Next Steps

Now that you're set up:

1. **Understand the Research**: Read `context/research.md`
2. **Review Architecture**: Read `docs/ARCHITECTURE.md`
3. **Check TODO**: See what's next in `TODO.md`
4. **Start Coding**: Begin with Phase 1 tasks
5. **Ask Claude**: Use Claude Code to help implement features

## Additional Resources

- [Research Paper](https://www.tekta.ai/ai-research-papers/recursive-language-models-2025)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [RestrictedPython Documentation](https://restrictedpython.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

Happy coding! 🚀


