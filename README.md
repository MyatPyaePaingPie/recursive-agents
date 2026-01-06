# Recursive Language Models (RLM) Implementation

## What This Is
This project implements the concepts from the research paper "Recursive Language Models: Processing Unlimited Context Through Code" (2025). It demonstrates how large language models can overcome fixed context window limitations by writing and executing code to recursively process arbitrarily long inputs.

## Core Concept
Traditional LLMs are limited by fixed context windows. RLMs solve this by:
1. Treating the prompt as an external environment variable
2. Writing code to examine and decompose large inputs
3. Recursively calling themselves on manageable snippets
4. Aggregating results to process unlimited context

## Tech Stack
- **Language**: Python 3.10+
- **LLM Framework**: Transformers (HuggingFace), OpenAI API, Anthropic API, Groq API
- **Code Execution**: RestrictedPython, Docker (for sandboxing)
- **Testing**: pytest, pytest-asyncio
- **Dependencies**: torch, numpy, pydantic

## Architecture
See `docs/ARCHITECTURE.md` for detailed architecture documentation.

The system consists of:
- **Recursive Inference Engine**: Orchestrates recursive prompt processing
- **Code Generation & Execution Module**: Generates and safely executes code
- **Context Manager**: Manages input chunking and state across recursive calls
- **Sub-model Handler**: Manages LLM invocations and result aggregation

## Getting Started

### Prerequisites
- **Python 3.10 or higher**
- **RTX 3060 Ti 8GB** (or similar GPU for local models)
- **Docker** (optional, for sandboxed code execution)

### Quick Start with Local Models (Recommended) 🚀

Perfect for your i7 + RTX 3060 Ti + Windows 11 setup!

```powershell
# 1. Install Ollama (easiest way to run local models)
winget install Ollama.Ollama

# 2. Download a model
ollama pull mistral:7b  # Best overall, ~5.4GB VRAM
# OR
ollama pull phi3.5:3.8b  # Fastest, 128K context, ~2.8GB VRAM
# OR  
ollama pull tinyllama:1.1b  # For testing, ~0.8GB VRAM

# 3. Set up Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure for local models
copy env.example .env
# Edit .env: Set USE_LOCAL_MODEL=true, OLLAMA_MODEL=mistral:7b

# 5. Test (after implementation)
python examples/basic_rlm_demo.py
```

**See `QUICKSTART_LOCAL.md` for detailed 5-minute setup guide!**

### Alternative: Cloud APIs

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure API keys
copy env.example .env  # Windows: copy env.example .env
# Edit .env and add:
# - OPENAI_API_KEY (if using OpenAI)
# - GROQ_API_KEY (free tier, blazingly fast!)
# - ANTHROPIC_API_KEY (if using Claude)

# Run tests
pytest
```

## Current Status
- [x] Project setup and documentation
- [ ] Core recursive inference engine
- [ ] Code generation and safe execution module
- [ ] Context management system
- [ ] Sub-model handler
- [ ] Integration tests
- [ ] Performance benchmarking
- [ ] Example implementations

## Project Structure
```
recursive-agents/
├── src/                    # Core implementation
│   ├── rlm/               # Main RLM package
│   │   ├── core/          # Core recursive engine
│   │   ├── execution/     # Code execution environment
│   │   ├── context/       # Context management
│   │   └── models/        # LLM integrations
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── examples/              # Example implementations
├── docs/                  # Documentation
├── context/               # Project context for Claude Code
└── benchmarks/            # Performance tests
```

## Key Research Paper Findings
1. **Unlimited Context**: RLMs can process arbitrarily long inputs by recursively examining them
2. **Code as Interface**: Using code generation enables structured access to context
3. **Efficiency**: Only relevant portions of input are processed, improving efficiency
4. **Composability**: Recursive approach allows hierarchical problem decomposition

## Notes
- This is a research implementation focused on understanding and testing RLM concepts
- Sandboxed execution is critical for security when running generated code
- Performance benchmarks compare RLM approach vs traditional context window methods

