# Implementation Guide: Step-by-Step Instructions

This guide provides a sequential path through implementing the Recursive Language Models project. Follow these steps in order for best results.

## Overview

You're building a system that allows LLMs to process unlimited context by:
1. Generating code to examine large inputs
2. Recursively breaking down problems
3. Safely executing code in a sandbox
4. Aggregating results

## Phase 1: Foundation (Days 1-2)

### Step 1.1: Environment Setup

**What to do**:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

**Verify**:
```bash
python --version  # Should be 3.10+
pip list  # Should show all dependencies
```

### Step 1.2: Create Package Structure

**What to do**:
```bash
# Create directory structure
mkdir -p src/rlm/{core,models,context,execution,utils}
mkdir -p tests/{unit,integration,security}
mkdir -p examples
mkdir -p benchmarks

# Create __init__.py files
touch src/rlm/__init__.py
touch src/rlm/core/__init__.py
touch src/rlm/models/__init__.py
touch src/rlm/context/__init__.py
touch src/rlm/execution/__init__.py
touch src/rlm/utils/__init__.py
```

**Verify**: Run `tree src/` and check structure matches

### Step 1.3: Configuration Management

**Create**: `src/rlm/config.py`

**What to implement**:
- RLMConfig class (Pydantic BaseSettings)
- Load from environment variables
- Validation

**Example**:
```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional

class RLMConfig(BaseSettings):
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4"
    
    # System Configuration
    max_recursion_depth: int = Field(default=10, ge=1, le=50)
    execution_timeout: int = Field(default=5, ge=1, le=30)
    enable_caching: bool = True
    log_level: str = "INFO"
    
    @validator('openai_api_key', 'anthropic_api_key')
    def check_at_least_one_key(cls, v, values):
        # Ensure at least one API key is set
        pass
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
```

**Test**: Create `tests/unit/test_config.py` and test loading config

### Step 1.4: Exception Hierarchy

**Create**: `src/rlm/exceptions.py`

**What to implement**:
```python
class RLMException(Exception):
    """Base exception for RLM system"""

class ConfigurationError(RLMException):
    """Configuration is invalid"""

class MaxDepthExceededError(RLMException):
    """Recursion depth limit exceeded"""

class CodeExecutionError(RLMException):
    """Generated code execution failed"""

class SecurityError(RLMException):
    """Security violation detected"""

class LLMError(RLMException):
    """LLM API call failed"""
```

### Step 1.5: Logging Setup

**Create**: `src/rlm/utils/logging.py`

**What to implement**:
- Configure loguru
- Structured logging format
- Context-aware logging

**Example**:
```python
from loguru import logger
import sys

def setup_logging(level: str = "INFO"):
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=level
    )
    logger.add(
        "logs/rlm.log",
        rotation="100 MB",
        retention="10 days",
        level=level
    )
```

## Phase 2: LLM Abstractions (Days 3-4)

### Step 2.1: Base LLM Interface

**Create**: `src/rlm/models/base.py`

**What to implement**:
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional

class LLMResponse(BaseModel):
    content: str
    tokens_used: int
    model: str
    finish_reason: str

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
        pass
    
    @abstractmethod
    async def generate_code(
        self,
        task_description: str
    ) -> str:
        """Generate Python code for task"""
        pass
```

### Step 2.2: Local Model Implementation (Recommended for RTX 3060 Ti) ⭐

**Create**: `src/rlm/models/ollama_llm.py`

**What to implement**:
- OllamaLLM class inheriting from BaseLLM
- Local model via Ollama (easiest setup)
- Uses OpenAI-compatible API
- No API key needed, runs locally

**Example**:
```python
import openai
from .base import BaseLLM, LLMResponse

class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "mistral:7b"):
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key="ollama"  # Required but not used
        )
        self.model = model
    
    async def generate(self, prompt: str, ...) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.model,
            finish_reason=response.choices[0].finish_reason
        )
```

**Setup Ollama**:
```bash
# Install
winget install Ollama.Ollama

# Pull recommended model
ollama pull mistral:7b

# Start server (runs automatically on Windows)
ollama serve
```

**Test**: Create `tests/unit/test_ollama_llm.py`

### Step 2.3: OpenAI Implementation (Optional, for comparison)

**Create**: `src/rlm/models/openai_llm.py`

**What to implement**: Similar to Ollama but with OpenAI API

### Step 2.4: Groq Implementation (Fast Cloud Alternative)

**Create**: `src/rlm/models/groq_llm.py`

**What to implement**:
```python
from groq import AsyncGroq
from .base import BaseLLM, LLMResponse

class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
    
    async def generate(self, prompt: str, ...) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Process response...
```

**Benefits**:
- Free tier: 30 requests/min
- 300+ tokens/second (blazingly fast)
- No local resources needed
- Get API key: https://console.groq.com/

### Step 2.5: LLM Factory

**Create**: `src/rlm/models/factory.py`

**What to implement**:
```python
from .base import BaseLLM
from .ollama_llm import OllamaLLM
from .groq_llm import GroqLLM
from .openai_llm import OpenAILLM

def create_llm(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseLLM:
    if provider == "ollama":
        return OllamaLLM(
            base_url=base_url or "http://localhost:11434/v1",
            model=model or "mistral:7b"
        )
    elif provider == "groq":
        if not api_key:
            raise ValueError("Groq requires API key")
        return GroqLLM(api_key, model or "llama-3.1-8b-instant")
    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI requires API key")
        return OpenAILLM(api_key, model or "gpt-4")
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

## Phase 3: Context Management (Days 5-6)

### Step 3.1: Data Models

**Create**: `src/rlm/context/models.py`

**What to implement**:
```python
from pydantic import BaseModel
from typing import Dict, Any

class ContextChunk(BaseModel):
    chunk_id: int
    content: str
    start_pos: int
    end_pos: int
    tokens: int
    metadata: Dict[str, Any] = {}

class ChunkAccess(BaseModel):
    chunk_id: int
    timestamp: float
    access_type: str  # "read" or "write"
```

### Step 3.2: Chunking Strategies

**Create**: `src/rlm/context/chunking.py`

**What to implement**:
```python
from abc import ABC, abstractmethod
from typing import List

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, max_size: int) -> List[str]:
        pass

class FixedSizeChunking(ChunkingStrategy):
    def chunk(self, text: str, max_size: int) -> List[str]:
        # Split into fixed-size chunks
        pass

class SemanticChunking(ChunkingStrategy):
    def chunk(self, text: str, max_size: int) -> List[str]:
        # Split on paragraph boundaries
        pass
```

**Test**: Test each strategy with various inputs

### Step 3.3: Context Manager

**Create**: `src/rlm/context/manager.py`

**What to implement**:
```python
from typing import List, Optional
from .models import ContextChunk
from .chunking import ChunkingStrategy

class ContextManager:
    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
        self.chunks: List[ContextChunk] = []
        self.access_log = []
    
    def load_context(self, content: str, max_chunk_size: int = 4000):
        """Load and chunk the context"""
        pass
    
    def get_chunk(self, chunk_id: int) -> str:
        """Get specific chunk"""
        pass
    
    def get_chunk_range(self, start: int, end: int) -> str:
        """Get range of chunks"""
        pass
    
    def get_context_length(self) -> int:
        """Get total length in tokens"""
        pass
```

## Phase 4: Safe Code Execution (Days 7-8)

### Step 4.1: Security Validator

**Create**: `src/rlm/execution/validator.py`

**What to implement**:
```python
import ast
from typing import Set

ALLOWED_BUILTINS = {'len', 'str', 'int', 'float', 'list', 'dict', ...}
ALLOWED_MODULES = {'math', 're', 'json'}
FORBIDDEN_NAMES = {'eval', 'exec', '__import__', 'open', 'file', ...}

class CodeValidator:
    def validate(self, code: str) -> bool:
        """Validate code is safe to execute"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        # Check for forbidden operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Check imports
                pass
            if isinstance(node, ast.Name):
                # Check names
                pass
        
        return True
```

**Test**: CRITICAL - Test with malicious code examples

### Step 4.2: Sandbox Environment

**Create**: `src/rlm/execution/sandbox.py`

**What to implement**:
```python
from RestrictedPython import compile_restricted
from typing import Dict, Any
import signal

class SandboxEnvironment:
    def __init__(self, timeout: int = 5, memory_limit: int = 512*1024*1024):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_builtins = {...}
    
    async def execute(self, code: str, context_api: Dict[str, Any]) -> Any:
        """Execute code in sandbox with timeout"""
        # Compile with RestrictedPython
        byte_code = compile_restricted(code, '<string>', 'exec')
        
        # Set up restricted globals
        restricted_globals = {
            '__builtins__': self._get_safe_builtins(),
            **context_api  # Functions for accessing context
        }
        
        # Execute with timeout
        # Implementation...
```

### Step 4.3: Context Access API

**Create**: `src/rlm/execution/context_api.py`

**What to implement**:
```python
class ContextAPI:
    """API available to generated code for accessing context"""
    
    def __init__(self, context_manager, engine):
        self.context_manager = context_manager
        self.engine = engine
    
    def get_context_length(self) -> int:
        return self.context_manager.get_context_length()
    
    def get_context_chunk(self, start: int, end: int) -> str:
        return self.context_manager.get_chunk_range(start, end)
    
    async def call_submodel(self, chunk: str, query: str) -> str:
        # Make recursive call through engine
        pass
```

## Phase 5: Recursive Engine (Days 9-11)

### Step 5.1: Prompt Templates

**Create**: `src/rlm/core/prompts.py`

**What to implement**:
```python
SYSTEM_PROMPT = """You are a code-writing assistant for a Recursive Language Model.
Your task is to write Python code that examines the provided context and decides how to process it.

Available functions:
- get_context_length() -> int: Get total context length in tokens
- get_context_chunk(start: int, end: int) -> str: Get specific chunk
- call_submodel(chunk: str, query: str) -> str: Make recursive call
- aggregate_results(results: List[str]) -> str: Aggregate results

Your code should:
1. Examine the context structure using get_context_length()
2. Decide if recursion is needed (if context > 4000 tokens)
3. If yes: chunk context and make recursive calls with call_submodel()
4. If no: return the direct answer
5. Return the final result
"""

def create_code_generation_prompt(query: str, context_summary: str) -> str:
    return f"""
Query: {query}
Context length: {context_summary}

Write Python code to process this query recursively if needed.
The code should return the final answer.
"""
```

### Step 5.2: Recursion Tree

**Create**: `src/rlm/core/models.py`

**What to implement**:
```python
from pydantic import BaseModel
from typing import List, Optional

class RecursionNode(BaseModel):
    depth: int
    query: str
    context_summary: str
    generated_code: Optional[str] = None
    result: Optional[str] = None
    sub_calls: List['RecursionNode'] = []
    tokens_used: int = 0
    execution_time: float = 0.0

class InferenceResult(BaseModel):
    answer: str
    recursion_tree: RecursionNode
    total_tokens: int
    execution_time: float
    num_recursive_calls: int
```

### Step 5.3: Recursive Inference Engine

**Create**: `src/rlm/core/engine.py`

**This is the heart of the system!**

**What to implement**:
```python
from typing import Optional
from ..models.base import BaseLLM
from ..context.manager import ContextManager
from ..execution.sandbox import SandboxEnvironment
from .models import RecursionNode, InferenceResult

class RecursiveInferenceEngine:
    def __init__(
        self,
        llm: BaseLLM,
        context_manager: ContextManager,
        sandbox: SandboxEnvironment,
        max_depth: int = 10
    ):
        self.llm = llm
        self.context_manager = context_manager
        self.sandbox = sandbox
        self.max_depth = max_depth
    
    async def process(
        self,
        query: str,
        context: str
    ) -> InferenceResult:
        """Main entry point"""
        # Load context
        self.context_manager.load_context(context)
        
        # Start recursive processing
        root_node = await self._recursive_step(query, 0)
        
        # Build result
        return InferenceResult(
            answer=root_node.result,
            recursion_tree=root_node,
            ...
        )
    
    async def _recursive_step(
        self,
        query: str,
        depth: int
    ) -> RecursionNode:
        """Execute one recursion step"""
        # Check depth
        if depth >= self.max_depth:
            raise MaxDepthExceededError(depth, self.max_depth)
        
        # Generate code
        code = await self._generate_code(query, depth)
        
        # Create context API
        context_api = self._create_context_api(depth)
        
        # Execute code
        result = await self.sandbox.execute(code, context_api)
        
        # Build node
        node = RecursionNode(
            depth=depth,
            query=query,
            generated_code=code,
            result=result,
            ...
        )
        
        return node
    
    async def _generate_code(self, query: str, depth: int) -> str:
        """Generate code for this step"""
        prompt = create_code_generation_prompt(
            query,
            self.context_manager.get_context_summary()
        )
        response = await self.llm.generate_code(prompt)
        return response
```

## Phase 6: Testing (Days 12-13)

### Step 6.1: Unit Tests

For each module, create comprehensive unit tests:
- Test with mocked dependencies
- Test edge cases
- Test error handling

### Step 6.2: Integration Tests

**Create**: `tests/integration/test_full_workflow.py`

**What to test**:
```python
@pytest.mark.asyncio
async def test_simple_query_no_recursion():
    """Test with context that fits in one call"""
    pass

@pytest.mark.asyncio
async def test_single_level_recursion():
    """Test with context requiring one level of recursion"""
    pass

@pytest.mark.asyncio
async def test_multi_level_recursion():
    """Test with deep recursion"""
    pass
```

### Step 6.3: Security Tests

**Create**: `tests/security/test_code_execution.py`

**What to test**:
```python
@pytest.mark.parametrize("malicious_code", [
    "import os; os.system('rm -rf /')",
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "while True: pass",
    # Add more...
])
def test_malicious_code_blocked(malicious_code):
    """Ensure malicious code is blocked"""
    pass
```

## Phase 7: Examples (Days 14-15)

### Step 7.1: Basic Example

**Create**: `examples/basic_rlm_demo.py`

**What to implement**:
```python
import asyncio
from rlm import RecursiveInferenceEngine
from rlm.models import create_llm
from rlm.context import ContextManager, SemanticChunking
from rlm.execution import SandboxEnvironment
from rlm.config import RLMConfig

async def main():
    # Load config
    config = RLMConfig()
    
    # Create components
    llm = create_llm("openai", config.openai_api_key, "gpt-4")
    context_manager = ContextManager(SemanticChunking())
    sandbox = SandboxEnvironment(timeout=5)
    
    # Create engine
    engine = RecursiveInferenceEngine(
        llm=llm,
        context_manager=context_manager,
        sandbox=sandbox,
        max_depth=10
    )
    
    # Process query
    long_text = "..." * 10000  # Long context
    result = await engine.process(
        query="Summarize this text",
        context=long_text
    )
    
    print(f"Answer: {result.answer}")
    print(f"Tokens used: {result.total_tokens}")
    print(f"Recursive calls: {result.num_recursive_calls}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 7.2: Advanced Examples

- Book summarization
- Codebase analysis
- Multi-document QA

## Tips for Success

1. **Work Incrementally**: Implement and test each component before moving on
2. **Use Claude Code**: Ask Claude to help implement following the patterns
3. **Test Continuously**: Run tests after each change
4. **Check Examples**: Look at `context/patterns.md` for code patterns
5. **Security First**: Be extra careful with code execution
6. **Ask Questions**: Better to clarify than make wrong assumptions

## Getting Unstuck

If you're stuck on a component:
1. Read the relevant section in `SPEC.md`
2. Check `context/patterns.md` for code patterns
3. Look at `context/gotchas.md` for known issues
4. Ask Claude Code with specific context from these files

## Success Criteria

You've successfully implemented the system when:
- [ ] All tests pass
- [ ] Basic example runs end-to-end
- [ ] Can process contexts larger than LLM window
- [ ] Security tests show malicious code is blocked
- [ ] Performance is reasonable (< 60s for typical queries)

Good luck! 🚀

