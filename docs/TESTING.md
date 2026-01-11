# Testing Guide

Complete guide to testing your RLM implementation.

## 🎯 Quick Test Commands

```powershell
# Test everything
pytest tests/ -v

# Test specific areas
pytest tests/unit/ -v              # Unit tests
pytest tests/integration/ -v       # Integration tests
pytest tests/security/ -v          # Security tests

# With coverage
pytest tests/ -v --cov=src/rlm --cov-report=html

# Run examples
python examples/test_ollama_integration.py
python examples/demo_recursive_processing.py
```

---

## 📊 Test Levels

### Level 1: Smoke Tests (5 minutes)
**Goal:** Verify basic functionality

```powershell
# 1. Test Ollama connection
ollama run mistral:7b "Say hello"

# 2. Test package installation
python -c "import rlm; print(rlm.__version__)"

# 3. Test basic integration
python examples/test_ollama_integration.py
```

**Success criteria:**
- ✅ Ollama responds
- ✅ Package imports without errors
- ✅ Integration test passes

---

### Level 2: Unit Tests (10 minutes)
**Goal:** Test individual components

```powershell
# Run all unit tests
pytest tests/unit/ -v

# Or test specific components
pytest tests/unit/test_config.py -v
pytest tests/unit/test_context.py -v
pytest tests/unit/test_exceptions.py -v
pytest tests/unit/test_validator.py -v
```

**What's tested:**
- ✅ Configuration loading
- ✅ Context chunking (4 strategies)
- ✅ Exception handling
- ✅ Code validation
- ✅ Data models

**Success criteria:**
- All tests pass
- No import errors
- Models validate correctly

---

### Level 3: Integration Tests (15 minutes)
**Goal:** Test components working together

```powershell
# Run integration tests
pytest tests/integration/ -v

# Run examples (integration tests in practice)
python examples/demo_recursive_processing.py
```

**What's tested:**
- ✅ LLM client creation
- ✅ End-to-end text generation
- ✅ Code generation
- ✅ Context management with LLM
- ✅ Multi-component workflows

**Success criteria:**
- Components integrate smoothly
- LLM responses are sensible
- No crashes or hangs

---

### Level 4: Security Tests (10 minutes)
**Goal:** Verify sandboxing works

```powershell
# Run security tests
pytest tests/security/ -v

# Should test that dangerous code is blocked
pytest tests/security/test_sandbox_security.py -v
```

**What's tested:**
- ✅ Malicious code is blocked
- ✅ Resource limits work
- ✅ Timeouts are enforced
- ✅ File system access is restricted
- ✅ Network access is blocked

**Success criteria:**
- All malicious code attempts fail
- System remains secure
- No privilege escalation

---

### Level 5: Performance Tests (30 minutes)
**Goal:** Measure speed and efficiency

```powershell
# Run benchmarks
python benchmarks/run_benchmarks.py

# Or manual testing
python examples/demo_full_recursion.py
```

**What to measure:**
- 🏃 Tokens/second (should be 40-50 on RTX 3060 Ti)
- ⏱️ Latency (first token, total time)
- 💾 Memory usage (should stay under 6GB VRAM)
- 🔄 Recursion overhead

**Success criteria:**
- Speed: 40-50 tok/s for 7B models
- Latency: <1s first token
- Memory: <6GB VRAM for Mistral 7B
- No memory leaks

---

## 🧪 Detailed Test Descriptions

### Unit Tests

#### `test_config.py`
```python
# Tests configuration loading
- Loading from .env
- Validation (max_depth, timeout, etc.)
- Default values
- Invalid config handling
```

#### `test_context.py`
```python
# Tests context management
- Fixed-size chunking
- Semantic chunking
- Hierarchical chunking
- Adaptive chunking
- Chunk retrieval
- Token counting
```

#### `test_exceptions.py`
```python
# Tests custom exceptions
- Exception hierarchy
- Error messages
- Context preservation
```

#### `test_validator.py`
```python
# Tests code validation
- AST parsing
- Security checks
- Forbidden operations
- Whitelist enforcement
```

### Security Tests

#### `test_sandbox_security.py`
```python
# Tests sandbox security
- Block os.system()
- Block file operations
- Block network access
- Block __import__
- Timeout enforcement
- Memory limits
```

**Example malicious code that SHOULD be blocked:**
```python
import os; os.system('rm -rf /')
__import__('requests').get('evil.com')
open('/etc/passwd').read()
while True: pass  # infinite loop
```

### Integration Tests

#### `test_full_workflow.py` (if exists)
```python
# Tests end-to-end workflows
- Simple query (no recursion)
- Single-level recursion
- Multi-level recursion
- Error recovery
- Result aggregation
```

---

## 🎮 Interactive Testing

### Test Your Own Queries

```powershell
# Start Python REPL
python

# Then run interactively:
```

```python
import asyncio
from rlm.models import create_llm

async def test_query(prompt):
    llm = create_llm("ollama", model="mistral:7b")
    response = await llm.generate(prompt)
    print(response.content)

# Run it
asyncio.run(test_query("Your question here"))
```

### Test Different Models

```python
# Test Phi-3.5 (faster, smaller)
llm = create_llm("ollama", model="phi3.5:3.8b")

# Test TinyLlama (testing/dev)
llm = create_llm("ollama", model="tinyllama:1.1b")

# Test Groq (cloud, super fast)
llm = create_llm("groq", api_key="your-key", model="llama-3.1-8b-instant")
```

---

## 🐛 Debugging Failed Tests

### ImportError: No module named 'rlm'
**Fix:**
```powershell
pip install -e .
```

### Connection refused (Ollama)
**Fix:**
```powershell
ollama serve
```

### Model not found
**Fix:**
```powershell
ollama pull mistral:7b
```

### Tests timeout
**Possible causes:**
1. Model is slow (expected for large models)
2. Increase timeout in pytest.ini
3. Use smaller model for testing

**Fix:**
```python
# In test file
@pytest.mark.timeout(60)  # Increase timeout
async def test_slow_operation():
    ...
```

### Security tests fail (code NOT blocked)
**This is SERIOUS!** 
- Check validator logic
- Update RestrictedPython
- Review sandbox implementation
- Add more security layers

---

## 📈 Test Coverage

### Check Coverage

```powershell
# Generate coverage report
pytest tests/ --cov=src/rlm --cov-report=html

# View report
start htmlcov/index.html  # Windows
```

### Coverage Goals

| Component | Target | Priority |
|-----------|--------|----------|
| **Core engine** | 90%+ | Critical |
| **Execution/sandbox** | 100% | Critical (security!) |
| **LLM models** | 80%+ | High |
| **Context management** | 85%+ | High |
| **Utils** | 70%+ | Medium |

---

## 🎯 Testing Checklist

Before pushing code:

- [ ] All unit tests pass
- [ ] All security tests pass
- [ ] Integration test passes
- [ ] Examples run without errors
- [ ] Coverage >80% for core modules
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Documentation updated

```powershell
# Run all checks
pytest tests/ -v --cov=src/rlm
ruff check src/
black --check src/
mypy src/
```

Or use the Makefile:
```powershell
make check-all
```

---

## 🚀 Continuous Testing

### Watch Mode (auto-run tests)

```powershell
# Install pytest-watch
pip install pytest-watch

# Run in watch mode
ptw tests/
```

Now tests auto-run when you save files!

### Pre-commit Hooks

```powershell
# Install hooks
pre-commit install

# Now tests run automatically before each commit
git commit -m "Your message"
```

---

## 📝 Writing New Tests

### Template for Unit Test

```python
import pytest
from rlm.your_module import YourClass

@pytest.fixture
def your_fixture():
    """Setup test data."""
    return YourClass()

def test_something(your_fixture):
    """Test description."""
    # Arrange
    input_data = "test"
    
    # Act
    result = your_fixture.process(input_data)
    
    # Assert
    assert result == expected
```

### Template for Async Test

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

### Template for Security Test

```python
import pytest
from rlm.execution import CodeValidator

@pytest.mark.parametrize("malicious_code", [
    "import os; os.system('evil')",
    "__import__('requests').get('bad.com')",
])
def test_blocks_malicious_code(malicious_code):
    """Ensure malicious code is blocked."""
    validator = CodeValidator()
    with pytest.raises(SecurityError):
        validator.validate(malicious_code)
```

---

## 🎓 Next Steps

1. **Start simple:** Run `pytest tests/unit/test_config.py -v`
2. **Test examples:** Run all examples in order
3. **Check coverage:** Generate HTML report
4. **Fix failures:** Debug any failing tests
5. **Write new tests:** Add tests for new features

---

## 📚 Resources

- **Pytest docs:** https://docs.pytest.org/
- **Coverage docs:** https://coverage.readthedocs.io/
- **Examples:** See `examples/` folder
- **Implementation guide:** `docs/IMPLEMENTATION_GUIDE.md`

