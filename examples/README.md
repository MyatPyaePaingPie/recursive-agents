# Examples

This folder contains working examples demonstrating the RLM system.

## 🚀 Quick Start (Run These in Order)

### 1. **test_ollama_integration.py** - Basic Connection Test
```powershell
python examples/test_ollama_integration.py
```

**What it does:**
- ✅ Tests connection to Ollama
- ✅ Verifies Mistral 7B is working
- ✅ Tests text generation
- ✅ Tests code generation

**Expected output:** Success messages with generated text and code

**Time:** ~10 seconds

---

### 2. **demo_recursive_processing.py** - Simple LLM Demo
```powershell
python examples/demo_recursive_processing.py
```

**What it does:**
- Shows how to use the LLM client
- Processes a document
- Generates code
- Demonstrates basic functionality

**Expected output:** Summary of a document, generated code

**Time:** ~20 seconds

---

### 3. **demo_full_recursion.py** - Full RLM System (Advanced)
```powershell
python examples/demo_full_recursion.py
```

**What it does:**
- Uses the COMPLETE RLM engine
- Processes large context recursively
- Shows recursion tree
- Demonstrates unlimited context processing

**Expected output:** Recursive processing with statistics

**Time:** ~30-60 seconds

**Note:** Requires full implementation (Phases 1-5 complete)

---

### 4. **basic_rlm_demo.py** - Original Demo
```powershell
python examples/basic_rlm_demo.py
```

The original RLM demo from the implementation guide.

---

## 🎯 What to Run Based on What You Want

### Just want to test if Ollama works?
→ Run `test_ollama_integration.py`

### Want to see text/code generation?
→ Run `demo_recursive_processing.py`

### Want to see FULL recursive processing?
→ Run `demo_full_recursion.py` (requires complete implementation)

### Want to build your own?
→ Copy `demo_recursive_processing.py` and modify!

---

## 📊 Expected Performance (RTX 3060 Ti)

| Example | Time | Tokens | Speed |
|---------|------|--------|-------|
| test_ollama_integration.py | ~10s | ~100 | 40-50 tok/s |
| demo_recursive_processing.py | ~20s | ~200 | 40-50 tok/s |
| demo_full_recursion.py | ~60s | ~500+ | 40-50 tok/s |

---

## 🐛 Troubleshooting

### "ollama: command not found"
**Fix:** Install Ollama: `winget install Ollama.Ollama`

### "Model 'mistral:7b' not found"
**Fix:** Pull the model: `ollama pull mistral:7b`

### "Connection refused"
**Fix:** Start Ollama: `ollama serve`

### Module import errors
**Fix:** Install package: `pip install -e .`

### Slow performance
**Check:**
1. GPU is being used (Task Manager → GPU)
2. Other apps aren't using VRAM
3. Try smaller model: `ollama pull phi3.5:3.8b`

---

## 🎓 Learning Path

1. **Start simple:** `test_ollama_integration.py`
2. **Understand the code:** Read `demo_recursive_processing.py`
3. **Try full system:** `demo_full_recursion.py`
4. **Customize:** Copy and modify examples
5. **Build your own:** Create new examples!

---

## 💡 Ideas for Your Own Examples

- **Document Q&A:** Load a PDF and answer questions
- **Code Analysis:** Analyze a large codebase recursively
- **Book Summary:** Process an entire book chapter by chapter
- **Multi-doc Search:** Search across multiple documents
- **Creative Writing:** Generate long-form content recursively

---

## 🔗 More Resources

- **Implementation Guide:** `docs/IMPLEMENTATION_GUIDE.md`
- **Architecture:** `docs/ARCHITECTURE.md`
- **Local Models:** `docs/LOCAL_MODELS_GUIDE.md`
- **Troubleshooting:** `docs/INSTALL_OLLAMA.md`

