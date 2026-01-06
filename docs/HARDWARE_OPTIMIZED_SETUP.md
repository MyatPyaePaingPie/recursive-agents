# RLM Setup Optimized for Your Hardware

## Your Specs: i7 + RTX 3060 Ti 8GB + Windows 11

This guide is specifically tailored for your hardware configuration to get maximum performance running **local LLMs for free**.

---

## 🎯 Perfect Models for Your 8GB VRAM

### Tier 1: Best Overall (Pick One as Primary)

#### **Mistral 7B v0.3** ⭐⭐⭐
```powershell
ollama pull mistral:7b
```
- **VRAM**: 5.4GB (Q5_K_M)
- **Speed**: 40-50 tokens/second on RTX 3060 Ti
- **Context**: 32K tokens
- **Quality**: Best 7B model available
- **Use for**: Production RLM, code generation, general tasks

#### **Phi-3.5 Mini 3.8B** ⭐⭐⭐
```powershell
ollama pull phi3.5:3.8b
```
- **VRAM**: 2.8GB (Q6_K)
- **Speed**: 60-80 tokens/second
- **Context**: 128K tokens (PERFECT for recursion tracking!)
- **Quality**: Exceptional reasoning for size
- **Use for**: Long context, fast iteration, recursion experiments

### Tier 2: Specialists

#### **Qwen 2.5 7B** (Best Reasoning)
```powershell
ollama pull qwen2.5:7b
```
- **VRAM**: 5.4GB
- **Speed**: 45-55 tok/s
- **Strength**: Complex reasoning, instruction following

#### **TinyLlama 1.1B** (Development/Testing)
```powershell
ollama pull tinyllama:1.1b
```
- **VRAM**: 0.8GB
- **Speed**: 100+ tok/s
- **Use for**: Testing RLM logic, instant feedback

#### **Gemma 2 9B** (Google Quality)
```powershell
ollama pull gemma2:9b
```
- **VRAM**: 6GB (Q5_K_M)
- **Speed**: 35-45 tok/s
- **Strength**: Google's quality, good reasoning

---

## 🚀 5-Minute Installation

### Step 1: Install Ollama
```powershell
winget install Ollama.Ollama
```

### Step 2: Download Your Primary Model
```powershell
# Recommended: Mistral 7B (best overall)
ollama pull mistral:7b

# Also get TinyLlama for fast testing
ollama pull tinyllama:1.1b
```

### Step 3: Test It
```powershell
ollama run mistral:7b "Write a recursive Python function"
```

If you see output, **you're done!** Ollama runs automatically on Windows.

---

## 💻 Performance Expectations on RTX 3060 Ti

### Mistral 7B (Q5_K_M)
- **Load time**: 2-5 seconds (first run)
- **First token latency**: 0.5-1 second
- **Generation speed**: 40-50 tokens/second
- **VRAM usage**: 5.4GB / 8GB
- **Remaining VRAM**: 2.6GB for system
- **Quality**: ⭐⭐⭐⭐⭐

### Phi-3.5 Mini (Q6_K)
- **Load time**: 1-3 seconds
- **First token latency**: 0.3-0.5 seconds
- **Generation speed**: 60-80 tokens/second
- **VRAM usage**: 2.8GB / 8GB
- **Remaining VRAM**: 5.2GB (can run other apps!)
- **Quality**: ⭐⭐⭐⭐

### TinyLlama 1.1B (Q6_K)
- **Load time**: <1 second
- **First token latency**: 0.1-0.2 seconds
- **Generation speed**: 100+ tokens/second
- **VRAM usage**: 0.8GB / 8GB
- **Remaining VRAM**: 7.2GB
- **Quality**: ⭐⭐⭐ (good for size)

---

## 📊 VRAM Management Strategy

### Single Model Setup (Simplest)
```
Mistral 7B:     5.4GB
System:         1.5GB
Windows:        1.0GB
Total:          7.9GB / 8GB ✅
```

### Dual Model Setup (Development)
```
TinyLlama:      0.8GB (testing)
Mistral 7B:     [load when needed]
System:         1.5GB
Windows:        1.0GB
Total:          3.3GB / 8GB ✅ (plenty of headroom)
```

### Aggressive Setup (Maximum Quality)
```
Gemma 2 9B:     6.0GB
System:         1.0GB
Windows:        1.0GB
Total:          8.0GB / 8GB ⚠️ (tight but works)
```

---

## 🎮 GPU Optimization Tips

### 1. GPU Layer Offloading
Ollama automatically uses optimal GPU layers, but you can control it:

```bash
# Check Ollama logs to see GPU usage
ollama ps

# In llama.cpp (if you switch to it)
-ngl 35  # Load 35 layers on GPU (full for 7B)
-ngl 25  # Partial GPU (saves VRAM)
```

### 2. Context Size Management
```python
# Smaller context = less VRAM
n_ctx=2048   # +0.5GB VRAM
n_ctx=4096   # +1GB VRAM  ← Recommended for RLM
n_ctx=8192   # +2GB VRAM
n_ctx=16384  # +4GB VRAM (only with Phi-3.5!)
```

### 3. Batch Size Tuning
```python
# Larger batch = faster but more VRAM
n_batch=512   # Default, good balance
n_batch=1024  # Faster, +0.5GB VRAM
n_batch=256   # Saves VRAM if needed
```

### 4. Close VRAM Hogs
Before running RLM:
- ✅ Close Chrome (uses 1-2GB VRAM)
- ✅ Close games
- ✅ Close video editing apps
- ✅ Check Task Manager → Performance → GPU

---

## 💰 Cost Analysis: Local vs Cloud

### Your Local Setup (1000 RLM queries)

**Hardware Cost**: Already have RTX 3060 Ti
**Model Cost**: $0 (open source)
**Electricity**: 200W system × 10 hours × $0.15/kWh = **$3**
**Privacy**: 100% private
**Speed**: 40-50 tok/s
**Total**: **$3**

### OpenAI GPT-4 (1000 queries)

Average RLM query: 10,000 tokens (recursive)
Cost: $0.03/1K input + $0.06/1K output
**Total**: ~**$450**

### Groq Cloud (Free Tier)

30 requests/min limit
Free tier: 14,400 requests/day
**Total**: **$0** (within limits)
Speed: 300+ tok/s (faster than local!)

### Your Savings

Local vs GPT-4: **Save $447 per 1000 queries**
Local + Groq hybrid: **Best of both worlds**

---

## 🏆 Recommended Setup for RLM Project

### Strategy: Hybrid Approach

**Use Ollama (Mistral 7B) for:**
- ✅ Code generation (sensitive data)
- ✅ Recursion planning
- ✅ Privacy-sensitive operations
- ✅ When offline
- ✅ Cost: $0

**Use Groq (free tier) for:**
- ✅ Simple recursive calls (fast)
- ✅ Final result aggregation (speed matters)
- ✅ Testing at scale
- ✅ Cost: $0 (within limits)

**Use TinyLlama for:**
- ✅ Development/debugging
- ✅ Unit tests
- ✅ Rapid iteration
- ✅ Cost: $0

### Configuration in .env

```bash
# Primary: Local Ollama
USE_LOCAL_MODEL=true
OLLAMA_MODEL=mistral:7b

# Fallback: Groq (free)
GROQ_API_KEY=your-free-key
GROQ_MODEL=llama-3.1-8b-instant

# Development: TinyLlama
DEV_MODEL=tinyllama:1.1b
```

---

## 🔧 Troubleshooting Your Hardware

### Issue: "CUDA out of memory"

**Symptoms**: Model crashes, error message
**Solutions**:
1. Close Chrome/Discord/other apps
2. Use smaller model: `phi3.5:3.8b` instead of `mistral:7b`
3. Reduce context: `n_ctx=2048`
4. Restart Ollama: `ollama serve`

### Issue: "Slow generation speed"

**Symptoms**: <30 tok/s on 7B model
**Solutions**:
1. Update GPU drivers
2. Check GPU usage in Task Manager (should be 90%+)
3. Ensure using CUDA version: `ollama show mistral:7b`
4. Try smaller model to isolate issue

### Issue: "Model not found"

**Symptoms**: 404 error
**Solutions**:
```powershell
# List downloaded models
ollama list

# Pull the model
ollama pull mistral:7b

# Verify
ollama run mistral:7b "test"
```

### Issue: "Ollama not running"

**Symptoms**: Connection refused
**Solutions**:
```powershell
# Check if running
ollama ps

# Start service
ollama serve

# On Windows, it usually auto-starts
# Check Windows Services for "Ollama"
```

---

## 📈 Scaling Strategy

### Phase 1: Development (Now)
- **Model**: TinyLlama 1.1B
- **VRAM**: 0.8GB
- **Speed**: 100+ tok/s
- **Purpose**: Fast iteration, test RLM logic

### Phase 2: Testing (After core implementation)
- **Model**: Mistral 7B
- **VRAM**: 5.4GB
- **Speed**: 40-50 tok/s
- **Purpose**: Validate quality, benchmark

### Phase 3: Production (Final)
- **Primary**: Mistral 7B (local)
- **Fallback**: Groq Llama 3.1 8B (cloud, free)
- **Cost**: ~$0
- **Privacy**: Hybrid (sensitive stays local)

---

## 🎯 Next Steps for Your Hardware

### Immediate (5 minutes)
1. Install Ollama: `winget install Ollama.Ollama`
2. Pull Mistral: `ollama pull mistral:7b`
3. Test: `ollama run mistral:7b "Hello"`

### Short-term (1 hour)
4. Set up Python environment (see QUICKSTART_LOCAL.md)
5. Configure .env for local models
6. Implement OllamaLLM class (see IMPLEMENTATION_GUIDE.md)

### Medium-term (1 week)
7. Build RLM core with local models
8. Test with TinyLlama (fast iteration)
9. Validate with Mistral (quality checks)
10. Benchmark local vs cloud

---

## 🎓 Learning Resources

### Understanding Your GPU
- VRAM vs RAM: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/
- CUDA cores: Your 3060 Ti has 4864 cores
- Memory bandwidth: 448 GB/s

### Model Quantization
- Q5_K_M explained: 5-bit with medium quality
- Q4 vs Q5 vs Q6: Trade quality for VRAM
- GGUF format: Optimized for inference

### Ollama Internals
- How Ollama works: https://ollama.com/blog
- Model library: https://ollama.com/library
- API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

---

## 📞 Getting Help

### If something doesn't work:

1. **Check Ollama status**: `ollama ps`
2. **View logs**: Check Windows Event Viewer → Ollama
3. **Test GPU**: `nvidia-smi` in PowerShell
4. **Verify drivers**: GeForce Experience → Drivers
5. **Ask in Discord**: Ollama has active community

### Common Windows 11 Issues:

**Firewall blocking**: Allow Ollama in Windows Defender
**Antivirus interfering**: Add Ollama to exclusions
**WSL confusion**: Ollama runs natively, not in WSL

---

## 🏁 Success Checklist

Before starting RLM implementation:
- [ ] Ollama installed and running
- [ ] Mistral 7B downloaded (~4GB)
- [ ] TinyLlama downloaded (~0.6GB)
- [ ] Test successful: `ollama run mistral:7b "test"`
- [ ] Python environment set up
- [ ] GPU drivers updated (latest)
- [ ] VRAM usage understood (<6GB for Mistral)
- [ ] .env configured for local models

**When all checked, you're ready to build! Start with Phase 1 in IMPLEMENTATION_GUIDE.md**

---

Your RTX 3060 Ti is perfect for running local LLMs. You'll get excellent performance at zero cost! 🚀

**Next**: Read `QUICKSTART_LOCAL.md` for detailed setup, then `IMPLEMENTATION_GUIDE.md` to build your RLM system.


