# Local Models Guide for RTX 3060 Ti (8GB VRAM)

## Why Local Models?

Running models locally on your RTX 3060 Ti gives you:
- **Zero API costs** - No per-token charges
- **Privacy** - Your data never leaves your machine
- **Speed** - No network latency (10-50 tokens/second)
- **Offline capability** - Work anywhere
- **Learning** - Understand LLM internals

## Your Hardware Profile

**Specs:**
- CPU: Intel i7 (Good for CPU offloading)
- GPU: RTX 3060 Ti (8GB VRAM) ⭐
- OS: Windows 11
- RAM: (Assuming 16GB+)

**What you can run:**
- ✅ 7B models at Q5/Q6 quantization (excellent quality)
- ✅ 13-14B models at Q4 quantization (very good quality)
- ✅ 3B models at Q8 quantization (maximum quality for small models)
- ⚠️ 70B+ models with heavy CPU offloading (slow but possible)

## Recommended Lightweight Models for Your Use Case

### Top 5 Models for RLM Project

#### 1. **Mistral 7B v0.3** (BEST OVERALL) ⭐⭐⭐
```
Model: Mistral-7B-Instruct-v0.3-Q5_K_M
VRAM: ~5.4GB
Context: 32K tokens
Speed: 40-50 tok/s
Quality: Excellent
```

**Why it's perfect:**
- Best 7B model available
- Great code generation
- 32K context (good for recursion)
- Fits comfortably in VRAM

**Download:**
```bash
# Via Ollama (easiest)
ollama pull mistral:7b

# Via LM Studio
# Search: bartowski/Mistral-7B-Instruct-v0.3-GGUF
# Download: Q5_K_M variant
```

#### 2. **Phi-3.5 Mini 3.8B** (MOST EFFICIENT) ⭐⭐⭐
```
Model: Phi-3.5-mini-instruct-Q6_K
VRAM: ~2.8GB
Context: 128K tokens (!!)
Speed: 60-80 tok/s
Quality: Punches above weight class
```

**Why it's perfect:**
- 128K context = perfect for RLM recursion
- Incredibly fast
- Leaves VRAM for other processes
- Microsoft's reasoning-focused training

**Download:**
```bash
# Via Ollama
ollama pull phi3.5:3.8b

# Via LM Studio
# Search: bartowski/Phi-3.5-mini-instruct-GGUF
```

#### 3. **Qwen2.5 7B** (MULTILINGUAL BEAST) ⭐⭐
```
Model: Qwen2.5-7B-Instruct-Q5_K_M
VRAM: ~5.4GB
Context: 32K tokens
Speed: 45-55 tok/s
Quality: Excellent reasoning
```

**Why it's perfect:**
- Strong reasoning and code
- Good instruction following
- Active development
- Great for complex tasks

**Download:**
```bash
# Via Ollama
ollama pull qwen2.5:7b

# Via LM Studio
# Search: Qwen/Qwen2.5-7B-Instruct-GGUF
```

#### 4. **TinyLlama 1.1B** (TESTING/DEVELOPMENT) ⭐
```
Model: TinyLlama-1.1B-Chat-Q6_K
VRAM: ~0.8GB
Context: 2K tokens
Speed: 100+ tok/s
Quality: Good for its size
```

**Why it's perfect:**
- Test your RLM system without burning VRAM
- Nearly instant responses
- Great for development/debugging
- Trained on 3 trillion tokens

**Download:**
```bash
# Via Ollama
ollama pull tinyllama:1.1b

# Via LM Studio
# Search: TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF
```

#### 5. **Gemma 2 9B** (GOOGLE QUALITY) ⭐⭐
```
Model: gemma-2-9b-it-Q5_K_M
VRAM: ~6GB
Context: 8K tokens
Speed: 35-45 tok/s
Quality: Excellent
```

**Why it's perfect:**
- Google's open source gem
- Exceptional quality
- Good instruction following
- Well-documented

**Download:**
```bash
# Via Ollama
ollama pull gemma2:9b

# Via LM Studio
# Search: bartowski/gemma-2-9b-it-GGUF
```

## Installation Options

### Option 1: Ollama (Recommended for Beginners) ⭐

**Pros:**
- One command to install
- Automatic GPU detection
- Simple model management
- Great CLI and API

**Installation:**
```powershell
# Install Ollama
winget install Ollama.Ollama

# Pull a model
ollama pull mistral:7b

# Test it
ollama run mistral:7b "Write a recursive function in Python"

# Run as API server (for RLM integration)
ollama serve
```

**Integration with RLM:**
```python
# Ollama exposes OpenAI-compatible API
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Not used but required
)

response = client.chat.completions.create(
    model="mistral:7b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Option 2: LM Studio (Best UI Experience) ⭐⭐

**Pros:**
- Beautiful interface
- One-click model downloads
- Model discovery built-in
- Parameter tuning UI
- Chat interface for testing

**Installation:**
1. Download from https://lmstudio.ai/
2. Install and launch
3. Click "Search" → Search "Mistral 7B"
4. Download Q5_K_M variant
5. Click "Local Server" → Start server
6. Server runs on `http://localhost:1234`

**Integration with RLM:**
```python
# LM Studio also exposes OpenAI-compatible API
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="local-model",  # Use whatever you loaded
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Option 3: llama.cpp (Maximum Control) ⭐⭐⭐

**Pros:**
- Direct model loading
- Full parameter control
- No server overhead
- Best performance
- Python bindings available

**Installation:**
```powershell
# Install Python bindings
pip install llama-cpp-python

# Or build from source for GPU support:
$env:CMAKE_ARGS="-DLLAMA_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Download Models:**
Visit Hugging Face and search for GGUF files:
```
https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF
```
Download the `Q5_K_M.gguf` file to `models/` folder.

**Integration with RLM:**
```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.3.Q5_K_M.gguf",
    n_ctx=4096,  # Context size
    n_gpu_layers=35,  # Offload to GPU (try different values)
    verbose=False
)

# Generate
response = llm(
    "Write a recursive function",
    max_tokens=512,
    temperature=0.7,
    stop=["</s>"]
)

print(response['choices'][0]['text'])
```

### Option 4: Groq API (Cloud Alternative - Free Tier!) ⭐⭐

**Pros:**
- Blazingly fast (300+ tok/s)
- Free tier: 30 requests/min
- No local resources used
- Hosts Llama, Mixtral, Gemma

**Installation:**
```bash
pip install groq
```

**Get API Key:**
1. Visit https://console.groq.com/
2. Sign up (free)
3. Generate API key

**Integration with RLM:**
```python
from groq import Groq

client = Groq(api_key="your-key")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",  # Super fast
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

**Available Models:**
- `llama-3.1-8b-instant` (Fast, recommended)
- `llama-3.1-70b-versatile` (Powerful)
- `mixtral-8x7b-32768` (Long context)
- `gemma2-9b-it` (Google quality)

## Quantization Guide

**What is quantization?**
Reducing model precision to save VRAM while maintaining quality.

**Quantization Levels (Best to Worst):**
- **Q8_0**: 8-bit, ~99% original quality, largest
- **Q6_K**: 6-bit, ~98% quality, excellent
- **Q5_K_M**: 5-bit medium, ~97% quality, **RECOMMENDED**
- **Q5_K_S**: 5-bit small, ~96% quality, slightly smaller
- **Q4_K_M**: 4-bit medium, ~95% quality, good for 13B models
- **Q4_K_S**: 4-bit small, ~93% quality
- **Q3_K_M**: 3-bit, ~90% quality, only for large models
- **Q2_K**: 2-bit, ~80% quality, experimental

**For RTX 3060 Ti:**
| Model Size | Recommended Quant | VRAM Usage |
|------------|------------------|------------|
| 1-3B | Q8_0 or Q6_K | 1-2.5GB |
| 7-8B | Q5_K_M | 5-6GB |
| 13-14B | Q4_K_M | 7-9GB |

## Performance Optimization

### GPU Layer Offloading

The `-ngl` (number of GPU layers) parameter controls how much runs on GPU vs CPU.

**Finding optimal value:**
```bash
# Start high and reduce if you get OOM errors
ollama run mistral:7b

# For llama.cpp, experiment:
# Full GPU: -ngl 35 (7B model)
# Partial: -ngl 20 (uses CPU + GPU)
# CPU only: -ngl 0
```

**Rule of thumb:**
- 7B model: Try `-ngl 32-35`
- 13B model: Try `-ngl 25-30`
- If you get "out of memory", reduce by 5

### Context Size vs VRAM

Larger context = more VRAM:
```
2K context: +0.5GB VRAM
4K context: +1GB VRAM
8K context: +2GB VRAM
16K context: +4GB VRAM
```

**For RLM (which needs context):**
- Use 4K-8K context for recursion depth tracking
- Don't use 128K unless you're testing specific features
- Phi-3.5's 128K is great but you'll use ~6-7GB total

### Batch Processing

```python
# Process multiple recursive calls in parallel
llm = Llama(
    model_path="...",
    n_batch=512,  # Batch size (higher = faster but more VRAM)
    n_threads=8   # CPU threads (match your i7 cores)
)
```

## Model Comparison for RLM Project

| Model | Size | VRAM | Speed | Code Gen | Reasoning | Context | Best For |
|-------|------|------|-------|----------|-----------|---------|----------|
| Mistral 7B | 7B | 5.4GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 32K | Overall best |
| Phi-3.5 Mini | 3.8B | 2.8GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128K | Speed + long context |
| Qwen2.5 7B | 7B | 5.4GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 32K | Complex reasoning |
| Gemma 2 9B | 9B | 6GB | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8K | Google quality |
| TinyLlama | 1.1B | 0.8GB | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 2K | Testing/dev |

## RLM-Specific Considerations

### Best Model for Code Generation
**Winner: Mistral 7B v0.3**
- Excellent Python code generation
- Good at generating recursive logic
- Understands function structure well

### Best Model for Recursion Planning
**Winner: Phi-3.5 Mini 3.8B**
- Strong reasoning despite size
- 128K context helps track recursion state
- Microsoft trained it on synthetic reasoning data

### Best Model for Testing
**Winner: TinyLlama 1.1B**
- Nearly instant responses (100+ tok/s)
- Good enough to test your RLM logic
- Won't blow your VRAM budget

### Best Model for Production
**Winner: Qwen2.5 7B**
- Excellent balance of quality and speed
- Good at following complex instructions
- Strong reasoning for chunking decisions

## Recommended Setup for RLM Project

### Development Setup (Best for testing)
```bash
# Install Ollama
winget install Ollama.Ollama

# Pull development model
ollama pull tinyllama:1.1b

# Pull production model
ollama pull mistral:7b

# Start server
ollama serve
```

### Production Setup (Best performance)
```bash
# Install llama-cpp-python with CUDA
$env:CMAKE_ARGS="-DLLAMA_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Download GGUF from Hugging Face:
# https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF
# Save to: models/mistral-7b-instruct-v0.3.Q5_K_M.gguf
```

### Hybrid Setup (Best flexibility)
```bash
# Use Groq for fast cloud inference
pip install groq

# Use Ollama for local privacy-sensitive tasks
ollama pull phi3.5:3.8b

# Switch based on use case:
# - Code generation: Groq (fast)
# - Sensitive data: Ollama (private)
# - Development: TinyLlama (instant)
```

## Sample RLM Configuration

### config.py
```python
from pydantic_settings import BaseSettings
from typing import Optional

class RLMConfig(BaseSettings):
    # Model selection
    use_local_model: bool = True
    model_backend: str = "ollama"  # Options: ollama, llama-cpp, groq, openai
    
    # Ollama config
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"
    
    # llama.cpp config
    local_model_path: str = "models/mistral-7b-instruct-v0.3.Q5_K_M.gguf"
    n_gpu_layers: int = 35
    n_ctx: int = 4096
    
    # Groq config
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    
    # OpenAI config (fallback)
    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
```

## Troubleshooting

### "CUDA out of memory"
**Solutions:**
1. Reduce `-ngl` (GPU layers): Try `-ngl 25` instead of `35`
2. Use smaller quantization: Q4_K_M instead of Q5_K_M
3. Reduce context: `-c 2048` instead of `4096`
4. Close other GPU applications
5. Switch to smaller model: Try 3.8B instead of 7B

### "Model loading is slow"
**Solutions:**
1. Use SSD for model storage (not HDD)
2. Use mmap: `use_mmap=True` in llama-cpp
3. Preload model at startup, not per-request
4. Consider keeping model in VRAM between calls

### "Responses are slow"
**Solutions:**
1. Increase GPU layers: Try `-ngl 35`
2. Check GPU utilization: Should be 90%+
3. Use smaller quantization if CPU-bound
4. Enable flash attention if available
5. Reduce context size

### "Model gives poor results"
**Solutions:**
1. Use higher quantization: Q5_K_M or Q6_K
2. Improve your prompts (more specific)
3. Try different model (Qwen vs Mistral)
4. Increase temperature for creativity (0.8-1.0)
5. Provide few-shot examples

## Cost Comparison

### Local Model Costs
- **Electricity**: ~$0.03/hour (200W system, $0.15/kWh)
- **One-time**: Free (open source models)
- **Per token**: $0.00

### API Costs (for comparison)
- **OpenAI GPT-4**: $0.03/1K tokens input, $0.06/1K output
- **Claude Opus**: $0.015/1K input, $0.075/1K output
- **Groq** (free tier): $0.00 (30 req/min limit)

**RLM Project Estimate:**
- Average recursion: 5 depth levels
- Tokens per level: 2000
- Total per query: 10,000 tokens

**Costs per 100 queries:**
- Local (Mistral 7B): $0.30 electricity
- GPT-4: $45
- Groq: $0 (if within limits)

## Next Steps

1. **Install Ollama**: `winget install Ollama.Ollama`
2. **Pull Mistral**: `ollama pull mistral:7b`
3. **Test it**: `ollama run mistral:7b "Write a hello world"`
4. **Integrate with RLM**: See `src/rlm/models/ollama_llm.py` (you'll create this)
5. **Benchmark**: Compare local vs cloud for your use case

## Resources

- **Ollama**: https://ollama.com/
- **LM Studio**: https://lmstudio.ai/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Groq**: https://console.groq.com/
- **Hugging Face GGUF Models**: https://huggingface.co/models?search=gguf
- **Quantization Guide**: https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md

Ready to start? I recommend beginning with **Ollama + Mistral 7B** - it's the easiest and most capable combo for your hardware! 🚀


