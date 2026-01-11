# Quick Start: RLM with Local Models (RTX 3060 Ti)

## 5-Minute Setup for Your Hardware

Your specs: **i7 + RTX 3060 Ti 8GB + Windows 11**

This guide gets you running with **zero API costs** using local models.

---

## Step 1: Install Ollama (2 minutes)

Ollama is the easiest way to run local models on Windows.

```powershell
# Install Ollama
winget install Ollama.Ollama

# Verify installation
ollama --version
```

That's it! Ollama is installed and running.

---

## Step 2: Download a Model (2 minutes)

Download Mistral 7B (best overall model for your VRAM):

```powershell
# Pull Mistral 7B (~4GB download, ~5.4GB VRAM)
ollama pull mistral:7b
```

**Alternative models** (pick one or all):

```powershell
# Phi-3.5 Mini - Fastest, 128K context, 3.8B params
ollama pull phi3.5:3.8b

# TinyLlama - For testing/development, 1.1B params
ollama pull tinyllama:1.1b

# Qwen 2.5 - Best reasoning, 7B params
ollama pull qwen2.5:7b

# Gemma 2 - Google quality, 9B params
ollama pull gemma2:9b
```

---

## Step 3: Test Your Model (1 minute)

```powershell
# Chat with the model
ollama run mistral:7b "Write a recursive Python function to calculate factorial"

# Or interactive mode
ollama run mistral:7b
>>> Write a hello world in Python
>>> /bye
```

If you see a response, **you're done!** 🎉

---

## Step 4: Set Up Your RLM Project

### Create Project Structure

```powershell
# Navigate to your project
cd C:\Users\myatp\OneDrive\Desktop\ClaudeCodePlay\recursive-agents

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment

```powershell
# Copy environment template
copy env.example .env

# Edit .env (use Notepad or your editor)
notepad .env
```

**Set these values in `.env`:**
```bash
# Local Model Configuration
USE_LOCAL_MODEL=true
LOCAL_MODEL_TYPE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Optional: Groq (free, fast cloud alternative)
GROQ_API_KEY=your-groq-key-here  # Get from console.groq.com
GROQ_MODEL=llama-3.1-8b-instant

# Optional: OpenAI (if you want to compare)
OPENAI_API_KEY=your-key-here
```

---

## Step 5: Verify Ollama API (30 seconds)

Ollama exposes an OpenAI-compatible API automatically.

**Test it:**
```powershell
# In a new terminal
curl http://localhost:11434/api/tags
```

You should see your downloaded models listed.

---

## Step 6: Create Your First Local LLM Integration

Create `src/rlm/models/ollama_llm.py`:

```python
"""Ollama local model integration."""

from typing import Optional
import openai
from .base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    """LLM implementation using Ollama local models."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "mistral:7b"
    ):
        """Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            model: Model name (e.g., mistral:7b, phi3.5:3.8b)
        """
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key="ollama"  # Required but not used
        )
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate completion using local model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with generated content
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=self.model,
            finish_reason=response.choices[0].finish_reason
        )
    
    async def generate_code(self, task_description: str) -> str:
        """Generate Python code for task.
        
        Args:
            task_description: Description of coding task
            
        Returns:
            Generated Python code
        """
        system_prompt = """You are an expert Python programmer.
Generate clean, efficient Python code for the given task.
Only output the code, no explanations."""
        
        response = await self.generate(
            prompt=task_description,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temp for code
            max_tokens=1500
        )
        
        return response.content
```

---

## Step 7: Test Your Integration

Create a test script `test_local.py`:

```python
"""Test local model integration."""

import asyncio
from src.rlm.models.ollama_llm import OllamaLLM


async def main():
    # Create client
    llm = OllamaLLM(model="mistral:7b")
    
    # Test simple generation
    print("Testing simple generation...")
    response = await llm.generate("What is recursion in programming?")
    print(f"\nResponse: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    
    # Test code generation
    print("\n" + "="*60)
    print("Testing code generation...")
    code = await llm.generate_code(
        "Write a recursive function to calculate Fibonacci numbers"
    )
    print(f"\nGenerated code:\n{code}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```powershell
python test_local.py
```

You should see responses from your local model! 🚀

---

## Performance Expectations on Your Hardware

### Mistral 7B (Q5_K_M)
- **Load time**: 2-5 seconds
- **First token**: 0.5-1 second
- **Speed**: 40-50 tokens/second
- **VRAM usage**: ~5.4GB
- **Quality**: Excellent

### Phi-3.5 Mini 3.8B
- **Load time**: 1-3 seconds
- **First token**: 0.3-0.5 seconds
- **Speed**: 60-80 tokens/second
- **VRAM usage**: ~2.8GB
- **Quality**: Very good for size

### TinyLlama 1.1B
- **Load time**: <1 second
- **First token**: 0.1-0.2 seconds
- **Speed**: 100+ tokens/second
- **VRAM usage**: ~0.8GB
- **Quality**: Good for testing

---

## Recommended Model Selection for RLM

### For Development/Testing
**Use: TinyLlama 1.1B**
```bash
ollama pull tinyllama:1.1b
```
- Nearly instant responses
- Minimal VRAM usage
- Perfect for debugging RLM logic

### For Production/Quality
**Use: Mistral 7B**
```bash
ollama pull mistral:7b
```
- Best overall quality
- Good code generation
- 32K context window

### For Speed + Long Context
**Use: Phi-3.5 Mini 3.8B**
```bash
ollama pull phi3.5:3.8b
```
- 128K context (perfect for recursion tracking)
- Very fast
- Excellent reasoning

### For Maximum Quality (within 8GB)
**Use: Qwen 2.5 7B**
```bash
ollama pull qwen2.5:7b
```
- Best reasoning
- Strong instruction following
- Excellent code generation

---

## Switching Models

Just change the model name in your code or `.env`:

```python
# In code
llm = OllamaLLM(model="phi3.5:3.8b")  # Switch to Phi-3.5

# Or in .env
OLLAMA_MODEL=phi3.5:3.8b
```

No API key changes needed!

---

## Cost Comparison

### Your Local Setup (Mistral 7B)
- **Setup cost**: $0 (open source)
- **Per-token cost**: $0
- **Electricity**: ~$0.03/hour (200W system)
- **Privacy**: 100% (data never leaves your PC)

### OpenAI GPT-4
- **Per-token cost**: $0.03/1K input, $0.06/1K output
- **100 RLM queries**: ~$45
- **1000 queries**: ~$450

### Your Savings
If you run 1000 RLM queries:
- **Local**: $3 electricity
- **GPT-4**: $450
- **You save**: $447 💰

---

## Troubleshooting

### "Connection refused" error
**Solution**: Make sure Ollama is running
```powershell
# Restart Ollama
ollama serve
```

### "Out of memory" error
**Solutions**:
1. Use smaller model: `ollama pull phi3.5:3.8b`
2. Close other applications
3. Reduce context size in code

### "Model not found"
**Solution**: Pull the model first
```powershell
ollama pull mistral:7b
```

### Slow performance
**Solutions**:
1. Check GPU is being used: Task Manager → GPU
2. Update GPU drivers
3. Close Chrome/other GPU apps
4. Try smaller model for speed

---

## Next Steps

Now that your local model is running:

1. **Follow IMPLEMENTATION_GUIDE.md** - Build the RLM system
2. **Implement Phase 2** - Create the LLM abstractions
3. **Test with local models** - Use TinyLlama for fast iteration
4. **Switch to Mistral** - When you need quality
5. **Compare to Groq** - Try cloud speed (free tier)

---

## Advanced: Multiple Models Strategy

You can use different models for different purposes:

```python
# Fast model for code generation
code_llm = OllamaLLM(model="mistral:7b")

# Lightweight for testing
test_llm = OllamaLLM(model="tinyllama:1.1b")

# Long context for tracking recursion
recursion_llm = OllamaLLM(model="phi3.5:3.8b")
```

---

## Resources

- **Ollama Docs**: https://ollama.com/
- **Model Library**: https://ollama.com/library
- **Local Models Guide**: See `docs/LOCAL_MODELS_GUIDE.md`
- **RLM Implementation**: See `IMPLEMENTATION_GUIDE.md`

---

**You're now running state-of-the-art LLMs locally for free!** 🎉

No API keys, no costs, full privacy. Start building your RLM system! 🚀

