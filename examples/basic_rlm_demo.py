#!/usr/bin/env python3
"""Basic demonstration of the Recursive Language Model system.

This example shows how to use the RLM system to process large contexts
that exceed typical LLM context windows.

Requirements:
    - Ollama running locally (default) OR
    - GROQ_API_KEY environment variable OR
    - OPENAI_API_KEY environment variable

Usage:
    python examples/basic_rlm_demo.py

    # With specific provider
    RLM_LLM_PROVIDER=groq python examples/basic_rlm_demo.py
"""

import asyncio
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlm import RecursiveInferenceEngine, RLMConfig
from rlm.models import create_llm
from rlm.utils.logging import setup_logging


# Sample long text for demonstration
SAMPLE_TEXT = """
# Introduction to Recursive Language Models

Recursive Language Models (RLMs) represent a paradigm shift in how we approach
large-scale text processing with artificial intelligence. Traditional language
models are constrained by fixed context windows, typically ranging from a few
thousand to a few hundred thousand tokens. This limitation poses significant
challenges when processing documents that exceed these boundaries.

## The Problem with Fixed Context Windows

Consider the task of summarizing a technical book or analyzing a large codebase.
A typical novel contains approximately 80,000-100,000 words, which translates to
roughly 100,000-130,000 tokens. While some modern models like Claude 3 support
up to 200,000 tokens, many real-world documents still exceed these limits.

### Challenges Include:
1. Loss of context when truncating documents
2. Inability to maintain coherence across chunks
3. Missing important information at the boundaries
4. Difficulty in cross-referencing sections

## The RLM Solution

The core insight of RLMs is elegant: instead of trying to fit all context into
the model's window, treat the prompt as an external data structure that the model
can programmatically access.

### How It Works

1. **Context Externalization**: The full document is stored outside the LLM's
   immediate context window, accessible through API functions.

2. **Code Generation**: The LLM generates Python code to examine the context
   structure and decide how to process it.

3. **Recursive Decomposition**: Large documents are broken into manageable chunks,
   each processed recursively.

4. **Safe Execution**: Generated code runs in a sandboxed environment with
   restricted permissions.

5. **Result Aggregation**: Results from recursive calls are combined into a
   coherent final answer.

## Benefits of the Approach

### Unlimited Context Processing
RLMs can handle documents of any size, limited only by computational budget
rather than architectural constraints.

### Efficient Token Usage
By processing only relevant sections, RLMs often use fewer tokens than naive
chunking approaches that process everything.

### Interpretable Processing
The generated code provides a clear trace of how the document was processed,
making the system more debuggable and trustworthy.

### Flexible Decomposition
The LLM can adapt its processing strategy based on document structure,
using different approaches for code, prose, or structured data.

## Implementation Considerations

### Security
Since RLMs execute generated code, robust sandboxing is essential. The system
uses RestrictedPython with whitelisted operations and strict resource limits.

### Cost Management
Recursive calls can accumulate costs. The system implements depth limits and
caching to control expenses.

### Quality vs Speed
There's a tradeoff between processing quality and speed. Deeper recursion
typically yields better results but takes longer.

## Conclusion

Recursive Language Models open new possibilities for processing large documents
with AI. By combining the reasoning capabilities of modern LLMs with the
precision of programmatic access, RLMs overcome the fundamental limitation
of fixed context windows while maintaining high-quality output.

The future may see RLMs integrated into document analysis tools, code review
systems, and research assistants - anywhere that large-scale text understanding
is needed.
""" * 5  # Repeat to make it larger


async def main() -> None:
    """Run the basic RLM demonstration."""
    # Setup logging
    setup_logging(level="INFO")

    print("=" * 60)
    print("Recursive Language Models - Basic Demo")
    print("=" * 60)

    # Load configuration
    config = RLMConfig()

    # Determine which provider to use
    provider = config.llm.provider
    api_key = config.get_api_key(provider)

    print(f"\nConfiguration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {config.llm.model}")
    print(f"  Max recursion depth: {config.max_recursion_depth}")
    print(f"  Chunk size: {config.default_chunk_size} tokens")

    # Check provider availability
    if provider == "ollama":
        print("\n  Using Ollama (local). Make sure 'ollama serve' is running.")
    elif provider in ("groq", "openai", "anthropic") and not api_key:
        print(f"\n  WARNING: {provider.upper()}_API_KEY not set!")
        print("  Falling back to Ollama...")
        provider = "ollama"
        api_key = None

    # Create LLM
    try:
        llm = create_llm(
            provider=provider,
            api_key=api_key,
            model=config.llm.model,
        )
        print(f"\n  Created LLM: {llm}")
    except Exception as e:
        print(f"\n  ERROR: Failed to create LLM: {e}")
        print("\n  Please ensure:")
        print("    - Ollama is running (ollama serve)")
        print("    - Or set GROQ_API_KEY/OPENAI_API_KEY environment variable")
        return

    # Create engine
    engine = RecursiveInferenceEngine(llm=llm, config=config)

    # Process the sample text
    query = "What are the main benefits and challenges of Recursive Language Models?"

    print(f"\nInput:")
    print(f"  Context length: {len(SAMPLE_TEXT)} characters")
    print(f"  Query: {query}")
    print("\nProcessing...")

    try:
        result = await engine.process(
            query=query,
            context=SAMPLE_TEXT,
        )

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"\nAnswer:\n{result.answer}")

        print("\n" + "-" * 60)
        print("Processing Statistics:")
        print(f"  Total tokens: {result.total_tokens}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        print(f"  Recursive calls: {result.num_recursive_calls}")
        print(f"  Max depth reached: {result.max_depth_reached}")

        if result.metadata:
            print(f"\nMetadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_with_different_queries() -> None:
    """Demonstrate different types of queries."""
    setup_logging(level="WARNING")

    config = RLMConfig()
    llm = create_llm(provider="ollama", model="mistral:7b")
    engine = RecursiveInferenceEngine(llm=llm, config=config)

    queries = [
        "Summarize this document in one paragraph.",
        "What are the security considerations mentioned?",
        "List the main components of an RLM system.",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        try:
            result = await engine.process(query=query, context=SAMPLE_TEXT)
            print(f"Answer: {result.answer[:300]}...")
            print(f"(Tokens: {result.total_tokens}, Time: {result.execution_time:.1f}s)")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run the basic demo
    asyncio.run(main())
