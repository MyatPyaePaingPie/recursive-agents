#!/usr/bin/env python3
"""Transparent demonstration of the Recursive Language Model system.

This example shows FULL VISIBILITY into every step of RLM processing:
- Context loading and chunking
- Code generation by lightweight model
- Code validation
- Sandbox execution
- Reasoning by heavyweight model
- Result aggregation

Requirements:
    - Ollama running locally (recommended for demo)
    - Or GROQ_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY

Usage:
    python examples/transparent_demo.py

    # With specific provider
    RLM_LLM_PROVIDER=groq python examples/transparent_demo.py

    # Quiet mode (just summary)
    python examples/transparent_demo.py --quiet
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlm import RLMConfig
from rlm.core.transparent import (
    Event,
    EventType,
    TransparentEngine,
    default_callback,
)
from rlm.models import create_llm
from rlm.models.tiers import ModelTier, ModelTiers, print_all_models


# Sample text for demonstration
SAMPLE_TEXT = """
# The Architecture of Recursive Language Models

## Chapter 1: Introduction

Language models have revolutionized natural language processing, but they face
a fundamental limitation: fixed context windows. Even the most advanced models
can only process a limited amount of text at once, typically between 4,000 and
200,000 tokens depending on the architecture.

### 1.1 The Context Window Problem

When processing large documents, traditional approaches include:

1. **Truncation**: Simply cut off text that doesn't fit
   - Pro: Simple to implement
   - Con: Loses potentially critical information

2. **Sliding Window**: Process overlapping chunks
   - Pro: Covers all content
   - Con: No global understanding, high token usage

3. **Summarization Chains**: Summarize chunks then summarize summaries
   - Pro: Reduces token usage
   - Con: Information loss at each level

### 1.2 A New Paradigm

Recursive Language Models (RLMs) take a fundamentally different approach:
instead of trying to fit context into the model, they let the model
programmatically access context as an external resource.

## Chapter 2: Core Architecture

The RLM system consists of three primary components:

### 2.1 The Python REPL Environment

The execution environment provides:
- `get_context_length()`: Total tokens in the external context
- `get_context_chunk(start, length)`: Retrieve a portion of context
- `call_submodel(query)`: Recursively invoke the LLM on a sub-problem
- `aggregate_results(results)`: Combine multiple results

### 2.2 The Root LLM Layer

The primary LLM generates Python code to:
1. Analyze the context structure
2. Decide on a processing strategy
3. Extract relevant information
4. Delegate to sub-models when needed

### 2.3 The Sub-Model Layer

Sub-models handle recursive calls. Importantly, the paper recommends
sub-models operate as standard LLMs (not RLMs themselves) with a maximum
recursion depth of 1.

## Chapter 3: Security Considerations

Since RLMs execute LLM-generated code, security is paramount:

### 3.1 Sandboxing

All code execution occurs in a restricted environment using:
- RestrictedPython for AST-level restrictions
- Whitelisted builtins only
- No file system access
- No network access
- Resource limits (CPU, memory, time)

### 3.2 Code Validation

Before execution, generated code is validated for:
- Forbidden operations (eval, exec, import)
- Dangerous patterns (file access, subprocess)
- Infinite loops (static analysis)
- Resource consumption estimates

## Chapter 4: Performance Optimization

### 4.1 Model Tiering

RLMs can use different models for different tasks:
- **Lightweight models** for code generation (fast, cheap)
- **Heavyweight models** for reasoning (accurate)

### 4.2 Caching Strategies

Common optimization techniques:
- Response caching for repeated queries
- Chunk caching for frequently accessed segments
- Result memoization across recursive calls

## Chapter 5: Implementation Example

Here's how a typical RLM processes a large document:

```python
# 1. Load context externally
context_manager.load_context(large_document)

# 2. LLM generates processing code
code = llm.generate("Write code to analyze this document...")

# 3. Code is validated
if validator.validate(code):
    # 4. Execute in sandbox
    result = sandbox.execute(code)
```

## Conclusion

Recursive Language Models represent a significant advancement in handling
large-scale text processing. By treating context as an external resource
and leveraging code generation, RLMs overcome the fundamental limitation
of fixed context windows while maintaining high-quality output.

The combination of lightweight code generation and heavyweight reasoning
enables cost-effective processing of documents that would otherwise be
impossible to handle with traditional approaches.
""" * 3  # Repeat to make larger


class VerboseCallback:
    """Callback that provides maximum visibility with structured output."""

    def __init__(self, show_full_output: bool = True):
        self.show_full_output = show_full_output
        self.event_count = 0
        self.start_time = datetime.now()

    def __call__(self, event: Event) -> None:
        """Handle an event with verbose output."""
        self.event_count += 1
        indent = "│   " * event.depth
        time_str = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        elapsed = (event.timestamp - self.start_time).total_seconds()

        # Color codes
        colors = {
            "green": "\033[32m",
            "bold_green": "\033[1;32m",
            "blue": "\033[34m",
            "bold_blue": "\033[1;34m",
            "yellow": "\033[33m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "red": "\033[31m",
            "bold_red": "\033[1;31m",
            "reset": "\033[0m",
            "dim": "\033[2m",
            "bold": "\033[1m",
        }

        def c(color: str, text: str) -> str:
            return f"{colors[color]}{text}{colors['reset']}"

        # Event type to color mapping
        type_colors = {
            EventType.PROCESS_START: "bold_green",
            EventType.PROCESS_END: "bold_green",
            EventType.CONTEXT_LOADING: "blue",
            EventType.CONTEXT_LOADED: "blue",
            EventType.CONTEXT_CHUNKED: "blue",
            EventType.LLM_PROMPT_PREPARED: "yellow",
            EventType.LLM_THINKING: "yellow",
            EventType.LLM_RESPONSE_RECEIVED: "yellow",
            EventType.CODE_GENERATED: "magenta",
            EventType.CODE_VALIDATING: "magenta",
            EventType.CODE_VALIDATION_RESULT: "magenta",
            EventType.CODE_EXECUTING: "cyan",
            EventType.CODE_EXECUTION_RESULT: "cyan",
            EventType.RECURSION_START: "bold_blue",
            EventType.RECURSION_STEP: "bold_blue",
            EventType.RECURSION_SUBMODEL_CALL: "bold_blue",
            EventType.RECURSION_END: "bold_blue",
            EventType.AGGREGATION_START: "green",
            EventType.AGGREGATION_END: "green",
            EventType.ERROR: "bold_red",
            EventType.FALLBACK_TRIGGERED: "red",
        }

        color = type_colors.get(event.type, "reset")
        duration = f" {c('dim', f'({event.duration_ms:.0f}ms)')}" if event.duration_ms else ""

        # Print separator for major events
        if event.type in (EventType.PROCESS_START, EventType.PROCESS_END):
            print(f"\n{c('bold', '═' * 70)}")

        # Print event header
        header = f"[{time_str}] (+{elapsed:.2f}s) #{self.event_count}"
        print(f"\n{c('dim', header)}")
        print(f"{indent}{c(color, '▶ ' + event.type.value.upper())}{duration}")

        if not self.show_full_output:
            return

        # Print event-specific details
        data = event.data

        if event.type == EventType.PROCESS_START:
            print(f"{indent}  {c('bold', 'Query:')} {data.get('query', '')}")
            print(f"{indent}  {c('bold', 'Context:')} {data.get('context_length', 0):,} characters")
            print(f"{indent}  {c('bold', 'Max Depth:')} {data.get('max_depth', 0)}")

        elif event.type == EventType.CONTEXT_CHUNKED:
            print(f"{indent}  Chunks created: {data.get('num_chunks', 0)}")
            print(f"{indent}  Total tokens: {data.get('total_tokens', 0):,}")
            print(f"{indent}  Strategy: {data.get('strategy', 'unknown')}")

        elif event.type == EventType.LLM_PROMPT_PREPARED:
            model = data.get('model', 'unknown')
            print(f"{indent}  {c('bold', 'Model:')} {model}")
            print(f"{indent}  Prompt length: {data.get('prompt_length', 0):,} chars")
            if data.get('system_prompt_preview'):
                preview = data['system_prompt_preview'][:100].replace('\n', ' ')
                print(f"{indent}  System: {c('dim', preview)}...")
            if data.get('user_prompt_preview'):
                preview = data['user_prompt_preview'][:100].replace('\n', ' ')
                print(f"{indent}  User: {c('dim', preview)}...")

        elif event.type == EventType.LLM_THINKING:
            model = data.get('model', 'unknown')
            task = data.get('task', 'inference')
            print(f"{indent}  {c('yellow', '⏳')} Model {c('bold', model)} is thinking...")
            print(f"{indent}  Task: {task}")

        elif event.type == EventType.LLM_RESPONSE_RECEIVED:
            print(f"{indent}  Tokens: {data.get('tokens_used', 0)}")
            print(f"{indent}  Finish reason: {data.get('finish_reason', 'unknown')}")
            if data.get('response_preview'):
                preview = data['response_preview'][:150].replace('\n', ' ')
                print(f"{indent}  Response: {c('dim', preview)}...")

        elif event.type == EventType.CODE_GENERATED:
            code = data.get('code', '')
            print(f"{indent}  Code length: {len(code)} characters")
            print(f"{indent}  {c('dim', '┌' + '─' * 60)}")
            for i, line in enumerate(code.split('\n')[:20]):
                line_num = c('dim', f'{i+1:3}│')
                print(f"{indent}  {line_num} {line}")
            if code.count('\n') > 20:
                remaining = code.count('\n') - 20
                print(f"{indent}  {c('dim', '   │ ... (' + str(remaining) + ' more lines)')}")
            print(f"{indent}  {c('dim', '└' + '─' * 60)}")

        elif event.type == EventType.CODE_VALIDATION_RESULT:
            is_valid = data.get('is_valid', False)
            status = c('green', '✅ PASSED') if is_valid else c('red', '❌ FAILED')
            print(f"{indent}  Validation: {status}")
            if data.get('errors'):
                for err in data['errors'][:5]:
                    print(f"{indent}  {c('red', '⚠')} {err}")
            if data.get('warnings'):
                for warn in data['warnings'][:3]:
                    print(f"{indent}  {c('yellow', 'ℹ')} {warn}")

        elif event.type == EventType.CODE_EXECUTING:
            print(f"{indent}  {c('cyan', '⚙')} Executing in sandbox...")
            print(f"{indent}  Timeout: {data.get('timeout', 5)}s")

        elif event.type == EventType.CODE_EXECUTION_RESULT:
            success = data.get('success', False)
            status = c('green', '✅ SUCCESS') if success else c('red', '❌ FAILED')
            print(f"{indent}  Execution: {status}")
            if data.get('result_preview'):
                print(f"{indent}  Result: {data['result_preview'][:150]}...")
            if data.get('error'):
                print(f"{indent}  {c('red', 'Error:')} {data['error']}")

        elif event.type == EventType.RECURSION_STEP:
            depth = data.get('current_depth', 0)
            max_d = data.get('max_depth', 1)
            print(f"{indent}  Depth: {depth} / {max_d}")
            print(f"{indent}  Query: {data.get('query', '')[:80]}...")

        elif event.type == EventType.RECURSION_SUBMODEL_CALL:
            print(f"{indent}  {c('blue', '📞')} Calling sub-model...")
            print(f"{indent}  Chunk size: {data.get('chunk_size', 0):,} chars")
            print(f"{indent}  Sub-query: {data.get('sub_query', '')[:60]}...")

        elif event.type == EventType.AGGREGATION_START:
            print(f"{indent}  Aggregating {data.get('num_results', 0)} results...")

        elif event.type == EventType.AGGREGATION_END:
            print(f"{indent}  Final result: {data.get('result_length', 0):,} chars")

        elif event.type == EventType.ERROR:
            print(f"{indent}  {c('bold_red', '❌ ERROR:')} {data.get('error', 'Unknown')}")

        elif event.type == EventType.FALLBACK_TRIGGERED:
            print(f"{indent}  {c('yellow', '⚠')} Falling back: {data.get('reason', 'unknown')}")

        elif event.type == EventType.PROCESS_END:
            print(f"{indent}  {c('bold', 'Total tokens:')} {data.get('total_tokens', 0):,}")
            print(f"{indent}  {c('bold', 'Total time:')} {data.get('total_time', 0):.2f}s")
            print(f"{indent}  {c('bold', 'Recursive calls:')} {data.get('num_calls', 0)}")
            if data.get('answer_preview'):
                print(f"{indent}  {c('bold', 'Answer preview:')}")
                for line in data['answer_preview'][:300].split('\n')[:5]:
                    print(f"{indent}    {line}")
            print(f"\n{c('bold', '═' * 70)}")


def print_header() -> None:
    """Print the demo header."""
    print("\033[1;36m")  # Bold cyan
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██████╗ ██╗     ███╗   ███╗    ████████╗██████╗  █████╗ ███╗   ██╗ ║
║   ██╔══██╗██║     ████╗ ████║    ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║ ║
║   ██████╔╝██║     ██╔████╔██║       ██║   ██████╔╝███████║██╔██╗ ██║ ║
║   ██╔══██╗██║     ██║╚██╔╝██║       ██║   ██╔══██╗██╔══██║██║╚██╗██║ ║
║   ██║  ██║███████╗██║ ╚═╝ ██║       ██║   ██║  ██║██║  ██║██║ ╚████║ ║
║   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ║
║                                                                      ║
║              Transparent Recursive Language Model Demo               ║
║                  Full Visibility Into Every Step                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    print("\033[0m")  # Reset


async def run_transparent_demo(
    provider: str = "ollama",
    use_tiers: bool = True,
    show_models: bool = True,
    quiet: bool = False,
) -> None:
    """Run the transparent demo with full visibility.

    Args:
        provider: LLM provider to use
        use_tiers: Whether to use tiered models (lightweight + heavyweight)
        show_models: Whether to show available models
        quiet: Quiet mode with minimal output
    """
    print_header()

    # Show available models
    if show_models and not quiet:
        print("\n\033[1m📚 Available Models:\033[0m")
        print_all_models()

    # Load configuration
    config = RLMConfig()

    # Create callback
    callback = VerboseCallback(show_full_output=not quiet) if not quiet else None

    # Setup models
    print(f"\n\033[1m🔧 Setting up models (provider: {provider})...\033[0m")

    if use_tiers:
        try:
            tiers = ModelTiers(provider)
            print(f"\n  Using tiered models from {provider}:")

            lightweight_spec = tiers.get_spec(ModelTier.LIGHTWEIGHT)
            heavyweight_spec = tiers.get_spec(ModelTier.HEAVYWEIGHT)

            print(f"  ├─ Code generation: {lightweight_spec.model}")
            print(f"  │    └─ {lightweight_spec.description}")
            print(f"  └─ Reasoning: {heavyweight_spec.model}")
            print(f"       └─ {heavyweight_spec.description}")

            code_llm = tiers.get_lightweight()
            reasoning_llm = tiers.get_heavyweight()

        except Exception as e:
            print(f"\n  ⚠ Could not create tiered models: {e}")
            print("  Falling back to single model...")
            code_llm = create_llm(provider=provider, model=config.llm.model)
            reasoning_llm = code_llm
    else:
        code_llm = create_llm(provider=provider, model=config.llm.model)
        reasoning_llm = code_llm
        print(f"\n  Using single model: {config.llm.model}")

    # Create transparent engine
    print("\n\033[1m🚀 Creating Transparent Engine...\033[0m")
    engine = TransparentEngine(
        llm=reasoning_llm,
        code_llm=code_llm,
        config=config,
        callback=callback or default_callback,
    )

    # Show configuration
    print(f"\n  Configuration:")
    print(f"  ├─ Max recursion depth: {config.max_recursion_depth}")
    print(f"  ├─ Default chunk size: {config.default_chunk_size} tokens")
    print(f"  ├─ Execution timeout: {config.execution.timeout}s")
    print(f"  └─ Chunking strategy: {config.chunking_strategy}")

    # Process the sample text
    query = "What are the three main components of an RLM system and how do they work together?"

    print(f"\n\033[1m📝 Processing Query:\033[0m")
    print(f"  Query: {query}")
    print(f"  Context: {len(SAMPLE_TEXT):,} characters")

    print("\n\033[1m" + "─" * 70 + "\033[0m")
    print("\033[1;33m  STARTING TRANSPARENT PROCESSING - WATCH EVERY STEP  \033[0m")
    print("\033[1m" + "─" * 70 + "\033[0m\n")

    try:
        result = await engine.process(
            query=query,
            context=SAMPLE_TEXT,
        )

        # Print final summary
        print("\n\033[1;32m" + "═" * 70 + "\033[0m")
        print("\033[1;32m  PROCESSING COMPLETE - FINAL RESULTS  \033[0m")
        print("\033[1;32m" + "═" * 70 + "\033[0m")

        print(f"\n\033[1m📊 Statistics:\033[0m")
        print(f"  ├─ Total tokens used: {result.total_tokens:,}")
        print(f"  ├─ Execution time: {result.execution_time:.2f}s")
        print(f"  ├─ Recursive calls: {result.num_recursive_calls}")
        print(f"  └─ Max depth reached: {result.max_depth_reached}")

        # Event summary
        summary = engine.get_event_summary()
        print(f"\n\033[1m📋 Event Summary:\033[0m")
        print(f"  ├─ Total events: {summary['total_events']}")
        print(f"  ├─ LLM time: {summary['total_llm_time_ms']:.0f}ms")
        print(f"  └─ Execution time: {summary['total_exec_time_ms']:.0f}ms")

        print(f"\n\033[1m📌 Answer:\033[0m")
        print("\033[36m" + "─" * 70 + "\033[0m")
        print(result.answer)
        print("\033[36m" + "─" * 70 + "\033[0m")

    except Exception as e:
        print(f"\n\033[1;31m❌ Error: {e}\033[0m")
        import traceback
        traceback.print_exc()


async def demo_model_comparison() -> None:
    """Demonstrate the difference between lightweight and heavyweight models."""
    print_header()

    print("\n\033[1m🔬 Model Comparison Demo\033[0m")
    print("This demo shows the difference between lightweight and heavyweight models.\n")

    provider = "ollama"  # Use local models for demo

    try:
        tiers = ModelTiers(provider)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Print tier information
    print("Available tiers:")
    tiers.print_catalog()

    # Get models
    code_llm = tiers.get_lightweight()
    reasoning_llm = tiers.get_heavyweight()

    # Simple test
    test_prompt = "What is 2+2? Answer in one word."

    print("\n\033[1mTesting both models with simple prompt:\033[0m")
    print(f"  Prompt: {test_prompt}\n")

    # Test lightweight
    print(f"  [LIGHTWEIGHT] {code_llm.model}")
    start = datetime.now()
    response = await code_llm.generate(test_prompt)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"    Response: {response.content.strip()}")
    print(f"    Time: {elapsed:.2f}s, Tokens: {response.tokens_used}")

    # Test heavyweight
    print(f"\n  [HEAVYWEIGHT] {reasoning_llm.model}")
    start = datetime.now()
    response = await reasoning_llm.generate(test_prompt)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"    Response: {response.content.strip()}")
    print(f"    Time: {elapsed:.2f}s, Tokens: {response.tokens_used}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transparent RLM Demo - Full visibility into processing"
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("RLM_LLM_PROVIDER", "ollama"),
        help="LLM provider (ollama, groq, openai, anthropic)",
    )
    parser.add_argument(
        "--no-tiers",
        action="store_true",
        help="Use single model instead of tiered models",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - minimal output",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run model comparison demo instead",
    )
    parser.add_argument(
        "--show-models",
        action="store_true",
        help="Show all available models before starting",
    )

    args = parser.parse_args()

    if args.compare:
        asyncio.run(demo_model_comparison())
    else:
        asyncio.run(run_transparent_demo(
            provider=args.provider,
            use_tiers=not args.no_tiers,
            show_models=args.show_models,
            quiet=args.quiet,
        ))


if __name__ == "__main__":
    main()
