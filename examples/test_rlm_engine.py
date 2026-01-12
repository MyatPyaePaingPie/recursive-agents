#!/usr/bin/env python3
"""Test the new paper-accurate RLM Engine.

This example demonstrates:
1. Complexity detection (heavyweight decides if RLM needed)
2. Raw context access (model discovers chunking)
3. FINAL() detection
4. Simple event tracking

Usage:
    # With Groq (free tier)
    GROQ_API_KEY=your_key python examples/test_rlm_engine.py

    # With Ollama (local)
    python examples/test_rlm_engine.py --provider ollama
"""

import argparse
import asyncio
import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlm.core import Event, RLMEngine
from rlm.models import create_llm


def print_event(event: Event) -> None:
    """Print events as they happen."""
    print(f"  [{event.type.value}] {event.data}")


async def main(provider: str, model: str | None = None) -> None:
    """Run the test."""
    print(f"\n{'='*60}")
    print("Testing Paper-Accurate RLM Engine")
    print(f"{'='*60}\n")

    # Create LLMs
    print(f"Provider: {provider}")

    if provider == "groq":
        heavyweight = create_llm("groq", model=model or "llama-3.3-70b-versatile")
        lightweight = create_llm("groq", model="llama-3.1-8b-instant")
    elif provider == "ollama":
        heavyweight = create_llm("ollama", model=model or "llama3:8b")
        lightweight = create_llm("ollama", model="phi3:3.8b")
    elif provider == "anthropic":
        heavyweight = create_llm("anthropic", model=model or "claude-3-5-sonnet-20241022")
        lightweight = create_llm("groq", model="llama-3.1-8b-instant")  # Cheaper for sub-calls
    else:
        raise ValueError(f"Unknown provider: {provider}")

    print(f"Heavyweight: {heavyweight}")
    print(f"Lightweight: {lightweight}")

    # Create engine
    engine = RLMEngine(
        heavyweight_llm=heavyweight,
        lightweight_llm=lightweight,
    )

    # Test 1: Small context (should skip RLM)
    print(f"\n{'-'*60}")
    print("TEST 1: Small context (should use DIRECT)")
    print(f"{'-'*60}")

    small_context = """
    The quick brown fox jumps over the lazy dog.
    This is a simple test document with minimal content.
    It should be processed directly without RLM.
    """

    result = await engine.process(
        query="What animals are mentioned?",
        context=small_context,
        on_event=print_event,
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Used RLM: {result.used_rlm}")
    print(f"Tokens: {result.total_tokens}")

    # Test 2: Large context (should use RLM)
    print(f"\n{'-'*60}")
    print("TEST 2: Large context (should use RLM)")
    print(f"{'-'*60}")

    # Generate a large context
    large_context = """
# Chapter 1: Introduction

This is the first chapter of our lengthy document. It introduces the main concepts
and sets the stage for what follows. The protagonist, Alice, discovers a mysterious
old book in her grandmother's attic. The book contains secrets about her family's
past and hints at a hidden treasure.

Alice is a curious young woman who works as a librarian. She has always been
fascinated by mysteries and puzzles. When she finds the book, she can't resist
the urge to investigate its secrets.

# Chapter 2: The Discovery

Alice opens the book and finds a map tucked between the pages. The map shows
a path through the nearby forest to what appears to be a cave. There are strange
symbols on the map that she doesn't recognize.

She decides to visit her uncle, Professor James, who is an expert in ancient
languages. He examines the symbols and tells her they are from an old Celtic
alphabet. The symbols spell out "Truth lies beneath the silver moon."

# Chapter 3: The Journey

Armed with this knowledge, Alice sets out on the night of the full moon.
She follows the path on the map, careful to note the landmarks mentioned.
After hours of walking, she finds the cave entrance hidden behind a waterfall.

Inside the cave, she discovers ancient artifacts and a chest. The chest
contains letters from her great-great-grandmother, revealing that their
family was once part of a secret society dedicated to preserving knowledge.

# Chapter 4: The Resolution

Alice realizes that the real treasure isn't gold or jewels, but the knowledge
contained in the letters. She learns about her family's role in protecting
important historical documents during times of war and persecution.

She decides to continue this legacy by establishing a foundation to preserve
and share historical knowledge. The mysterious book becomes the first item
in a new archive that she creates.

# Epilogue

Years later, Alice's archive has grown into a renowned research center.
Scholars from around the world come to study the documents she has collected.
Alice often thinks about that day in her grandmother's attic when everything began.
""" * 10  # Repeat to make it larger

    result = await engine.process(
        query="Summarize the main plot of this story and identify the protagonist's key discoveries.",
        context=large_context,
        on_event=print_event,
    )

    print(f"\nAnswer: {result.answer[:500]}...")
    print(f"Used RLM: {result.used_rlm}")
    print(f"Tokens: {result.total_tokens}")
    print(f"Turns: {len(result.trajectory.turns)}")

    # Test 3: Force RLM on small context
    print(f"\n{'-'*60}")
    print("TEST 3: Force RLM on small context")
    print(f"{'-'*60}")

    result = await engine.process(
        query="What is this text about?",
        context=small_context,
        force_rlm=True,
        on_event=print_event,
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Used RLM: {result.used_rlm}")
    print(f"Tokens: {result.total_tokens}")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RLM Engine")
    parser.add_argument(
        "--provider",
        choices=["groq", "ollama", "anthropic"],
        default="groq",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        help="Model to use for heavyweight (provider-specific)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.provider, args.model))
