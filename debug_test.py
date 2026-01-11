#!/usr/bin/env python3
"""Debug script to test TransparentEngine directly."""

import asyncio
import os
import sys
from pathlib import Path

# Load .env file
project_root = Path(__file__).parent
env_file = project_root / ".env"
if env_file.exists():
    print(f"Loading .env from {env_file}")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Add src to path
sys.path.insert(0, str(project_root / "src"))

print("=" * 60)
print("DEBUG: Testing TransparentEngine directly")
print("=" * 60)

# Check API keys
print(f"\nAPI Keys present:")
print(f"  GROQ_API_KEY: {bool(os.environ.get('GROQ_API_KEY'))}")
print(f"  OPENAI_API_KEY: {bool(os.environ.get('OPENAI_API_KEY'))}")
print(f"  ANTHROPIC_API_KEY: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")

# Step 1: Test imports
print("\n[1/6] Testing imports...")
try:
    from rlm import RLMConfig
    print("  ✓ RLMConfig imported")
except Exception as e:
    print(f"  ✗ RLMConfig import failed: {e}")
    sys.exit(1)

try:
    from rlm.core.transparent import TransparentEngine, EventType, Event
    print("  ✓ TransparentEngine imported")
except Exception as e:
    print(f"  ✗ TransparentEngine import failed: {e}")
    sys.exit(1)

try:
    from rlm.models import create_llm, ModelTiers, ModelTier
    print("  ✓ Model imports successful")
except Exception as e:
    print(f"  ✗ Model imports failed: {e}")
    sys.exit(1)

# Step 2: Test config
print("\n[2/6] Testing configuration...")
try:
    config = RLMConfig()
    print(f"  ✓ Config loaded")
    print(f"    Provider: {config.llm.provider}")
    print(f"    Model: {config.llm.model}")
except Exception as e:
    print(f"  ✗ Config failed: {e}")
    sys.exit(1)

# Step 3: Test LLM creation
print("\n[3/6] Testing LLM creation...")
provider = "groq"  # Change to "ollama" if needed

try:
    tiers = ModelTiers(provider)
    code_llm = tiers.get_lightweight()
    reasoning_llm = tiers.get_heavyweight()
    print(f"  ✓ Model tiers created for {provider}")
    print(f"    Code LLM: {code_llm.model}")
    print(f"    Reasoning LLM: {reasoning_llm.model}")
except Exception as e:
    print(f"  ✗ LLM creation failed: {e}")
    print(f"    Falling back to single model...")
    try:
        llm = create_llm(provider=provider)
        code_llm = llm
        reasoning_llm = llm
        print(f"  ✓ Single model created: {llm.model}")
    except Exception as e2:
        print(f"  ✗ Single model also failed: {e2}")
        sys.exit(1)

# Step 4: Test event callback
print("\n[4/6] Testing event callback...")
events_received = []

def test_callback(event: Event):
    """Simple callback that logs events."""
    events_received.append(event)
    print(f"  [EVENT] {event.type.value} (depth={event.depth})")
    if event.data:
        # Show some key data
        for key in ['query', 'context_length', 'model', 'code', 'error']:
            if key in event.data:
                value = str(event.data[key])[:80]
                print(f"          {key}: {value}...")

print("  ✓ Callback defined")

# Step 5: Create engine
print("\n[5/6] Creating TransparentEngine...")
try:
    engine = TransparentEngine(
        llm=reasoning_llm,
        code_llm=code_llm,
        config=config,
        callback=test_callback
    )
    print("  ✓ TransparentEngine created")
except Exception as e:
    print(f"  ✗ TransparentEngine creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Run a test query
print("\n[6/6] Running test query...")
print("=" * 60)

async def run_test():
    query = "What are the main components mentioned?"
    context = """
    Recursive Language Models (RLMs) consist of three main components:
    1. The Context Manager - handles large documents by chunking them
    2. The Code Generation layer - creates Python code to process context
    3. The Execution Sandbox - safely runs generated code
    """

    print(f"\nQuery: {query}")
    print(f"Context length: {len(context)} chars")
    print("\nProcessing (events will appear below)...")
    print("-" * 60)

    try:
        result = await engine.process(query=query, context=context)
        print("-" * 60)
        print(f"\n✓ Processing complete!")
        print(f"  Total events: {len(events_received)}")
        print(f"  Total tokens: {result.total_tokens}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        print(f"  Recursive calls: {result.num_recursive_calls}")
        print(f"\nAnswer:\n{result.answer[:500]}...")

    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

# Run the async test
asyncio.run(run_test())

print("\n" + "=" * 60)
print("DEBUG TEST COMPLETE")
print("=" * 60)
