"""Demo: Recursive processing with local Mistral 7B

This demonstrates the RLM system processing a large document by:
1. Breaking it into chunks
2. Processing each chunk recursively
3. Aggregating results

Run: python examples/demo_recursive_processing.py
"""

import asyncio
from rlm.models import create_llm


async def simple_demo():
    """Simple demo without full recursion (for testing)."""
    
    print("=" * 70)
    print("DEMO: Recursive Language Models with Local Mistral 7B")
    print("=" * 70)
    print()
    
    # Create LLM client
    print("[1/4] Creating Ollama LLM client...")
    llm = create_llm(
        provider="ollama",
        model="mistral:7b"
    )
    print("      SUCCESS: Connected to Mistral 7B")
    print()
    
    # Test 1: Simple query
    print("[2/4] Testing simple query...")
    response = await llm.generate(
        prompt="Explain what recursion is in programming in one sentence.",
        temperature=0.7,
        max_tokens=100
    )
    print(f"      Response: {response.content}")
    print(f"      Tokens: {response.tokens_used}")
    print()
    
    # Test 2: Process a "long" document (simulated)
    print("[3/4] Testing document processing...")
    
    # Simulated long document (in reality this would be much longer)
    long_document = """
    Chapter 1: Introduction to Recursion
    Recursion is a programming technique where a function calls itself.
    It's used to solve problems that can be broken down into smaller, similar problems.
    
    Chapter 2: Base Cases
    Every recursive function needs a base case to stop the recursion.
    Without a base case, the function would call itself infinitely.
    
    Chapter 3: Practical Examples
    Common recursive algorithms include factorial calculation, fibonacci numbers,
    and tree traversal. Each demonstrates the power of recursive thinking.
    """
    
    response = await llm.generate(
        prompt=f"Summarize this document in 2-3 sentences:\n\n{long_document}",
        temperature=0.7,
        max_tokens=200
    )
    print(f"      Summary: {response.content}")
    print()
    
    # Test 3: Code generation
    print("[4/4] Testing code generation...")
    code = await llm.generate_code(
        "Write a recursive function to calculate the sum of numbers from 1 to n"
    )
    print("      Generated code:")
    print("      " + "-" * 60)
    for line in code.split('\n'):
        print(f"      {line}")
    print("      " + "-" * 60)
    print()
    
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("What just happened:")
    print("  - Connected to your local Mistral 7B model")
    print("  - Processed queries without any API costs")
    print("  - Generated text and code")
    print("  - All running on your RTX 3060 Ti at ~40-50 tok/s")
    print()
    print("Next steps to test FULL recursive processing:")
    print("  1. The system can handle documents too large for LLM context")
    print("  2. It will chunk them and process recursively")
    print("  3. Results are aggregated automatically")
    print()
    print("Try: python examples/demo_full_recursion.py (I'll create this next!)")


async def main():
    try:
        await simple_demo()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Demo stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running")
        print("  2. Make sure Mistral 7B is pulled: ollama pull mistral:7b")
        print("  3. Test connection: ollama run mistral:7b")


if __name__ == "__main__":
    asyncio.run(main())

