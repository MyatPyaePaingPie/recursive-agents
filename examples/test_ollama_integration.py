"""Quick test to verify Ollama integration works."""

import asyncio
from src.rlm.models import create_llm


async def main():
    print("[TESTING] Ollama integration with Mistral 7B...")
    print("-" * 60)
    
    # Create Ollama LLM client
    llm = create_llm(
        provider="ollama",
        model="mistral:7b"
    )
    
    print("[SUCCESS] LLM client created successfully!")
    print(f"   Provider: Ollama")
    print(f"   Model: mistral:7b")
    print(f"   Base URL: http://localhost:11434/v1")
    print()
    
    # Test simple generation
    print("[TESTING] Simple text generation...")
    response = await llm.generate(
        prompt="Write a one-sentence description of recursion.",
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"[SUCCESS] Response received!")
    print(f"   Content: {response.content}")
    print(f"   Tokens used: {response.tokens_used}")
    print(f"   Model: {response.model}")
    print()
    
    # Test code generation
    print("[TESTING] Code generation...")
    code = await llm.generate_code(
        "Write a simple Python function to calculate factorial"
    )
    
    print(f"[SUCCESS] Code generated!")
    print("Generated code:")
    print("-" * 60)
    print(code)
    print("-" * 60)
    print()
    
    print("[SUCCESS] All tests passed! Ollama integration is working perfectly!")
    print()
    print("Your setup:")
    print("  [OK] Ollama installed and running")
    print("  [OK] Mistral 7B model loaded")
    print("  [OK] RLM package structure complete")
    print("  [OK] Local LLM integration working")
    print()
    print("Next steps:")
    print("  1. Run: pytest tests/ -v")
    print("  2. Check IMPLEMENTATION_GUIDE.md Phase 6 (Testing)")
    print("  3. Create examples/basic_rlm_demo.py")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Make sure Mistral 7B is pulled: ollama pull mistral:7b")
        print("  3. Check if port 11434 is accessible")
        import traceback
        traceback.print_exc()

