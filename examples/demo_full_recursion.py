"""Full RLM Demo: Recursive context processing

This demonstrates the COMPLETE RLM system with:
- Large context that exceeds typical LLM windows
- Automatic chunking
- Recursive processing
- Result aggregation

Run: python examples/demo_full_recursion.py
"""

import asyncio
from rlm.models import create_llm
from rlm.context import ContextManager, SemanticChunking
from rlm.execution import SandboxEnvironment
from rlm.core import RecursiveInferenceEngine
from rlm.config import RLMConfig


async def full_rlm_demo():
    """Demonstrate full recursive processing."""
    
    print("=" * 70)
    print("FULL RLM DEMO: Processing Large Context Recursively")
    print("=" * 70)
    print()
    
    # Step 1: Setup
    print("[SETUP] Initializing RLM system...")
    
    config = RLMConfig()
    llm = create_llm(provider="ollama", model="mistral:7b")
    context_manager = ContextManager(strategy=SemanticChunking())
    sandbox = SandboxEnvironment(timeout=10)
    
    engine = RecursiveInferenceEngine(
        llm=llm,
        context_manager=context_manager,
        sandbox=sandbox,
        max_depth=5
    )
    
    print("      SUCCESS: All components initialized")
    print(f"      - LLM: Ollama Mistral 7B")
    print(f"      - Chunking: Semantic")
    print(f"      - Max recursion depth: 5")
    print()
    
    # Step 2: Create a large document
    print("[INPUT] Creating large document...")
    
    # This is a simulated "large" document
    # In real use, this would be 100K+ tokens
    large_document = """
    SECTION 1: Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on the development of computer programs that can access data and
    use it to learn for themselves. The process of learning begins with observations
    or data, such as examples, direct experience, or instruction, in order to look
    for patterns in data and make better decisions in the future.
    
    SECTION 2: Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised
    learning, and reinforcement learning. Supervised learning uses labeled data to
    train algorithms. Unsupervised learning finds hidden patterns in unlabeled data.
    Reinforcement learning learns through trial and error with rewards and penalties.
    
    SECTION 3: Applications
    
    Machine learning is used in many applications including image recognition,
    natural language processing, recommendation systems, fraud detection, and
    autonomous vehicles. These applications have transformed industries and continue
    to evolve with advances in computational power and algorithm development.
    
    SECTION 4: Deep Learning
    
    Deep learning is a subset of machine learning that uses neural networks with
    multiple layers. These networks can learn hierarchical representations of data
    and have achieved state-of-the-art results in many domains. Convolutional neural
    networks excel at image tasks, while recurrent neural networks are powerful for
    sequential data.
    
    SECTION 5: Future of AI
    
    The future of artificial intelligence includes more sophisticated algorithms,
    better hardware, and increased integration with everyday technology. Ethical
    considerations and responsible AI development will become increasingly important
    as these systems become more powerful and prevalent in society.
    """
    
    print(f"      Document length: {len(large_document)} characters")
    print(f"      Estimated tokens: ~{len(large_document.split())}")
    print()
    
    # Step 3: Process with RLM
    print("[PROCESSING] Running recursive inference...")
    print("      (This may take 30-60 seconds with local model)")
    print()
    
    query = "Summarize the key points of this document about machine learning"
    
    try:
        result = await engine.process(
            query=query,
            context=large_document
        )
        
        print("[RESULTS] Processing complete!")
        print("=" * 70)
        print()
        print("Summary:")
        print(result.answer)
        print()
        print("=" * 70)
        print("Processing Statistics:")
        print(f"  - Total tokens used: {result.total_tokens}")
        print(f"  - Execution time: {result.execution_time:.2f} seconds")
        print(f"  - Recursive calls: {result.num_recursive_calls}")
        print(f"  - Recursion depth: {result.recursion_tree.depth}")
        print()
        
        print("Recursion Tree:")
        print_recursion_tree(result.recursion_tree, indent=0)
        print()
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        print()
        print("This is expected if the full recursive engine isn't complete yet.")
        print("The basic LLM integration is working though!")
        import traceback
        traceback.print_exc()


def print_recursion_tree(node, indent=0):
    """Pretty print the recursion tree."""
    prefix = "  " * indent
    print(f"{prefix}├─ Depth {node.depth}: {node.query[:50]}...")
    print(f"{prefix}   Tokens: {node.tokens_used}")
    
    for sub_node in node.sub_calls:
        print_recursion_tree(sub_node, indent + 1)


async def main():
    print()
    print("NOTE: This demo requires the full RLM engine to be complete.")
    print("If you get errors, try the simpler demo first:")
    print("  python examples/demo_recursive_processing.py")
    print()
    input("Press Enter to continue...")
    print()
    
    try:
        await full_rlm_demo()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Demo stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nIf the engine isn't fully implemented yet:")
        print("  1. Run the simpler demo: python examples/demo_recursive_processing.py")
        print("  2. Complete implementation following docs/IMPLEMENTATION_GUIDE.md")
        print("  3. Check .claude/TODO.md for remaining tasks")


if __name__ == "__main__":
    asyncio.run(main())

