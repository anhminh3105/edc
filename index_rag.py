#!/usr/bin/env python
"""
CLI script for KG-RAG: index, search, and generate answers from knowledge graphs.

Supports:
- Indexing triplets from EDC pipeline output (canon_kg.txt)
- Search-only mode (retrieve relevant triplets)
- Generate mode (full RAG with LLM answer generation)

Usage:
    # Index EDC output
    python index_rag.py --input ./output/tmp --output_dir ./output/rag

    # Search only (retrieve triplets)
    python index_rag.py --load ./output/rag --query "Where is Trane located?"

    # Generate answers with LLM
    python index_rag.py --load ./output/rag --generate --query "Where is Trane located?"

    # Interactive Q&A with LLM
    python index_rag.py --load ./output/rag --generate --interactive
"""

import argparse
import logging
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import KGRagIndexer


def setup_logging(level: str) -> None:
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def index_mode(args: argparse.Namespace) -> None:
    """Run the indexing pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("KG-RAG Indexer")
    logger.info("=" * 60)

    # Create indexer
    indexer = KGRagIndexer(
        embedding_model=args.embedding_model,
        device=args.device,
        use_gpu_faiss=args.gpu_faiss,
    )

    # Run indexing
    indexer.index_from_path(
        path=args.input,
        mode=args.mode,
        batch_size=args.batch_size,
        show_progress=True,
    )

    # Save
    indexer.save(args.output_dir, prefix=args.prefix)

    # Print stats
    stats = indexer.get_stats()
    logger.info("\n" + "=" * 60)
    logger.info("Indexing Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    print(f"\nIndex saved to: {args.output_dir}")
    print(f"Files created:")
    print(f"  - {args.prefix}.faiss (FAISS index)")
    print(f"  - {args.prefix}_meta.json (metadata)")
    print(f"  - {args.prefix}_config.json (config)")


def search_mode(args: argparse.Namespace) -> None:
    """Run search on an existing index (without LLM generation)."""
    logger = logging.getLogger(__name__)

    # Load indexer
    logger.info(f"Loading index from {args.load}")
    indexer = KGRagIndexer.load(
        args.load,
        prefix=args.prefix,
        device=args.device,
        use_gpu_faiss=args.gpu_faiss,
    )

    if args.interactive:
        interactive_search(indexer, args.top_k)
    elif args.query:
        # Single query
        results = indexer.search(args.query, top_k=args.top_k)
        print_search_results(args.query, results)
    else:
        print("No query provided. Use --query or --interactive")


def generate_mode(args: argparse.Namespace) -> None:
    """Run full RAG pipeline with LLM generation."""
    logger = logging.getLogger(__name__)

    # Import generator (requires LLM dependencies)
    from rag import KGRagGenerator

    # Load indexer
    logger.info(f"Loading index from {args.load}")
    indexer = KGRagIndexer.load(
        args.load,
        prefix=args.prefix,
        device=args.device,
        use_gpu_faiss=args.gpu_faiss,
    )

    # Create generator
    logger.info("Initializing KG-RAG Generator...")
    generator = KGRagGenerator(indexer)

    if args.interactive:
        interactive_generate(generator, args.top_k, args.temperature, args.max_tokens)
    elif args.query:
        # Single query
        result = generator.generate(
            args.query,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print_generation_result(result)
    else:
        print("No query provided. Use --query or --interactive")


def interactive_search(indexer: KGRagIndexer, top_k: int = 10) -> None:
    """Interactive search loop (retrieval only)."""
    print("\n" + "=" * 60)
    print("Interactive Search Mode (retrieval only)")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Query: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            results = indexer.search(query, top_k=top_k)
            print_search_results(query, results)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


def interactive_generate(
    generator,
    top_k: int = 10,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> None:
    """Interactive Q&A loop with LLM generation."""
    print("\n" + "=" * 60)
    print("Interactive Q&A Mode (with LLM generation)")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Question: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print("\nGenerating answer...")
            result = generator.generate(
                query,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print_generation_result(result)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


def print_search_results(query: str, results: list) -> None:
    """Print search results (triplets only)."""
    print(f"\nResults for: '{query}'")
    print("-" * 60)

    if not results:
        print("  No results found.")
        return

    for result in results:
        print(f"\n  [{result.rank + 1}] Score: {result.score:.4f}")

        # Print document/text
        doc = result.metadata.get("document", "")
        if doc:
            print(f"      Document: {doc}")

        # Print triplet info if available
        subject = result.metadata.get("subject", "")
        predicate = result.metadata.get("predicate", "")
        obj = result.metadata.get("object", "")
        if subject and predicate and obj:
            print(f"      Triplet: ({subject}, {predicate}, {obj})")

    print("-" * 60)


def print_generation_result(result) -> None:
    """Print generation result with answer and sources."""
    print("\n" + "=" * 60)
    print(f"Question: {result.query}")
    print("=" * 60)

    print("\n>>> ANSWER <<<")
    print(result.answer)

    print("\n--- Sources ---")
    if result.sources:
        for (s, p, o), score in zip(result.sources, result.scores):
            s_h = s.replace("_", " ")
            p_h = p.replace("_", " ")
            o_h = o.replace("_", " ")
            print(f"  [{score:.3f}] ({s_h}, {p_h}, {o_h})")
    else:
        print("  No sources retrieved.")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="KG-RAG: Index, search, and generate answers from knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index EDC output (from canon_kg.txt)
  python index_rag.py --input ./output/tmp --output_dir ./output/rag

  # Search only (retrieve triplets without LLM)
  python index_rag.py --load ./output/rag --query "Where is Trane located?"

  # Generate answer with LLM
  python index_rag.py --load ./output/rag --generate --query "Where is Trane located?"

  # Interactive Q&A with LLM
  python index_rag.py --load ./output/rag --generate --interactive

  # Interactive search (retrieval only)
  python index_rag.py --load ./output/rag --interactive
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--input",
        help="Path to EDC output directory or canon_kg.txt (indexing mode)",
    )
    mode_group.add_argument(
        "--load",
        help="Path to load existing index from (search/generate mode)",
    )

    # Indexing options
    parser.add_argument(
        "--output_dir",
        default="./output/rag",
        help="Output directory for index files (default: ./output/rag)",
    )
    parser.add_argument(
        "--mode",
        choices=["triplet_text", "entity_context"],
        default="triplet_text",
        help="Representation mode (default: triplet_text)",
    )
    parser.add_argument(
        "--prefix",
        default="kg_triplets",
        help="Filename prefix for saved files (default: kg_triplets)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)",
    )

    # Model options
    parser.add_argument(
        "--embedding_model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence transformer model (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device for embeddings (default: auto)",
    )
    parser.add_argument(
        "--gpu_faiss",
        action="store_true",
        help="Use GPU for FAISS (requires faiss-gpu)",
    )

    # Search/Generate options
    parser.add_argument(
        "--query",
        help="Query string for search/generate mode",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of triplets to retrieve (default: 10)",
    )

    # Generation options
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Enable LLM generation mode (requires local LLM setup)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature for generation (default: 0.1)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )

    # Logging
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Determine mode
    if args.input:
        index_mode(args)
    elif args.load:
        if args.generate:
            generate_mode(args)
        else:
            search_mode(args)
    else:
        parser.print_help()
        print("\nError: Must specify --input (to index) or --load (to search/generate)")
        sys.exit(1)


if __name__ == "__main__":
    main()
