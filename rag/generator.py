"""
Step 7: End-to-end RAG generator for KG-based Q&A.

Combines retrieval, prompt augmentation, and LLM generation.

Supports multiple LLM providers via environment configuration:
- Local LLM: source export_local_llm.sh
- Google AI Studio: source export_google_ai.sh  
- SambaNova: source export_sambanova.sh
"""

import os
import sys
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

# Add parent directory for edc imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .kg_rag_indexer import KGRagIndexer
from .retriever import KGRetriever, RetrievalResult, ContextFormat
from .prompt_builder import KGPromptBuilder, get_prompt_builder

# Import the unified chat completion function that routes to local or API
from edc.utils.llm_utils import openai_chat_completion

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Result from KG-RAG generation.
    
    Attributes:
        answer: Generated answer text
        sources: List of (subject, predicate, object) triplets used
        scores: Relevance scores of source triplets
        query: Original query
        context: Full retrieval result
        prompt: The augmented prompt sent to the LLM
    """
    answer: str
    sources: List[Tuple[str, str, str]]
    scores: List[float]
    query: str
    context: Optional[RetrievalResult] = None
    prompt: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"GenerationResult(answer='{self.answer[:50]}...', sources={len(self.sources)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": [
                {"subject": s, "predicate": p, "object": o, "score": score}
                for (s, p, o), score in zip(self.sources, self.scores)
            ],
            "num_sources": len(self.sources),
        }


class KGRagGenerator:
    """
    End-to-end RAG pipeline for KG-based question answering.
    
    Combines:
    1. KGRetriever - Retrieve relevant triplets
    2. KGPromptBuilder - Build augmented prompt
    3. LLM - Generate answer (via openai_chat_completion)
    
    Supports multiple LLM providers based on environment configuration:
    - Local LLM: source export_local_llm.sh (requires GPU + bitsandbytes)
    - Google AI Studio: source export_google_ai.sh
    - SambaNova: source export_sambanova.sh
    
    Usage:
        # Configure provider first: source export_google_ai.sh
        indexer = KGRagIndexer.load("./output/rag")
        generator = KGRagGenerator(indexer)
        result = generator.generate("Where is Trane located?")
        print(result.answer)
    """
    
    def __init__(
        self,
        indexer: KGRagIndexer,
        template_path: Optional[str] = None,
        context_format: ContextFormat = ContextFormat.TRIPLET_LIST,
    ):
        """
        Initialize the generator.
        
        Args:
            indexer: A loaded KGRagIndexer instance
            template_path: Optional custom prompt template path
            context_format: Format for retrieved context
        """
        self.indexer = indexer
        self.retriever = KGRetriever(indexer, default_format=context_format)
        self.prompt_builder = get_prompt_builder(template_path=template_path)
    
    def generate(
        self,
        query: str,
        top_k: int = 10,
        temperature: float = 0.1,
        max_tokens: int = 256,
        include_scores: bool = False,
        return_prompt: bool = False,
    ) -> GenerationResult:
        """
        Generate an answer using the full RAG pipeline.
        
        Args:
            query: Natural language question
            top_k: Number of triplets to retrieve
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens to generate
            include_scores: Include relevance scores in prompt
            return_prompt: Include the prompt in the result
            
        Returns:
            GenerationResult with answer and sources
        """
        logger.info(f"Generating answer for: {query}")
        
        # Step 5: Retrieve relevant triplets
        logger.debug("Step 5: Retrieving relevant triplets...")
        context = self.retriever.retrieve(query, top_k=top_k)
        logger.debug(f"Retrieved {len(context)} triplets")
        
        # Step 6: Build augmented prompt
        logger.debug("Step 6: Building augmented prompt...")
        messages = self.prompt_builder.build_chat_messages(
            query, context, include_scores=include_scores
        )
        system_prompt = self.prompt_builder.get_system_prompt()
        
        # Step 7: Generate with LLM (routes to local or API based on env config)
        logger.debug("Step 7: Generating with LLM...")
        answer = openai_chat_completion(
            system_prompt=system_prompt,
            history=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        logger.info(f"Generated answer: {answer[:100]}...")
        
        # Build result
        result = GenerationResult(
            answer=answer,
            sources=context.triplets,
            scores=context.scores,
            query=query,
            context=context if return_prompt else None,
            prompt=self.prompt_builder.build(query, context) if return_prompt else None,
        )
        
        return result
    
    def generate_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> List[GenerationResult]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of questions
            top_k: Number of triplets per query
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens per answer
            
        Returns:
            List of GenerationResult objects
        """
        results = []
        for query in queries:
            result = self.generate(
                query,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)
        return results
    
    def retrieve_only(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Only retrieve triplets without LLM generation.
        
        Useful for debugging or when you want to handle generation separately.
        
        Args:
            query: Natural language question
            top_k: Number of triplets to retrieve
            
        Returns:
            RetrievalResult with triplets and formatted context
        """
        return self.retriever.retrieve(query, top_k=top_k)
    
    def get_prompt(
        self,
        query: str,
        top_k: int = 10,
        include_scores: bool = False,
    ) -> str:
        """
        Get the prompt that would be sent to the LLM.
        
        Useful for debugging or customization.
        
        Args:
            query: Natural language question
            top_k: Number of triplets to retrieve
            include_scores: Include relevance scores
            
        Returns:
            Formatted prompt string
        """
        context = self.retriever.retrieve(query, top_k=top_k)
        return self.prompt_builder.build(query, context, include_scores=include_scores)


def create_generator(
    index_path: str,
    template_path: Optional[str] = None,
    device: Optional[str] = None,
) -> KGRagGenerator:
    """
    Convenience function to create a KGRagGenerator.
    
    LLM provider is determined by environment configuration:
    - source export_local_llm.sh for local LLM
    - source export_google_ai.sh for Google AI Studio
    - source export_sambanova.sh for SambaNova
    
    Args:
        index_path: Path to saved FAISS index
        template_path: Optional prompt template path
        device: Device for embeddings
        
    Returns:
        Configured KGRagGenerator
    """
    indexer = KGRagIndexer.load(index_path, device=device)
    return KGRagGenerator(
        indexer=indexer,
        template_path=template_path,
    )


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Test KG-RAG Generator")
    parser.add_argument("--index", required=True, help="Path to FAISS index")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--top_k", type=int, default=5, help="Number of triplets to retrieve")
    
    args = parser.parse_args()
    
    # Create generator
    generator = create_generator(args.index)
    
    # Generate
    result = generator.generate(args.query, top_k=args.top_k, return_prompt=True)
    
    print("\n" + "=" * 60)
    print(f"Query: {result.query}")
    print("=" * 60)
    
    print("\n=== Retrieved Sources ===")
    for (s, p, o), score in zip(result.sources, result.scores):
        print(f"  [{score:.3f}] ({s}, {p}, {o})")
    
    print("\n=== Generated Answer ===")
    print(result.answer)
    print("=" * 60)


