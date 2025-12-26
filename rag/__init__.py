"""
RAG (Retrieval-Augmented Generation) module for Knowledge Graph triplets.

This module provides functionality to:
1. Load and parse KG triplets from EDC pipeline output (canon_kg.txt)
2. Convert triplets to embeddable text representations
3. Generate embeddings using sentence-transformers
4. Store and retrieve from FAISS index
5. Retrieve relevant triplets for a query
6. Build augmented prompts with KG context
7. Generate answers using local LLM

Usage:
    from rag import KGRagIndexer, KGRagGenerator
    
    # Index from EDC output
    indexer = KGRagIndexer()
    indexer.index_from_path("./output/tmp", mode="triplet_text")
    indexer.save("./output/rag")
    
    # Search only
    indexer = KGRagIndexer.load("./output/rag")
    results = indexer.search("Where is Trane located?")
    
    # Full RAG with LLM generation
    generator = KGRagGenerator(indexer)
    result = generator.generate("Where is Trane located?")
    print(result.answer)
"""

from .triplet_loader import TripletLoader, Triplet
from .representation import TripletRepresenter, RepresentationMode, EmbeddableItem, get_representer
from .embedder import Embedder, get_embedder
from .faiss_store import FaissStore, SearchResult
from .kg_rag_indexer import KGRagIndexer
from .retriever import KGRetriever, RetrievalResult, ContextFormat
from .prompt_builder import KGPromptBuilder, get_prompt_builder
from .generator import KGRagGenerator, GenerationResult, create_generator

__all__ = [
    # Main interfaces
    "KGRagIndexer",
    "KGRagGenerator",
    "create_generator",
    
    # Data loading
    "TripletLoader",
    "Triplet",
    
    # Representation
    "TripletRepresenter",
    "RepresentationMode",
    "EmbeddableItem",
    "get_representer",
    
    # Embeddings
    "Embedder",
    "get_embedder",
    
    # Storage
    "FaissStore",
    "SearchResult",
    
    # Retrieval (Step 5)
    "KGRetriever",
    "RetrievalResult",
    "ContextFormat",
    
    # Prompt Building (Step 6)
    "KGPromptBuilder",
    "get_prompt_builder",
    
    # Generation (Step 7)
    "GenerationResult",
]
