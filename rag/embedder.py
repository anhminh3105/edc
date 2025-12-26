"""
Step 3: Embedding generation using sentence-transformers.

Generates embeddings for embeddable items using BGE or other sentence-transformer models.
"""

import logging
import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .representation import EmbeddableItem

logger = logging.getLogger(__name__)

# Default model from export_local_llm.sh
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Instruction prefix for retrieval (improves BGE performance)
DEFAULT_QUERY_INSTRUCTION = "Instruct: Retrieve descriptions of relations that are present in the given text.\nQuery: "


class Embedder:
    """
    Generates embeddings using sentence-transformers.
    
    Supports BGE models with instruction-based querying.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run on ("cuda", "cpu", or None for auto)
            query_instruction: Instruction prefix for query embeddings (BGE models)
        """
        self.model_name = model_name
        self.query_instruction = query_instruction
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_items(
        self,
        items: List[EmbeddableItem],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of embeddable items.
        
        Args:
            items: List of EmbeddableItem objects
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            
        Returns:
            numpy array of shape (len(items), embedding_dim)
        """
        texts = [item.text for item in items]
        return self.embed_texts(texts, batch_size, show_progress, normalize)
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def embed_query(
        self,
        query: str,
        use_instruction: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a query string for retrieval.
        
        For BGE models, adds the instruction prefix for better retrieval.
        
        Args:
            query: Query text
            use_instruction: Whether to add instruction prefix
            normalize: Whether to L2-normalize
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        if use_instruction and self.query_instruction:
            query = self.query_instruction + query
        
        embedding = self.model.encode(
            query,
            normalize_embeddings=normalize,
        )
        
        return embedding
    
    def embed_queries(
        self,
        queries: List[str],
        use_instruction: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed multiple query strings for retrieval.
        
        Args:
            queries: List of query texts
            use_instruction: Whether to add instruction prefix
            normalize: Whether to L2-normalize
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        if use_instruction and self.query_instruction:
            queries = [self.query_instruction + q for q in queries]
        
        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=normalize,
        )
        
        return embeddings


def get_embedder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Embedder:
    """
    Factory function to create an Embedder.
    
    Args:
        model_name: Model name (defaults to BGE-small)
        device: Device to use
        
    Returns:
        Configured Embedder instance
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    return Embedder(model_name=model_name, device=device)


if __name__ == "__main__":
    # Simple test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create embedder
    embedder = get_embedder()
    
    # Test embedding
    test_texts = [
        "instance of: The subject entity is an instance or example of the type specified by the object entity.",
        "country: The subject entity is located in the country specified by the object entity.",
        "John Doe is a student at National University of Singapore.",
    ]
    
    embeddings = embedder.embed_texts(test_texts)
    print(f"\nEmbedded {len(test_texts)} texts")
    print(f"Shape: {embeddings.shape}")
    
    # Test query
    query = "Where is John Doe studying?"
    query_embedding = embedder.embed_query(query)
    print(f"\nQuery embedding shape: {query_embedding.shape}")
    
    # Compute similarities
    similarities = embeddings @ query_embedding
    print(f"\nSimilarities to query '{query}':")
    for i, (text, sim) in enumerate(zip(test_texts, similarities)):
        print(f"  {i}: {sim:.4f} - {text[:60]}...")

