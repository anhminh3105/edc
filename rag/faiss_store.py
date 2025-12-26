"""
Step 4: FAISS vector store for fast similarity search.

Stores embeddings in FAISS index with separate JSON metadata file.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is not installed. Install with: "
        "pip install faiss-gpu"
    )

from .representation import EmbeddableItem

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a similarity search."""
    score: float
    metadata: Dict[str, Any]
    rank: int
    
    def __repr__(self) -> str:
        doc = self.metadata.get("document", "")[:50]
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, doc='{doc}...')"


class FaissStore:
    """
    Vector store using FAISS for fast similarity search.
    
    FAISS only stores vectors, so metadata is stored in a separate JSON file
    indexed by position (parallel to the FAISS index).
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        use_gpu: bool = False,
    ):
        """
        Initialize the FAISS store.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index ("flat" for exact search, "ivf" for approximate)
            use_gpu: Whether to use GPU acceleration
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Create FAISS index
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU for FAISS")
            except Exception as e:
                logger.warning(f"Failed to use GPU: {e}. Falling back to CPU.")
        
        # Metadata storage (parallel to index)
        self.metadata: List[Dict[str, Any]] = []
        
        logger.info(f"Created FAISS index with dim={embedding_dim}, type={index_type}")
    
    def add(
        self,
        embeddings: np.ndarray,
        items: List[EmbeddableItem],
    ) -> None:
        """
        Add embeddings and their metadata to the store.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            items: List of EmbeddableItem objects (must match embeddings length)
        """
        if len(embeddings) != len(items):
            raise ValueError(
                f"Embeddings length ({len(embeddings)}) must match items length ({len(items)})"
            )
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        for item in items:
            self.metadata.append(item.metadata)
        
        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for not enough results
                continue
            
            results.append(SearchResult(
                score=float(score),
                metadata=self.metadata[idx],
                rank=rank,
            ))
        
        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Query vectors of shape (n_queries, embedding_dim)
            top_k: Number of results per query
            
        Returns:
            List of lists of SearchResult objects
        """
        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
        
        scores, indices = self.index.search(query_embeddings, top_k)
        
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for rank, (score, idx) in enumerate(zip(query_scores, query_indices)):
                if idx < 0:
                    continue
                results.append(SearchResult(
                    score=float(score),
                    metadata=self.metadata[idx],
                    rank=rank,
                ))
            all_results.append(results)
        
        return all_results
    
    def save(self, directory: str, prefix: str = "kg_triplets") -> Tuple[str, str]:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            directory: Directory to save to
            prefix: Filename prefix
            
        Returns:
            Tuple of (index_path, metadata_path)
        """
        os.makedirs(directory, exist_ok=True)
        
        index_path = os.path.join(directory, f"{prefix}.faiss")
        metadata_path = os.path.join(directory, f"{prefix}_meta.json")
        
        # Convert GPU index to CPU for saving if needed
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        
        return index_path, metadata_path
    
    @classmethod
    def load(
        cls,
        directory: str,
        prefix: str = "kg_triplets",
        use_gpu: bool = False,
    ) -> "FaissStore":
        """
        Load a FAISS store from disk.
        
        Args:
            directory: Directory containing saved files
            prefix: Filename prefix
            use_gpu: Whether to load to GPU
            
        Returns:
            Loaded FaissStore instance
        """
        index_path = os.path.join(directory, f"{prefix}.faiss")
        metadata_path = os.path.join(directory, f"{prefix}_meta.json")
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        
        # Load metadata
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Create store instance
        embedding_dim = index.d
        store = cls(embedding_dim=embedding_dim, use_gpu=use_gpu)
        store.index = index
        store.metadata = metadata
        
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                store.index = faiss.index_cpu_to_gpu(res, 0, store.index)
                logger.info("Moved index to GPU")
            except Exception as e:
                logger.warning(f"Failed to use GPU: {e}")
        
        logger.info(f"Loaded {store.index.ntotal} vectors")
        
        return store
    
    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal


if __name__ == "__main__":
    # Simple test
    import tempfile
    
    logging.basicConfig(level=logging.INFO)
    
    # Create store
    store = FaissStore(embedding_dim=384)
    
    # Create dummy data
    n_vectors = 100
    embeddings = np.random.randn(n_vectors, 384).astype(np.float32)
    
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create dummy items
    items = [
        EmbeddableItem(
            text=f"Document {i}",
            metadata={"id": i, "document": f"Document {i}"}
        )
        for i in range(n_vectors)
    ]
    
    # Add to store
    store.add(embeddings, items)
    print(f"Store size: {store.size}")
    
    # Search
    query = embeddings[0]  # Use first vector as query
    results = store.search(query, top_k=5)
    
    print("\nSearch results:")
    for r in results:
        print(f"  {r}")
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir, prefix="test")
        
        loaded_store = FaissStore.load(tmpdir, prefix="test")
        print(f"\nLoaded store size: {loaded_store.size}")
        
        # Verify search works
        results = loaded_store.search(query, top_k=5)
        print("Search results from loaded store:")
        for r in results:
            print(f"  {r}")

