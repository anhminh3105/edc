"""
Main KG-RAG Indexer that orchestrates the full pipeline.

Combines:
1. TripletLoader - Load and parse triplets from canon_kg.txt
2. TripletRepresenter - Convert to embeddable format
3. Embedder - Generate embeddings
4. FaissStore - Store and retrieve
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any

from .triplet_loader import TripletLoader, Triplet
from .representation import TripletRepresenter, RepresentationMode, EmbeddableItem, get_representer
from .embedder import Embedder, get_embedder, DEFAULT_MODEL
from .faiss_store import FaissStore, SearchResult

logger = logging.getLogger(__name__)


class KGRagIndexer:
    """
    Knowledge Graph RAG Indexer.
    
    Full pipeline for indexing KG triplets for retrieval:
    1. Load triplets from EDC pipeline output (canon_kg.txt)
    2. Convert to embeddable representations
    3. Generate embeddings
    4. Store in FAISS index
    
    Usage:
        indexer = KGRagIndexer()
        indexer.index_from_path("./output/tmp", mode="triplet_text")
        indexer.save("./output/rag")
        
        # Later, for retrieval:
        indexer = KGRagIndexer.load("./output/rag")
        results = indexer.search("Where is Trane located?")
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
        use_gpu_faiss: bool = False,
    ):
        """
        Initialize the indexer.
        
        Args:
            embedding_model: Sentence transformer model name
            device: Device for embeddings ("cuda", "cpu", or None for auto)
            use_gpu_faiss: Whether to use GPU for FAISS
        """
        self.embedding_model_name = embedding_model or DEFAULT_MODEL
        self.device = device
        self.use_gpu_faiss = use_gpu_faiss
        
        # Components (lazy-loaded)
        self._embedder: Optional[Embedder] = None
        self._store: Optional[FaissStore] = None
        
        # State
        self.triplets: List[Triplet] = []
        self.items: List[EmbeddableItem] = []
        self.representation_mode: Optional[str] = None
        
    @property
    def embedder(self) -> Embedder:
        """Lazy-load the embedder."""
        if self._embedder is None:
            self._embedder = get_embedder(
                model_name=self.embedding_model_name,
                device=self.device,
            )
        return self._embedder
    
    @property
    def store(self) -> FaissStore:
        """Get the FAISS store (must be initialized first)."""
        if self._store is None:
            raise ValueError("Store not initialized. Call index_from_path() or load() first.")
        return self._store
    
    def index_from_path(
        self,
        path: str,
        mode: str = "triplet_text",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> "KGRagIndexer":
        """
        Index triplets from EDC pipeline output (canon_kg.txt).
        
        Automatically finds the latest canon_kg.txt in the output directory.
        
        Args:
            path: Path to canon_kg.txt or EDC output directory
            mode: Representation mode ("triplet_text" or "entity_context")
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bars
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting indexing pipeline for {path}")
        logger.info(f"Using mode: {mode}")
        
        self.representation_mode = mode
        
        # Step 1: Load triplets from canon_kg.txt
        logger.info("Step 1: Loading triplets from canon_kg.txt...")
        loader = TripletLoader(path)
        loader.load()
        self.triplets = loader.parse()
        logger.info(f"Loaded {len(self.triplets)} triplets")
        
        if len(self.triplets) == 0:
            raise ValueError("No triplets found. Check your input file/directory.")
        
        # Step 2: Convert to representations
        logger.info("Step 2: Converting to representations...")
        representer = get_representer(mode)
        self.items = representer.convert(self.triplets)
        logger.info(f"Created {len(self.items)} embeddable items")
        
        if len(self.items) == 0:
            raise ValueError(
                "No embeddable items created. "
                "Use 'triplet_text' or 'entity_context' mode."
            )
        
        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        embeddings = self.embedder.embed_items(
            self.items,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True,
        )
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        # Step 4: Store in FAISS
        logger.info("Step 4: Storing in FAISS index...")
        self._store = FaissStore(
            embedding_dim=self.embedder.embedding_dim,
            use_gpu=self.use_gpu_faiss,
        )
        self._store.add(embeddings, self.items)
        logger.info(f"Indexed {self._store.size} vectors")
        
        logger.info("Indexing complete!")
        return self
    
    def index_triplets(
        self,
        triplets: List[Triplet],
        mode: str = "triplet_text",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> "KGRagIndexer":
        """
        Index triplets directly (without loading from file).
        
        Args:
            triplets: List of Triplet objects
            mode: Representation mode
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bars
            
        Returns:
            Self for method chaining
        """
        self.representation_mode = mode
        self.triplets = triplets
        
        # Step 2: Convert to representations
        representer = get_representer(mode)
        self.items = representer.convert(self.triplets)
        
        # Step 3: Generate embeddings
        embeddings = self.embedder.embed_items(
            self.items,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True,
        )
        
        # Step 4: Store in FAISS
        self._store = FaissStore(
            embedding_dim=self.embedder.embedding_dim,
            use_gpu=self.use_gpu_faiss,
        )
        self._store.add(embeddings, self.items)
        
        return self
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_instruction: bool = True,
    ) -> List[SearchResult]:
        """
        Search for relevant triplets given a query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            use_instruction: Whether to use instruction prefix for query
            
        Returns:
            List of SearchResult objects
        """
        # Embed query
        query_embedding = self.embedder.embed_query(
            query,
            use_instruction=use_instruction,
            normalize=True,
        )
        
        # Search
        results = self.store.search(query_embedding, top_k=top_k)
        
        return results
    
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        use_instruction: bool = True,
    ) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            use_instruction: Whether to use instruction prefix
            
        Returns:
            List of lists of SearchResult objects
        """
        # Embed queries
        query_embeddings = self.embedder.embed_queries(
            queries,
            use_instruction=use_instruction,
            normalize=True,
        )
        
        # Batch search
        results = self.store.search_batch(query_embeddings, top_k=top_k)
        
        return results
    
    def save(self, directory: str, prefix: str = "kg_triplets") -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            directory: Output directory
            prefix: Filename prefix
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index and metadata
        self.store.save(directory, prefix=prefix)
        
        # Save config
        config = {
            "embedding_model": self.embedding_model_name,
            "representation_mode": self.representation_mode,
            "num_triplets": len(self.triplets),
            "num_items": len(self.items),
            "embedding_dim": self.embedder.embedding_dim,
        }
        
        config_path = os.path.join(directory, f"{prefix}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved index to {directory}")
    
    @classmethod
    def load(
        cls,
        directory: str,
        prefix: str = "kg_triplets",
        device: Optional[str] = None,
        use_gpu_faiss: bool = False,
    ) -> "KGRagIndexer":
        """
        Load a saved index.
        
        Args:
            directory: Directory containing saved files
            prefix: Filename prefix
            device: Device for embeddings
            use_gpu_faiss: Whether to use GPU for FAISS
            
        Returns:
            Loaded KGRagIndexer instance
        """
        # Load config
        config_path = os.path.join(directory, f"{prefix}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create indexer
        indexer = cls(
            embedding_model=config.get("embedding_model"),
            device=device,
            use_gpu_faiss=use_gpu_faiss,
        )
        
        indexer.representation_mode = config.get("representation_mode")
        
        # Load FAISS store
        indexer._store = FaissStore.load(
            directory,
            prefix=prefix,
            use_gpu=use_gpu_faiss,
        )
        
        logger.info(f"Loaded index with {indexer.store.size} vectors")
        
        return indexer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data."""
        return {
            "num_triplets": len(self.triplets),
            "num_items": len(self.items),
            "num_vectors": self.store.size if self._store else 0,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedder.embedding_dim if self._embedder else None,
            "representation_mode": self.representation_mode,
        }
