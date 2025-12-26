"""
Step 5: High-level retrieval interface for KG-RAG.

Wraps KGRagIndexer.search() with formatted output for prompt augmentation.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .kg_rag_indexer import KGRagIndexer
from .faiss_store import SearchResult

logger = logging.getLogger(__name__)


class ContextFormat(Enum):
    """Available formats for context output."""
    TRIPLET_LIST = "triplet_list"      # (subj, pred, obj) per line
    NATURAL = "natural"                 # "subj pred obj" sentences
    TABLE = "table"                     # Markdown table format


@dataclass
class RetrievalResult:
    """
    Result from KG retrieval.
    
    Attributes:
        triplets: List of (subject, predicate, object) tuples
        formatted_context: String formatted for prompt insertion
        scores: Relevance scores for each triplet
        raw_results: Original SearchResult objects
    """
    triplets: List[Tuple[str, str, str]]
    formatted_context: str
    scores: List[float]
    raw_results: List[SearchResult] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __bool__(self) -> bool:
        return len(self.triplets) > 0


class KGRetriever:
    """
    High-level retrieval interface for KG triplets.
    
    Wraps KGRagIndexer.search() and formats results for prompt augmentation.
    
    Usage:
        retriever = KGRetriever(indexer)
        result = retriever.retrieve("Where is Trane located?", top_k=5)
        print(result.formatted_context)
    """
    
    def __init__(
        self,
        indexer: KGRagIndexer,
        default_format: ContextFormat = ContextFormat.TRIPLET_LIST,
    ):
        """
        Initialize the retriever.
        
        Args:
            indexer: A loaded KGRagIndexer instance
            default_format: Default format for context output
        """
        self.indexer = indexer
        self.default_format = default_format
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        format: Optional[ContextFormat] = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """
        Retrieve relevant triplets for a query.
        
        Args:
            query: Natural language query
            top_k: Maximum number of results
            format: Output format (defaults to self.default_format)
            min_score: Minimum relevance score threshold
            
        Returns:
            RetrievalResult with triplets and formatted context
        """
        format = format or self.default_format
        
        # Search using the indexer
        search_results = self.indexer.search(query, top_k=top_k)
        
        # Filter by minimum score
        if min_score > 0:
            search_results = [r for r in search_results if r.score >= min_score]
        
        # Extract triplets from metadata (may return multiple per result for entity_context mode)
        triplets = []
        scores = []
        
        for result in search_results:
            extracted = self._extract_triplets(result)
            for triplet in extracted:
                triplets.append(triplet)
                scores.append(result.score)  # Same score for all triplets from same result
        
        # Format context
        formatted_context = self._format_context(triplets, scores, format)
        
        logger.debug(f"Retrieved {len(triplets)} triplets for query: {query[:50]}...")
        
        return RetrievalResult(
            triplets=triplets,
            formatted_context=formatted_context,
            scores=scores,
            raw_results=search_results,
        )
    
    def _extract_triplets(self, result: SearchResult) -> List[Tuple[str, str, str]]:
        """
        Extract all (subject, predicate, object) tuples from search result.
        
        For triplet_text mode: returns a single triplet
        For entity_context mode: returns ALL triplets associated with the entity
        """
        meta = result.metadata
        
        # Handle triplet_text mode - single triplet
        if "subject" in meta and "predicate" in meta and "object" in meta:
            return [(meta["subject"], meta["predicate"], meta["object"])]
        
        # Handle entity_context mode - extract ALL triplets from list
        if "triplets" in meta and isinstance(meta["triplets"], list):
            parsed = []
            for triplet_str in meta["triplets"]:
                triplet = self._parse_triplet_string(triplet_str)
                if triplet:
                    parsed.append(triplet)
            if parsed:
                return parsed
        
        # Fallback: try to extract from document field (single triplet)
        if "document" in meta:
            triplet = self._parse_triplet_string(meta["document"])
            if triplet:
                return [triplet]
        
        return []
    
    def _parse_triplet_string(self, triplet_str: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse a triplet string like "(subj, pred, obj)" into a tuple.
        
        Returns None if parsing fails.
        """
        if not triplet_str:
            return None
        
        try:
            # Remove parentheses and split by comma
            clean = triplet_str.strip()
            if clean.startswith("(") and clean.endswith(")"):
                clean = clean[1:-1]
            
            parts = [p.strip() for p in clean.split(",")]
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        except Exception:
            pass
        
        return None
    
    def _format_context(
        self,
        triplets: List[Tuple[str, str, str]],
        scores: List[float],
        format: ContextFormat,
    ) -> str:
        """Format triplets as a string for prompt insertion."""
        if not triplets:
            return "No relevant facts found."
        
        if format == ContextFormat.TRIPLET_LIST:
            return self._format_triplet_list(triplets)
        elif format == ContextFormat.NATURAL:
            return self._format_natural(triplets)
        elif format == ContextFormat.TABLE:
            return self._format_table(triplets, scores)
        else:
            return self._format_triplet_list(triplets)
    
    def _format_triplet_list(self, triplets: List[Tuple[str, str, str]]) -> str:
        """Format as list of (subject, predicate, object) triplets."""
        lines = []
        for subj, pred, obj in triplets:
            # Humanize underscores
            subj_h = subj.replace("_", " ")
            pred_h = pred.replace("_", " ")
            obj_h = obj.replace("_", " ")
            lines.append(f"({subj_h}, {pred_h}, {obj_h})")
        return "\n".join(lines)
    
    def _format_natural(self, triplets: List[Tuple[str, str, str]]) -> str:
        """Format as natural language sentences."""
        lines = []
        for subj, pred, obj in triplets:
            # Humanize underscores
            subj_h = subj.replace("_", " ")
            pred_h = pred.replace("_", " ")
            obj_h = obj.replace("_", " ")
            lines.append(f"{subj_h} {pred_h} {obj_h}.")
        return "\n".join(lines)
    
    def _format_table(
        self,
        triplets: List[Tuple[str, str, str]],
        scores: List[float],
    ) -> str:
        """Format as markdown table."""
        lines = ["| Subject | Predicate | Object | Score |", "|---------|-----------|--------|-------|"]
        for (subj, pred, obj), score in zip(triplets, scores):
            subj_h = subj.replace("_", " ")
            pred_h = pred.replace("_", " ")
            obj_h = obj.replace("_", " ")
            lines.append(f"| {subj_h} | {pred_h} | {obj_h} | {score:.3f} |")
        return "\n".join(lines)
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        format: Optional[ContextFormat] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve triplets for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Maximum results per query
            format: Output format
            
        Returns:
            List of RetrievalResult objects
        """
        return [self.retrieve(q, top_k, format) for q in queries]


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage: python retriever.py <index_dir> <query>")
        sys.exit(1)
    
    index_dir = sys.argv[1]
    query = sys.argv[2]
    
    # Load indexer
    indexer = KGRagIndexer.load(index_dir)
    
    # Create retriever
    retriever = KGRetriever(indexer)
    
    # Retrieve
    result = retriever.retrieve(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(result)} triplets\n")
    print("=== Formatted Context ===")
    print(result.formatted_context)
    print("\n=== Triplets ===")
    for (s, p, o), score in zip(result.triplets, result.scores):
        print(f"  [{score:.3f}] ({s}, {p}, {o})")

