"""
Step 1: Data loader for triplets from EDC pipeline output.

Loads triplets from canon_kg.txt produced by the EDC pipeline.
"""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """Represents a single knowledge graph triplet with metadata."""
    subject: str
    predicate: str
    obj: str  # 'object' is a Python builtin, so we use 'obj'
    source_text: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "source_text": self.source_text,
        }
    
    def __repr__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"


class TripletLoader:
    """
    Loads and parses triplets from EDC pipeline output.
    
    Expected format: canon_kg.txt with one line per input text.
    Each line contains a list of triplets: [['subject', 'predicate', 'object'], ...]
    Empty lines are represented as [].
    """
    
    def __init__(self, path: str):
        """
        Initialize the loader.
        
        Args:
            path: Path to canon_kg.txt or EDC output directory
        """
        self.path = Path(path)
        self.raw_lines: List[str] = []
        self.triplets: List[Triplet] = []
        
    def load(self) -> "TripletLoader":
        """
        Load the canon_kg.txt file.
        
        Returns:
            Self for method chaining
        """
        kg_path = self._resolve_path()
        
        logger.info(f"Loading triplets from {kg_path}")
        
        with open(kg_path, "r", encoding="utf-8") as f:
            self.raw_lines = f.readlines()
        
        logger.info(f"Loaded {len(self.raw_lines)} lines from canon_kg.txt")
        return self
    
    def _resolve_path(self) -> Path:
        """Resolve the path to the actual canon_kg.txt file."""
        # If it's already a file, use it directly
        if self.path.is_file():
            return self.path
        
        # Try to find canon_kg.txt in the directory
        # Look for the latest iteration
        if self.path.is_dir():
            # Find all iter* subdirectories
            iter_dirs = sorted(
                [d for d in self.path.iterdir() if d.is_dir() and d.name.startswith("iter")],
                key=lambda d: int(d.name.replace("iter", "")),
                reverse=True  # Latest first
            )
            
            # Try each iteration directory (latest first)
            for iter_dir in iter_dirs:
                canon_path = iter_dir / "canon_kg.txt"
                if canon_path.exists():
                    logger.info(f"Found canon_kg.txt in {iter_dir.name}")
                    return canon_path
            
            # Try directly in the output directory
            direct_path = self.path / "canon_kg.txt"
            if direct_path.exists():
                return direct_path
        
        raise FileNotFoundError(
            f"Could not find canon_kg.txt in {self.path}. "
            "Make sure you've run the EDC pipeline first."
        )
    
    def parse(self, skip_invalid: bool = True) -> List[Triplet]:
        """
        Parse all triplets from the loaded file.
        
        Args:
            skip_invalid: If True, skip lines with invalid data
            
        Returns:
            List of parsed Triplet objects
        """
        if not self.raw_lines:
            raise ValueError("Must call load() before parse()")
        
        self.triplets = []
        skipped_count = 0
        
        for line_num, line in enumerate(self.raw_lines):
            try:
                triplets_from_line = self._parse_line(line.strip(), line_num)
                self.triplets.extend(triplets_from_line)
            except Exception as e:
                if skip_invalid:
                    skipped_count += 1
                    logger.debug(f"Skipping line {line_num}: {e}")
                else:
                    raise
        
        logger.info(f"Parsed {len(self.triplets)} triplets from {len(self.raw_lines)} lines")
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} invalid lines")
        
        return self.triplets
    
    def _parse_line(self, line: str, line_num: int) -> List[Triplet]:
        """Parse triplets from a single line."""
        triplets = []
        
        # Skip empty lines
        if not line or line == "[]":
            return []
        
        try:
            # Parse the list of triplets
            triplet_list = ast.literal_eval(line)
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse line {line_num}: {line[:50]}...")
            return []
        
        if not isinstance(triplet_list, list):
            return []
        
        for triple in triplet_list:
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            
            subject, predicate, obj = triple
            
            triplets.append(Triplet(
                subject=str(subject),
                predicate=str(predicate),
                obj=str(obj),
                source_text=None,  # Not available in canon_kg.txt
            ))
        
        return triplets
    
    def get_unique_relations(self) -> List[str]:
        """
        Get all unique relations/predicates.
        
        Returns:
            List of unique relation names
        """
        return list(set(t.predicate for t in self.triplets))
    
    def get_triplets_by_subject(self, subject: str) -> List[Triplet]:
        """Get all triplets for a given subject entity."""
        return [t for t in self.triplets if t.subject.lower() == subject.lower()]
    
    def get_triplets_by_relation(self, relation: str) -> List[Triplet]:
        """Get all triplets for a given relation/predicate."""
        return [t for t in self.triplets if t.predicate.lower() == relation.lower()]
    
    @classmethod
    def from_edc_output(cls, output_dir: str) -> "TripletLoader":
        """
        Load triplets from EDC pipeline output directory.
        
        Automatically finds the latest canon_kg.txt.
        
        Args:
            output_dir: Path to EDC output directory (e.g., ./output/tmp)
            
        Returns:
            Loaded and parsed TripletLoader
        """
        loader = cls(output_dir)
        loader.load()
        loader.parse()
        return loader


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python triplet_loader.py <edc_output_dir_or_canon_kg.txt>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    loader = TripletLoader(path)
    triplets = loader.load().parse()
    
    print(f"\nLoaded {len(triplets)} triplets")
    print(f"Unique relations: {len(loader.get_unique_relations())}")
    
    print("\nSample triplets:")
    for t in triplets[:10]:
        print(f"  {t}")
