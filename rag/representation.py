"""
Step 2: Triplet representation strategies for embedding.

Provides different ways to convert triplets into text for embedding:
- triplet_text: Simple "subject predicate object" format
- entity_context: Group all facts about each entity
"""

import logging
from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from .triplet_loader import Triplet

logger = logging.getLogger(__name__)


class RepresentationMode(Enum):
    """Available representation strategies."""
    TRIPLET_TEXT = "triplet_text"
    ENTITY_CONTEXT = "entity_context"


@dataclass
class EmbeddableItem:
    """
    An item ready to be embedded.
    
    Attributes:
        text: The text to embed
        metadata: Associated metadata for retrieval
    """
    text: str
    metadata: Dict
    
    def __repr__(self) -> str:
        return f"EmbeddableItem(text='{self.text[:50]}...', metadata_keys={list(self.metadata.keys())})"


def humanize(text: str) -> str:
    """
    Convert underscore-separated text to human-readable format.
    
    Examples:
        "John_Doe" -> "John Doe"
        "student_at" -> "student at"
    """
    return text.replace("_", " ")


class TripletRepresenter:
    """
    Converts triplets to embeddable text representations.
    
    Supports two modes:
    - triplet_text: Simple text like "John Doe student at NUS"
    - entity_context: Entity-centric grouping of all facts
    """
    
    def __init__(self, mode: RepresentationMode = RepresentationMode.TRIPLET_TEXT):
        """
        Initialize the representer.
        
        Args:
            mode: The representation strategy to use
        """
        self.mode = mode
        
    def convert(self, triplets: List[Triplet]) -> List[EmbeddableItem]:
        """
        Convert triplets to embeddable items based on the current mode.
        
        Args:
            triplets: List of Triplet objects
            
        Returns:
            List of EmbeddableItem objects ready for embedding
        """
        if self.mode == RepresentationMode.TRIPLET_TEXT:
            return self._convert_triplet_text(triplets)
        elif self.mode == RepresentationMode.ENTITY_CONTEXT:
            return self._convert_entity_context(triplets)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _convert_triplet_text(self, triplets: List[Triplet]) -> List[EmbeddableItem]:
        """
        Convert each triplet to a simple text sentence.
        
        Example: (John_Doe, student_at, NUS) -> "John Doe student at NUS"
        """
        items = []
        
        for triplet in triplets:
            text = f"{humanize(triplet.subject)} {humanize(triplet.predicate)} {humanize(triplet.obj)}"
            
            metadata = {
                "subject": triplet.subject,
                "predicate": triplet.predicate,
                "object": triplet.obj,
                "source_text": triplet.source_text or "",
                "document": f"({triplet.subject}, {triplet.predicate}, {triplet.obj})",
                "representation_mode": "triplet_text",
            }
            
            items.append(EmbeddableItem(text=text, metadata=metadata))
        
        logger.info(f"Created {len(items)} triplet_text representations")
        return items
    
    def _convert_entity_context(self, triplets: List[Triplet]) -> List[EmbeddableItem]:
        """
        Group all facts about each entity and create a context string.
        
        Example for entity "John_Doe":
            "John Doe: student at NUS, born in Singapore, occupation Researcher"
        """
        # Group triplets by subject entity
        entity_facts: Dict[str, List[str]] = defaultdict(list)
        entity_triplets: Dict[str, List[Triplet]] = defaultdict(list)
        
        for triplet in triplets:
            fact = f"{humanize(triplet.predicate)}: {humanize(triplet.obj)}"
            entity_facts[triplet.subject].append(fact)
            entity_triplets[triplet.subject].append(triplet)
        
        items = []
        
        for entity, facts in entity_facts.items():
            # Combine all facts into a context string
            facts_str = ", ".join(facts)
            text = f"{humanize(entity)}: {facts_str}"
            
            # Collect all triplets for this entity
            triplets_list = [
                f"({t.subject}, {t.predicate}, {t.obj})" 
                for t in entity_triplets[entity]
            ]
            
            metadata = {
                "entity": entity,
                "fact_count": len(facts),
                "triplets": triplets_list,
                "document": text,
                "representation_mode": "entity_context",
            }
            
            items.append(EmbeddableItem(text=text, metadata=metadata))
        
        logger.info(f"Created {len(items)} entity_context representations")
        return items


def get_representer(mode: str) -> TripletRepresenter:
    """
    Factory function to get a TripletRepresenter by mode name.
    
    Args:
        mode: One of "triplet_text" or "entity_context"
        
    Returns:
        Configured TripletRepresenter
    """
    try:
        representation_mode = RepresentationMode(mode)
    except ValueError:
        valid_modes = [m.value for m in RepresentationMode]
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
    
    return TripletRepresenter(mode=representation_mode)


if __name__ == "__main__":
    from .triplet_loader import TripletLoader
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python representation.py <edc_output_dir> [mode]")
        sys.exit(1)
    
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "triplet_text"
    
    # Load triplets
    loader = TripletLoader(path)
    triplets = loader.load().parse()
    
    # Convert to embeddings
    representer = get_representer(mode)
    items = representer.convert(triplets)
    
    print(f"\nCreated {len(items)} embeddable items using '{mode}' mode")
    print("\nSample items:")
    for item in items[:5]:
        print(f"  Text: {item.text[:100]}...")
        print(f"  Metadata: {item.metadata}")
        print()
