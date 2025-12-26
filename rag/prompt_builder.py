"""
Step 6: Prompt builder for KG-augmented question answering.

Creates prompts by combining retrieved KG facts with user queries.
"""

import os
import logging
from typing import Optional
from pathlib import Path

from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Default template path
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "prompt_templates" / "kg_qa.txt"

# Fallback template if file not found
FALLBACK_TEMPLATE = """You are a knowledge graph question answering assistant.

=== KNOWLEDGE GRAPH FACTS ===
{facts}

=== QUESTION ===
{question}

Answer using only the facts above. If the answer is not in the facts, say "I cannot find this information."

=== ANSWER ===
"""


class KGPromptBuilder:
    """
    Builds prompts with KG context for LLM generation.
    
    Combines retrieved triplets with the user's question using a template.
    
    Usage:
        builder = KGPromptBuilder()
        prompt = builder.build(query, retrieval_result)
    """
    
    def __init__(
        self,
        template_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the prompt builder.
        
        Args:
            template_path: Path to prompt template file
            system_prompt: Optional system prompt for chat models
        """
        self.template = self._load_template(template_path)
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _load_template(self, template_path: Optional[str] = None) -> str:
        """Load the prompt template from file."""
        path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                template = f.read()
            logger.debug(f"Loaded template from {path}")
            return template
        except FileNotFoundError:
            logger.warning(f"Template not found at {path}, using fallback")
            return FALLBACK_TEMPLATE
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for chat models."""
        return (
            "You are a helpful assistant that answers questions based on knowledge graph facts. "
            "Only use the provided facts to answer. Be concise and accurate."
        )
    
    def build(
        self,
        query: str,
        context: RetrievalResult,
        include_scores: bool = False,
    ) -> str:
        """
        Build an augmented prompt with retrieved KG facts.
        
        Args:
            query: User's question
            context: Retrieved KG facts
            include_scores: Whether to include relevance scores in facts
            
        Returns:
            Formatted prompt string
        """
        # Format facts
        if include_scores:
            facts = self._format_facts_with_scores(context)
        else:
            facts = context.formatted_context
        
        # Handle empty context
        if not context:
            facts = "No relevant facts were found in the knowledge graph."
        
        # Fill template
        prompt = self.template.format(
            facts=facts,
            question=query,
        )
        
        return prompt
    
    def build_chat_messages(
        self,
        query: str,
        context: RetrievalResult,
        include_scores: bool = False,
    ) -> list:
        """
        Build chat messages format for LLM APIs.
        
        Args:
            query: User's question
            context: Retrieved KG facts
            include_scores: Whether to include relevance scores
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        # Format facts
        if include_scores:
            facts = self._format_facts_with_scores(context)
        else:
            facts = context.formatted_context
        
        if not context:
            facts = "No relevant facts were found in the knowledge graph."
        
        # Build user message with facts and question
        user_content = f"""Use the following knowledge graph facts to answer my question.

=== KNOWLEDGE GRAPH FACTS ===
{facts}

=== MY QUESTION ===
{query}

Please answer based only on the facts provided above."""
        
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        return messages
    
    def _format_facts_with_scores(self, context: RetrievalResult) -> str:
        """Format facts including relevance scores."""
        if not context.triplets:
            return "No relevant facts found."
        
        lines = []
        for (subj, pred, obj), score in zip(context.triplets, context.scores):
            subj_h = subj.replace("_", " ")
            pred_h = pred.replace("_", " ")
            obj_h = obj.replace("_", " ")
            lines.append(f"[{score:.2f}] ({subj_h}, {pred_h}, {obj_h})")
        
        return "\n".join(lines)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for chat models."""
        return self.system_prompt
    
    def set_template(self, template: str) -> None:
        """Set a custom template string."""
        self.template = template
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set a custom system prompt."""
        self.system_prompt = system_prompt


def get_prompt_builder(
    template_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> KGPromptBuilder:
    """
    Factory function to create a KGPromptBuilder.
    
    Args:
        template_path: Optional custom template path
        system_prompt: Optional custom system prompt
        
    Returns:
        Configured KGPromptBuilder instance
    """
    return KGPromptBuilder(
        template_path=template_path,
        system_prompt=system_prompt,
    )


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Demo with mock data
    from .retriever import RetrievalResult
    
    # Create mock retrieval result
    mock_result = RetrievalResult(
        triplets=[
            ("Trane", "location", "Swords_Dublin"),
            ("Alan_B._Miller_Hall", "country", "Virginia"),
        ],
        formatted_context="(Trane, location, Swords Dublin)\n(Alan B. Miller Hall, country, Virginia)",
        scores=[0.92, 0.85],
        raw_results=[],
    )
    
    # Build prompt
    builder = KGPromptBuilder()
    prompt = builder.build("Where is Trane located?", mock_result)
    
    print("=== Generated Prompt ===")
    print(prompt)
    
    print("\n=== Chat Messages ===")
    messages = builder.build_chat_messages("Where is Trane located?", mock_result)
    for msg in messages:
        print(f"[{msg['role']}]: {msg['content'][:200]}...")


