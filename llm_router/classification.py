"""Rule-based classification system for Phase 4.1.

This module provides keyword-based prompt classification that categorizes
prompts into task types without requiring ML or embeddings.
"""

from typing import List
from llm_router.models import PromptClassification


class KeywordClassifier:
    """Simple keyword-based classifier for prompt categorization."""

    def __init__(self):
        """Initialize the keyword classifier."""
        # Keyword patterns for each category
        self._code_keywords = ["function", "debug", "algorithm", "python", "code"]
        self._creative_keywords = ["story", "creative", "imagine", "narrative"]
        self._qa_keywords = ["what", "how", "why", "explain"]

    def classify(self, prompt: str) -> PromptClassification:
        """
        Classify a prompt based on keyword matching.
        
        Args:
            prompt: The input prompt to classify
            
        Returns:
            PromptClassification with category, confidence, and reasoning
        """
        prompt_lower = prompt.lower()
        
        # Count matches for each category
        category_scores = {
            "code": self._count_keyword_matches(prompt_lower, self._code_keywords),
            "creative": self._count_keyword_matches(prompt_lower, self._creative_keywords), 
            "qa": self._count_keyword_matches(prompt_lower, self._qa_keywords)
        }
        
        # Find the category with the highest score
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        if best_score > 0:
            # Calculate confidence based on matches and total keywords in category
            total_keywords = len(getattr(self, f"_{best_category}_keywords"))
            confidence = min(0.9, 0.3 + (best_score / total_keywords) * 0.6)  # Scale from 0.3 to 0.9
            
            # Get matched keywords for reasoning
            matched_keywords = [kw for kw in getattr(self, f"_{best_category}_keywords") if kw in prompt_lower]
            reasoning = f"Matched {best_score} keyword(s): {', '.join(matched_keywords)}"
            
            return PromptClassification(
                category=best_category,
                confidence=confidence,
                embedding=[],  # Empty for rule-based classification
                reasoning=reasoning
            )
        
        # Default fallback - no matches found
        return PromptClassification(
            category="code",  # Default to code for now
            confidence=0.1,
            embedding=[],
            reasoning="No specific keywords detected, using default classification"
        )
    
    def _count_keyword_matches(self, prompt_lower: str, keywords: list[str]) -> int:
        """Count how many keywords from the list are found in the prompt."""
        return sum(1 for keyword in keywords if keyword in prompt_lower)
