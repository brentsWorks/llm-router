"""Tests for Phase 4.1: Rule-Based Classification.

This module tests the keyword-based classification system that categorizes
prompts into task types for optimal model selection.
"""

import pytest
from llm_router.models import PromptClassification


class TestKeywordClassifier:
    """Test keyword-based classification system."""

    def test_should_classify_code_prompt_with_function_keyword(self):
        """Test that prompts containing 'function' are classified as 'code'."""
        from llm_router.classification import KeywordClassifier
        
        classifier = KeywordClassifier()
        prompt = "Write a Python function to calculate fibonacci numbers"
        
        result = classifier.classify(prompt)
        
        assert result.category == "code"
        assert result.confidence > 0.0
        assert "function" in result.reasoning.lower()

    def test_should_classify_creative_prompt_with_story_keyword(self):
        """Test that prompts containing 'story' are classified as 'creative'."""
        from llm_router.classification import KeywordClassifier
        
        classifier = KeywordClassifier()
        prompt = "Write a short story about a brave knight"
        
        result = classifier.classify(prompt)
        
        assert result.category == "creative"
        assert result.confidence > 0.0
        assert "story" in result.reasoning.lower()

    def test_should_classify_qa_prompt_with_what_keyword(self):
        """Test that prompts containing 'what' are classified as 'qa'."""
        from llm_router.classification import KeywordClassifier
        
        classifier = KeywordClassifier()
        prompt = "What is the capital of France?"
        
        result = classifier.classify(prompt)
        
        assert result.category == "qa"
        assert result.confidence > 0.0
        assert "what" in result.reasoning.lower()

    def test_should_calculate_higher_confidence_for_multiple_keyword_matches(self):
        """Test that multiple keyword matches result in higher confidence."""
        from llm_router.classification import KeywordClassifier
        
        classifier = KeywordClassifier()
        
        # Single keyword match
        single_match_prompt = "Write a function to sort data"
        single_result = classifier.classify(single_match_prompt)
        
        # Multiple keyword matches (we'll need to add more keywords)
        multiple_match_prompt = "Write a Python function to debug algorithm issues"
        multiple_result = classifier.classify(multiple_match_prompt)
        
        # Multiple matches should have higher confidence
        assert multiple_result.confidence > single_result.confidence
        assert multiple_result.category == "code"
        assert single_result.category == "code"

    def test_should_have_low_confidence_for_no_keyword_matches(self):
        """Test that prompts with no keyword matches have low confidence."""
        from llm_router.classification import KeywordClassifier
        
        classifier = KeywordClassifier()
        prompt = "The weather is nice today"  # No keywords from any category
        
        result = classifier.classify(prompt)
        
        # Should default to low confidence
        assert result.confidence < 0.5
        assert result.confidence > 0.0  # But not zero

    def test_confidence_should_be_bounded_between_zero_and_one(self):
        """Test that confidence values are always between 0.0 and 1.0."""
        from llm_router.classification import KeywordClassifier
        
        classifier = KeywordClassifier()
        prompts = [
            "Write a function",
            "Tell me a story",
            "What is Python?",
            "Random text with no keywords"
        ]
        
        for prompt in prompts:
            result = classifier.classify(prompt)
            assert 0.0 <= result.confidence <= 1.0, f"Confidence {result.confidence} out of bounds for prompt: {prompt}"
