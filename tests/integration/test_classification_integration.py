"""Integration tests for classification system.

These tests verify that the KeywordClassifier integrates properly with:
- PromptClassification Pydantic model
- TaskType enum system
- Data validation and serialization
"""

import pytest
from typing import Dict, Any

from llm_router.classification import KeywordClassifier
from llm_router.models import PromptClassification
from llm_router.capabilities import TaskType


class TestClassificationIntegration:
    """Test integration between KeywordClassifier and data models."""

    def test_classifier_returns_valid_prompt_classification_model(self):
        """Test that KeywordClassifier returns a valid PromptClassification instance."""
        classifier = KeywordClassifier()
        prompt = "Write a Python function to calculate fibonacci"
        
        result = classifier.classify(prompt)
        
        # Should return a valid PromptClassification instance
        assert isinstance(result, PromptClassification)
        assert result.category == "code"
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.embedding, list)
        assert result.reasoning is not None

    def test_classification_serializes_to_valid_dict(self):
        """Test that PromptClassification from classifier serializes properly."""
        classifier = KeywordClassifier()
        prompt = "Tell me a creative story about dragons"
        
        result = classifier.classify(prompt)
        serialized = result.model_dump()
        
        # Should serialize to a valid dictionary
        assert isinstance(serialized, dict)
        assert "category" in serialized
        assert "confidence" in serialized
        assert "embedding" in serialized
        assert "reasoning" in serialized
        
        # Category should be string value due to use_enum_values=True
        assert serialized["category"] == "creative"
        assert isinstance(serialized["confidence"], float)

    def test_classification_deserializes_from_dict(self):
        """Test that we can recreate PromptClassification from serialized data."""
        classifier = KeywordClassifier()
        prompt = "What is machine learning?"
        
        # Get original result
        original = classifier.classify(prompt)
        serialized = original.model_dump()
        
        # Recreate from serialized data
        recreated = PromptClassification(**serialized)
        
        # Should match original
        assert recreated.category == original.category
        assert recreated.confidence == original.confidence
        assert recreated.embedding == original.embedding
        assert recreated.reasoning == original.reasoning

    def test_classifier_handles_all_valid_task_types(self):
        """Test that classifier can potentially return all valid TaskType categories."""
        classifier = KeywordClassifier()
        
        # Test prompts that should trigger different categories
        test_cases = [
            ("Write a function to sort data", "code"),
            ("Tell me a story about adventure", "creative"), 
            ("What is the capital of France?", "qa"),
        ]
        
        for prompt, expected_category in test_cases:
            result = classifier.classify(prompt)
            
            # Should return valid PromptClassification with expected category
            assert isinstance(result, PromptClassification)
            assert result.category == expected_category
            
            # Verify the category value is a valid TaskType
            # Note: Since use_enum_values=True, category is stored as string
            valid_categories = [task_type.value for task_type in TaskType]
            assert result.category in valid_categories

    def test_classification_validation_enforces_confidence_bounds(self):
        """Test that PromptClassification validates confidence bounds."""
        # Valid confidence values should work
        valid_classification = PromptClassification(
            category="code",
            confidence=0.5,
            embedding=[],
            reasoning="Test"
        )
        assert valid_classification.confidence == 0.5
        
        # Invalid confidence values should raise ValidationError
        with pytest.raises(ValueError):  # Pydantic V2 raises ValueError
            PromptClassification(
                category="code", 
                confidence=1.5,  # Invalid: > 1.0
                embedding=[],
                reasoning="Test"
            )
            
        with pytest.raises(ValueError):  # Pydantic V2 raises ValueError
            PromptClassification(
                category="code",
                confidence=-0.1,  # Invalid: < 0.0
                embedding=[],
                reasoning="Test"
            )

    def test_classifier_output_always_passes_model_validation(self):
        """Test that KeywordClassifier always produces valid PromptClassification objects."""
        classifier = KeywordClassifier()
        
        # Test various types of prompts
        test_prompts = [
            "Write a function",
            "Tell a story", 
            "What is AI?",
            "Random text with no keywords",
            "Multiple function debug algorithm code keywords",
            "",  # Empty prompt
            "A" * 1000,  # Very long prompt
        ]
        
        for prompt in test_prompts:
            result = classifier.classify(prompt)
            
            # Should always return a valid PromptClassification
            assert isinstance(result, PromptClassification)
            
            # All fields should be valid
            assert isinstance(result.category, str)
            assert result.category in [task_type.value for task_type in TaskType]
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.embedding, list)
            assert result.reasoning is not None
            assert isinstance(result.reasoning, str)
