"""
Unit tests for RAG-enhanced classification system.

This module tests the RAG classifier that combines semantic similarity search
with LLM-based classification using Gemini Pro.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from llm_router.rag_classification import RAGClassifier, RAGClassificationError
from llm_router.models import PromptClassification
from llm_router.dataset import ExamplePrompt, PromptCategory
from llm_router.vector_stores import SearchResult


class TestRAGClassifier:
    """Test RAG classifier functionality."""

    @pytest.fixture
    def mock_vector_service(self):
        """Create a mock vector service."""
        mock_service = MagicMock()
        mock_service.find_similar_examples.return_value = []
        return mock_service

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock Gemini client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "recommended_models": ["gpt-4", "codex"], "reasoning": "Based on similar examples"}'
        mock_client.generate_content.return_value = mock_response
        return mock_client

    @pytest.fixture
    def sample_similar_examples(self):
        """Create sample similar examples for testing."""
        return [
            SearchResult(
                vector=np.array([0.1] * 384),
                similarity=0.9,
                index=0,
                metadata={
                    "text": "Write a Python function to sort data",
                    "category": "code",
                    "preferred_models": ["gpt-4", "codex"],
                    "difficulty": "medium",
                    "domain": "programming"
                }
            ),
            SearchResult(
                vector=np.array([0.2] * 384),
                similarity=0.8,
                index=1,
                metadata={
                    "text": "Create a JavaScript function for validation",
                    "category": "code", 
                    "preferred_models": ["codex", "gpt-4"],
                    "difficulty": "easy",
                    "domain": "programming"
                }
            )
        ]

    def test_should_initialize_rag_classifier_with_dependencies(self, mock_vector_service, mock_gemini_client):
        """Test that RAG classifier initializes with required dependencies."""
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        assert classifier.vector_service == mock_vector_service
        assert classifier.gemini_client == mock_gemini_client
        assert classifier.api_key == "test-key"
        assert classifier.confidence_threshold == 0.7  # Default value
        assert classifier.max_similar_examples == 3  # Default value

    def test_should_require_api_key_for_initialization(self, mock_vector_service, mock_gemini_client):
        """Test that API key is required for initialization."""
        with pytest.raises(RAGClassificationError) as exc_info:
            RAGClassifier(
                vector_service=mock_vector_service,
                gemini_client=mock_gemini_client,
                api_key=""
            )
        
        assert "API key is required" in str(exc_info.value)

    def test_should_classify_prompt_using_similar_examples(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test that classifier uses similar examples for classification."""
        # Setup mocks
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "recommended_models": ["gpt-4", "codex"], "reasoning": "Based on programming examples with high similarity"}'
        mock_gemini_client.generate_content.return_value = mock_response
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        # Classify prompt
        result = classifier.classify("Write a Python function to calculate fibonacci")
        
        # Verify vector service was called
        mock_vector_service.find_similar_examples.assert_called_once_with(
            "Write a Python function to calculate fibonacci",
            k=3
        )
        
        # Verify Gemini was called with proper prompt
        mock_gemini_client.generate_content.assert_called_once()
        call_args = mock_gemini_client.generate_content.call_args[0][0]
        assert "Write a Python function to calculate fibonacci" in call_args
        assert "Write a Python function to sort data" in call_args  # Similar example
        
        # Verify result
        assert isinstance(result, PromptClassification)
        assert result.category == "code"
        assert result.confidence == 0.85
        assert hasattr(result, '_recommended_models')
        assert "gpt-4" in getattr(result, '_recommended_models', [])
        assert "Based on programming examples" in result.reasoning

    def test_should_handle_no_similar_examples_found(self, mock_vector_service, mock_gemini_client):
        """Test handling when no similar examples are found."""
        # Setup mocks - no similar examples
        mock_vector_service.find_similar_examples.return_value = []
        mock_response = MagicMock()
        mock_response.text = '{"category": "qa", "confidence": 0.6, "recommended_models": ["gpt-3.5", "gpt-4"], "reasoning": "No similar examples found, general classification"}'
        mock_gemini_client.generate_content.return_value = mock_response
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        result = classifier.classify("What is the weather like?")
        
        # Should still work but with lower confidence
        assert isinstance(result, PromptClassification)
        assert result.category == "qa"
        assert result.confidence == 0.6
        assert "No similar examples found" in result.reasoning

    def test_should_validate_prompt_input(self, mock_vector_service, mock_gemini_client):
        """Test that prompt input is validated."""
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        # Test empty prompt
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("")
        assert "Prompt cannot be empty" in str(exc_info.value)
        
        # Test None prompt
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify(None)
        assert "Prompt cannot be None" in str(exc_info.value)
        
        # Test whitespace-only prompt
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("   ")
        assert "Prompt cannot be empty" in str(exc_info.value)

    def test_should_handle_gemini_api_errors(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test handling of Gemini API errors."""
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        mock_gemini_client.generate_content.side_effect = Exception("API rate limit exceeded")
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Gemini API error" in str(exc_info.value)
        assert "API rate limit exceeded" in str(exc_info.value)

    def test_should_handle_invalid_gemini_response_format(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test handling of invalid JSON response from Gemini."""
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON response"
        mock_gemini_client.generate_content.return_value = mock_response
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Invalid response format from Gemini" in str(exc_info.value)

    def test_should_validate_gemini_response_fields(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test validation of required fields in Gemini response."""
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        # Missing required fields
        mock_response = MagicMock()
        mock_response.text = '{"category": "code"}'  # Missing confidence, reasoning
        mock_gemini_client.generate_content.return_value = mock_response
        
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Missing required fields in Gemini response" in str(exc_info.value)

    def test_should_handle_vector_service_errors(self, mock_vector_service, mock_gemini_client):
        """Test handling of vector service errors."""
        mock_vector_service.find_similar_examples.side_effect = Exception("Vector search failed")
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Vector search error" in str(exc_info.value)
        assert "Vector search failed" in str(exc_info.value)

    def test_should_use_configurable_parameters(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test that configurable parameters are used correctly."""
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "recommended_models": ["gpt-4"], "reasoning": "Test"}'
        mock_gemini_client.generate_content.return_value = mock_response
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key",
            confidence_threshold=0.8,
            max_similar_examples=5
        )
        
        classifier.classify("Test prompt")
        
        # Verify custom parameters were used
        mock_vector_service.find_similar_examples.assert_called_once_with(
            "Test prompt",
            k=5  # Custom max_similar_examples
        )
        assert classifier.confidence_threshold == 0.8

    def test_should_format_similar_examples_for_gemini_prompt(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test that similar examples are properly formatted for Gemini prompt."""
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "recommended_models": ["gpt-4"], "reasoning": "Test"}'
        mock_gemini_client.generate_content.return_value = mock_response
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        classifier.classify("Test prompt")
        
        # Check that the prompt includes formatted similar examples
        call_args = mock_gemini_client.generate_content.call_args[0][0]
        assert "SIMILAR EXAMPLES:" in call_args
        assert "Write a Python function to sort data" in call_args
        assert "similarity: 0.9" in call_args
        assert "Preferred Models: ['gpt-4', 'codex']" in call_args

    def test_should_return_classification_with_proper_structure(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test that returned classification has proper structure."""
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "recommended_models": ["gpt-4", "codex"], "reasoning": "Based on similar programming examples"}'
        mock_gemini_client.generate_content.return_value = mock_response
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        result = classifier.classify("Write a Python function")
        
        # Verify PromptClassification structure
        assert isinstance(result, PromptClassification)
        assert result.category == "code"
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.embedding, list)  # Should be empty for RAG
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
        
        # Verify recommended models are stored as attribute
        assert hasattr(result, '_recommended_models')
        recommended_models = getattr(result, '_recommended_models', [])
        assert isinstance(recommended_models, list)
        assert "gpt-4" in recommended_models

    def test_should_handle_confidence_bounds_validation(self, mock_vector_service, mock_gemini_client, sample_similar_examples):
        """Test that confidence values are properly bounded."""
        mock_vector_service.find_similar_examples.return_value = sample_similar_examples
        
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        # Test confidence > 1.0
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 1.5, "recommended_models": ["gpt-4"], "reasoning": "Test"}'
        mock_gemini_client.generate_content.return_value = mock_response
        
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Invalid confidence value" in str(exc_info.value)
        
        # Test confidence < 0.0
        mock_response.text = '{"category": "code", "confidence": -0.1, "recommended_models": ["gpt-4"], "reasoning": "Test"}'
        
        with pytest.raises(RAGClassificationError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Invalid confidence value" in str(exc_info.value)


class TestRAGClassifierIntegration:
    """Integration tests for RAG classifier with real-like scenarios."""

    @pytest.fixture
    def mock_vector_service(self):
        """Create a mock vector service."""
        mock_service = MagicMock()
        mock_service.find_similar_examples.return_value = []
        return mock_service

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock Gemini client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "recommended_models": ["gpt-4", "codex"], "reasoning": "Based on similar examples"}'
        mock_client.generate_content.return_value = mock_response
        return mock_client

    def test_should_classify_different_prompt_types(self, mock_vector_service, mock_gemini_client):
        """Test classification of different types of prompts."""
        classifier = RAGClassifier(
            vector_service=mock_vector_service,
            gemini_client=mock_gemini_client,
            api_key="test-key"
        )
        
        test_cases = [
            {
                "prompt": "Write a Python function to sort a list",
                "expected_category": "code",
                "similar_examples": [
                    SearchResult(
                        vector=np.array([0.1] * 384),
                        similarity=0.9,
                        index=0,
                        metadata={
                            "text": "Generate a Python function to calculate factorial",
                            "category": "code",
                            "preferred_models": ["codex", "gpt-4"]
                        }
                    )
                ]
            },
            {
                "prompt": "Tell me a story about dragons",
                "expected_category": "creative",
                "similar_examples": [
                    SearchResult(
                        vector=np.array([0.2] * 384),
                        similarity=0.85,
                        index=0,
                        metadata={
                            "text": "Write a creative short story about robots",
                            "category": "creative",
                            "preferred_models": ["gpt-4", "claude-3"]
                        }
                    )
                ]
            }
        ]
        
        for case in test_cases:
            mock_vector_service.find_similar_examples.return_value = case["similar_examples"]
            mock_response = MagicMock()
            mock_response.text = f'{{"category": "{case["expected_category"]}", "confidence": 0.8, "recommended_models": ["gpt-4"], "reasoning": "Based on similar examples"}}'
            mock_gemini_client.generate_content.return_value = mock_response
            
            result = classifier.classify(case["prompt"])
            
            assert result.category == case["expected_category"]
            assert result.confidence > 0.0


class TestRAGClassificationError:
    """Test custom exception for RAG classification errors."""

    def test_should_create_rag_classification_error(self):
        """Test that RAGClassificationError can be created with message."""
        error = RAGClassificationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_should_create_rag_classification_error_with_cause(self):
        """Test that RAGClassificationError can wrap other exceptions."""
        original_error = ValueError("Original error")
        try:
            raise RAGClassificationError("Wrapper error") from original_error
        except RAGClassificationError as error:
            assert str(error) == "Wrapper error"
            assert error.__cause__ == original_error
