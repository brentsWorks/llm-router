"""
Unit tests for LLM Fallback Classification system.

This module tests the LLM-based classification service that handles edge cases
and novel prompt types when RAG + rule-based classification fails.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from llm_router.llm_fallback import LLMFallbackClassifier, LLMFallbackError
from llm_router.models import PromptClassification
from llm_router.capabilities import TaskType


class TestLLMFallbackClassifier:
    """Test LLM fallback classifier functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "reasoning": "This appears to be a programming question based on the technical terminology and request for code implementation."}'
        mock_client.generate_content.return_value = mock_response
        return mock_client

    @pytest.fixture
    def sample_classification_prompt(self):
        """Create a sample classification prompt for testing."""
        return """
        Analyze the following prompt and classify it into one of these categories:
        - code: Programming, debugging, technical implementation
        - creative: Writing, storytelling, creative content
        - qa: Questions, explanations, educational content
        - analysis: Data analysis, research, analytical tasks
        - summarization: Summarizing, condensing information
        - translation: Language translation tasks
        - reasoning: Complex reasoning, problem-solving

        Prompt: "Write a Python function to implement quicksort algorithm"
        
        Respond with JSON format:
        {
            "category": "category_name",
            "confidence": 0.0-1.0,
            "reasoning": "explanation of classification decision"
        }
        """

    def test_should_initialize_llm_fallback_classifier(self, mock_llm_client):
        """Test that LLM fallback classifier initializes correctly."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        assert classifier.api_key == "test-key"
        assert classifier.llm_client == mock_llm_client
        assert classifier.model_name == "gpt-3.5-turbo"  # Default model

    def test_should_require_api_key_for_initialization(self, mock_llm_client):
        """Test that API key is required for initialization."""
        with pytest.raises(LLMFallbackError) as exc_info:
            LLMFallbackClassifier(
                llm_client=mock_llm_client,
                api_key=""
            )
        
        assert "API key is required" in str(exc_info.value)

    def test_should_classify_prompt_using_llm(self, mock_llm_client, sample_classification_prompt):
        """Test that classifier uses LLM for classification."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 0.85, "reasoning": "This is clearly a programming request for algorithm implementation"}'
        mock_llm_client.generate_content.return_value = mock_response
        
        result = classifier.classify("Write a Python function to implement quicksort algorithm")
        
        # Verify LLM was called
        mock_llm_client.generate_content.assert_called_once()
        call_args = mock_llm_client.generate_content.call_args[0][0]
        assert "Write a Python function to implement quicksort algorithm" in call_args
        
        # Verify result
        assert isinstance(result, PromptClassification)
        assert result.category == "code"
        assert result.confidence == 0.85
        assert "programming request" in result.reasoning

    def test_should_handle_llm_response_parsing_errors(self, mock_llm_client):
        """Test handling of invalid LLM responses."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON response"
        mock_llm_client.generate_content.return_value = mock_response
        
        with pytest.raises(LLMFallbackError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Failed to parse LLM response" in str(exc_info.value)

    def test_should_validate_classification_categories(self, mock_llm_client):
        """Test that only valid categories are accepted."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        # Mock response with invalid category
        mock_response = MagicMock()
        mock_response.text = '{"category": "invalid_category", "confidence": 0.85, "reasoning": "test"}'
        mock_llm_client.generate_content.return_value = mock_response
        
        with pytest.raises(LLMFallbackError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Invalid category" in str(exc_info.value)

    def test_should_validate_confidence_bounds(self, mock_llm_client):
        """Test that confidence values are within valid bounds."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        # Mock response with invalid confidence
        mock_response = MagicMock()
        mock_response.text = '{"category": "code", "confidence": 1.5, "reasoning": "test"}'
        mock_llm_client.generate_content.return_value = mock_response
        
        with pytest.raises(LLMFallbackError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "Invalid confidence value" in str(exc_info.value)

    def test_should_handle_llm_api_failures(self, mock_llm_client):
        """Test handling of LLM API failures."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        # Mock API failure
        mock_llm_client.generate_content.side_effect = Exception("API rate limit exceeded")
        
        with pytest.raises(LLMFallbackError) as exc_info:
            classifier.classify("Test prompt")
        
        assert "LLM API error" in str(exc_info.value)

    def test_should_use_custom_model_name(self, mock_llm_client):
        """Test that custom model name is used."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key",
            model_name="gpt-4"
        )
        
        assert classifier.model_name == "gpt-4"

    def test_should_generate_proper_classification_prompt(self, mock_llm_client):
        """Test that proper classification prompt is generated."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        prompt = "Write a creative story about dragons"
        expected_categories = [cat.value for cat in TaskType]
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = '{"category": "creative", "confidence": 0.9, "reasoning": "This is a creative writing request"}'
        mock_llm_client.generate_content.return_value = mock_response
        
        classifier.classify(prompt)
        
        # Verify the prompt contains all valid categories
        call_args = mock_llm_client.generate_content.call_args[0][0]
        for category in expected_categories:
            assert category in call_args
        assert prompt in call_args


class TestLLMFallbackClassifierIntegration:
    """Integration tests for LLM fallback classifier."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = MagicMock()
        return mock_client

    def test_should_classify_different_prompt_types(self, mock_llm_client):
        """Test classification of different types of prompts."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        test_cases = [
            ("Write a Python function", "code", 0.9),
            ("Tell me a story about space", "creative", 0.85),
            ("What is machine learning?", "qa", 0.8),
            ("Analyze this data set", "analysis", 0.9),
            ("Summarize this article", "summarization", 0.85),
            ("Translate to Spanish", "translation", 0.9),
            ("Solve this math problem", "reasoning", 0.8)
        ]
        
        for prompt, expected_category, expected_confidence in test_cases:
            # Mock response for each test case
            mock_response = MagicMock()
            mock_response.text = f'{{"category": "{expected_category}", "confidence": {expected_confidence}, "reasoning": "Test reasoning"}}'
            mock_llm_client.generate_content.return_value = mock_response
            
            result = classifier.classify(prompt)
            
            assert result.category == expected_category
            assert result.confidence == expected_confidence
            assert isinstance(result.embedding, list)  # Should have empty embedding
            assert result.reasoning == "Test reasoning"

    def test_should_handle_edge_cases(self, mock_llm_client):
        """Test handling of edge cases and novel prompts."""
        classifier = LLMFallbackClassifier(
            llm_client=mock_llm_client,
            api_key="test-key"
        )
        
        edge_cases = [
            "xyzabc nonsense gibberish random words",
            "This is a completely novel type of request that doesn't fit any pattern",
            "Help me with something that I can't describe well",
            "I need assistance with a unique problem"
        ]
        
        for prompt in edge_cases:
            # Mock response for edge cases
            mock_response = MagicMock()
            mock_response.text = '{"category": "qa", "confidence": 0.3, "reasoning": "Unclear prompt, defaulting to Q&A"}'
            mock_llm_client.generate_content.return_value = mock_response
            
            result = classifier.classify(prompt)
            
            assert isinstance(result, PromptClassification)
            assert result.category in [cat.value for cat in TaskType]
            assert 0.0 <= result.confidence <= 1.0
            assert result.reasoning is not None
