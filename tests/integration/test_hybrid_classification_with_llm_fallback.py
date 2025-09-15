"""
Integration tests for Hybrid Classifier with LLM Fallback integration.

This module tests the integration of LLM fallback classification into the
simplified hybrid classifier: RAG → LLM Fallback flow.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from llm_router.hybrid_classification import HybridClassifier
from llm_router.llm_fallback import LLMFallbackClassifier
from llm_router.models import PromptClassification
from llm_router.vector_stores import SearchResult


class TestHybridClassifierWithLLMFallbackIntegration:
    """Integration tests for simplified RAG → LLM Fallback flow."""

    @pytest.fixture
    def mock_vector_service(self):
        """Create a mock vector service."""
        mock_service = MagicMock()
        mock_service.find_similar_examples.return_value = []
        return mock_service

    @pytest.fixture
    def mock_rag_classifier(self):
        """Create a mock RAG classifier."""
        mock_rag = MagicMock()
        mock_rag.classify.return_value = PromptClassification(
            category="code",
            confidence=0.8,  # High confidence
            embedding=[0.1, 0.2, 0.3],
            reasoning="High confidence RAG result"
        )
        return mock_rag

    @pytest.fixture
    def mock_llm_fallback(self):
        """Create a mock LLM fallback classifier."""
        mock_llm = MagicMock()
        mock_llm.classify.return_value = PromptClassification(
            category="creative",
            confidence=0.85,  # High confidence from LLM
            embedding=[],
            reasoning="LLM analysis suggests this is creative content"
        )
        return mock_llm

    def test_should_initialize_with_llm_fallback(self, mock_vector_service, mock_rag_classifier, mock_llm_fallback):
        """Test that hybrid classifier initializes with LLM fallback."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            llm_fallback=mock_llm_fallback,
            rag_threshold=0.5
        )
        
        assert classifier.llm_fallback == mock_llm_fallback
        assert classifier.rag_threshold == 0.5

    def test_should_use_rag_when_high_confidence(self, mock_vector_service, mock_rag_classifier, mock_llm_fallback):
        """Test that RAG is used when it has high confidence."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            llm_fallback=mock_llm_fallback,
            rag_threshold=0.5
        )
        
        # RAG returns high confidence
        mock_rag_classifier.classify.return_value = PromptClassification(
            category="code", confidence=0.8, embedding=[], reasoning="High confidence RAG"
        )
        
        result = classifier.classify("Write a Python function")
        
        # Should use RAG, not LLM fallback
        mock_rag_classifier.classify.assert_called_once()
        mock_llm_fallback.classify.assert_not_called()
        assert result.category == "code"
        assert result.confidence == 0.8
        assert classifier.get_last_classification_method() == "rag"

    def test_should_use_llm_fallback_when_rag_has_low_confidence(self, mock_vector_service, mock_rag_classifier, mock_llm_fallback):
        """Test that LLM fallback is used when RAG has low confidence."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            llm_fallback=mock_llm_fallback,
            rag_threshold=0.5
        )
        
        # RAG returns low confidence (below threshold)
        mock_rag_classifier.classify.return_value = PromptClassification(
            category="code", confidence=0.3, embedding=[], reasoning="Low confidence RAG"
        )
        
        result = classifier.classify("Some novel prompt that doesn't match patterns")
        
        # Should use LLM fallback due to low confidence
        mock_llm_fallback.classify.assert_called_once_with("Some novel prompt that doesn't match patterns")
        assert result.category == "creative"
        assert result.confidence == 0.85
        assert "LLM analysis" in result.reasoning
        assert classifier.get_last_classification_method() == "llm_fallback"

    def test_should_use_llm_fallback_when_rag_fails(self, mock_vector_service, mock_rag_classifier, mock_llm_fallback):
        """Test that LLM fallback is used when RAG fails completely."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            llm_fallback=mock_llm_fallback,
            rag_threshold=0.5
        )
        
        # RAG fails completely
        mock_rag_classifier.classify.side_effect = Exception("RAG classification failed")
        
        result = classifier.classify("Write a function")
        
        # Should use LLM fallback
        mock_rag_classifier.classify.assert_called_once()
        mock_llm_fallback.classify.assert_called_once()
        assert result.category == "creative"
        assert result.confidence == 0.85
        assert classifier.get_last_classification_method() == "llm_fallback"

    def test_should_handle_llm_fallback_errors_gracefully(self, mock_vector_service, mock_rag_classifier, mock_llm_fallback):
        """Test that LLM fallback errors are handled gracefully."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            llm_fallback=mock_llm_fallback,
            rag_threshold=0.5
        )
        
        # RAG fails
        mock_rag_classifier.classify.side_effect = Exception("RAG classification failed")
        
        # LLM fallback also fails
        mock_llm_fallback.classify.side_effect = Exception("LLM API error")
        
        with pytest.raises(Exception) as exc_info:
            classifier.classify("Some prompt")
        
        assert "All classifiers failed" in str(exc_info.value)

    def test_should_track_classification_method_used(self, mock_vector_service, mock_rag_classifier, mock_llm_fallback):
        """Test that the classification method used is tracked correctly."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            llm_fallback=mock_llm_fallback,
            rag_threshold=0.5
        )
        
        # Test RAG method
        mock_rag_classifier.classify.return_value = PromptClassification(
            category="code", confidence=0.8, embedding=[], reasoning="High confidence RAG"
        )
        classifier.classify("Test prompt")
        assert classifier.get_last_classification_method() == "rag"
        
        # Test LLM fallback method
        mock_rag_classifier.classify.return_value = PromptClassification(
            category="code", confidence=0.3, embedding=[], reasoning="Low confidence RAG"
        )
        mock_llm_fallback.classify.return_value = PromptClassification(
            category="creative", confidence=0.85, embedding=[], reasoning="LLM analysis"
        )
        classifier.classify("Test prompt")
        assert classifier.get_last_classification_method() == "llm_fallback"

    def test_should_work_without_llm_fallback(self, mock_vector_service, mock_rag_classifier):
        """Test that hybrid classifier works without LLM fallback (backward compatibility)."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            llm_fallback=None,  # No LLM fallback
            rag_threshold=0.5
        )
        
        # RAG fails
        mock_rag_classifier.classify.side_effect = Exception("RAG classification failed")
        
        with pytest.raises(Exception) as exc_info:
            classifier.classify("Some prompt")
        
        assert "All classifiers failed" in str(exc_info.value)