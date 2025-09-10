"""
Unit tests for hybrid classification system.

Tests the HybridClassifier that combines RAG and rule-based classification
with confidence threshold fallback logic.
"""

import pytest
from unittest.mock import Mock, patch
from llm_router.hybrid_classification import HybridClassifier, HybridClassificationError
from llm_router.models import PromptClassification
from llm_router.rag_classification import RAGClassifier
from llm_router.classification import KeywordClassifier


class TestHybridClassifier:
    """Test cases for HybridClassifier."""

    @pytest.fixture
    def mock_rag_classifier(self):
        """Mock RAG classifier."""
        mock = Mock(spec=RAGClassifier)
        mock.get_confidence_threshold.return_value = 0.7
        return mock

    @pytest.fixture
    def mock_rule_classifier(self):
        """Mock rule-based classifier."""
        mock = Mock(spec=KeywordClassifier)
        return mock

    @pytest.fixture
    def hybrid_classifier(self, mock_rag_classifier, mock_rule_classifier):
        """Create HybridClassifier instance."""
        return HybridClassifier(
            rag_classifier=mock_rag_classifier,
            rule_classifier=mock_rule_classifier,
            rag_threshold=0.7,
            rule_threshold=0.5
        )

    def test_init_with_valid_parameters(self, mock_rag_classifier, mock_rule_classifier):
        """Test initialization with valid parameters."""
        classifier = HybridClassifier(
            rag_classifier=mock_rag_classifier,
            rule_classifier=mock_rule_classifier,
            rag_threshold=0.8,
            rule_threshold=0.6
        )
        
        assert classifier.rag_classifier == mock_rag_classifier
        assert classifier.rule_classifier == mock_rule_classifier
        assert classifier.rag_threshold == 0.8
        assert classifier.rule_threshold == 0.6

    def test_init_with_invalid_thresholds(self, mock_rag_classifier, mock_rule_classifier):
        """Test initialization with invalid threshold values."""
        # Test invalid RAG threshold
        with pytest.raises(HybridClassificationError, match="RAG threshold must be between 0.0 and 1.0"):
            HybridClassifier(
                rag_classifier=mock_rag_classifier,
                rule_classifier=mock_rule_classifier,
                rag_threshold=1.5,
                rule_threshold=0.5
            )
        
        # Test invalid rule threshold
        with pytest.raises(HybridClassificationError, match="Rule threshold must be between 0.0 and 1.0"):
            HybridClassifier(
                rag_classifier=mock_rag_classifier,
                rule_classifier=mock_rule_classifier,
                rag_threshold=0.7,
                rule_threshold=-0.1
            )

    def test_classify_high_confidence_rag(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test classification when RAG classifier has high confidence."""
        # Setup RAG classifier to return high confidence
        rag_result = PromptClassification(
            category="code",
            confidence=0.85,
            embedding=[],
            reasoning="High confidence RAG classification"
        )
        mock_rag_classifier.classify.return_value = rag_result
        
        result = hybrid_classifier.classify("Write a Python function")
        
        # Should use RAG result
        assert result.category == "code"
        assert result.confidence == 0.85
        assert "RAG" in result.reasoning
        mock_rag_classifier.classify.assert_called_once_with("Write a Python function")
        mock_rule_classifier.classify.assert_not_called()

    def test_classify_low_confidence_rag_fallback_to_rule(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test fallback to rule-based when RAG confidence is low."""
        # Setup RAG classifier to return low confidence
        rag_result = PromptClassification(
            category="code",
            confidence=0.4,
            embedding=[],
            reasoning="Low confidence RAG classification"
        )
        mock_rag_classifier.classify.return_value = rag_result
        
        # Setup rule classifier to return higher confidence
        rule_result = PromptClassification(
            category="creative",
            confidence=0.6,
            embedding=[],
            reasoning="Rule-based classification"
        )
        mock_rule_classifier.classify.return_value = rule_result
        
        result = hybrid_classifier.classify("Write a story")
        
        # Should use rule-based result
        assert result.category == "creative"
        assert result.confidence == 0.6
        assert "rule-based" in result.reasoning.lower()
        mock_rag_classifier.classify.assert_called_once_with("Write a story")
        mock_rule_classifier.classify.assert_called_once_with("Write a story")

    def test_classify_both_low_confidence_use_higher(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test using higher confidence when both are below thresholds."""
        # Setup RAG classifier to return low confidence
        rag_result = PromptClassification(
            category="code",
            confidence=0.3,
            embedding=[],
            reasoning="Low confidence RAG"
        )
        mock_rag_classifier.classify.return_value = rag_result
        
        # Setup rule classifier to return slightly higher but still low confidence
        rule_result = PromptClassification(
            category="creative",
            confidence=0.4,
            embedding=[],
            reasoning="Low confidence rule"
        )
        mock_rule_classifier.classify.return_value = rule_result
        
        result = hybrid_classifier.classify("Ambiguous prompt")
        
        # Should use rule-based result (higher confidence)
        assert result.category == "creative"
        assert result.confidence == 0.4
        assert "fallback" in result.reasoning.lower()

    def test_classify_rag_error_fallback_to_rule(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test fallback to rule-based when RAG classifier fails."""
        # Setup RAG classifier to raise an exception
        mock_rag_classifier.classify.side_effect = Exception("RAG error")
        
        # Setup rule classifier to return valid result
        rule_result = PromptClassification(
            category="qa",
            confidence=0.7,
            embedding=[],
            reasoning="Rule-based classification"
        )
        mock_rule_classifier.classify.return_value = rule_result
        
        result = hybrid_classifier.classify("What is AI?")
        
        # Should use rule-based result
        assert result.category == "qa"
        assert result.confidence == 0.7
        assert "rule-based" in result.reasoning.lower()

    def test_classify_both_fail_raises_error(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test error when both classifiers fail."""
        # Setup both classifiers to raise exceptions
        mock_rag_classifier.classify.side_effect = Exception("RAG error")
        mock_rule_classifier.classify.side_effect = Exception("Rule error")
        
        with pytest.raises(HybridClassificationError, match="Both classifiers failed"):
            hybrid_classifier.classify("Test prompt")

    def test_classify_empty_prompt(self, hybrid_classifier):
        """Test error handling for empty prompt."""
        with pytest.raises(HybridClassificationError, match="Prompt cannot be empty"):
            hybrid_classifier.classify("")
        
        with pytest.raises(HybridClassificationError, match="Prompt cannot be empty"):
            hybrid_classifier.classify("   ")

    def test_classify_none_prompt(self, hybrid_classifier):
        """Test error handling for None prompt."""
        with pytest.raises(HybridClassificationError, match="Prompt cannot be None"):
            hybrid_classifier.classify(None)

    def test_get_classification_method_rag(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test getting the classification method used (RAG)."""
        # Setup RAG classifier to return high confidence
        rag_result = PromptClassification(
            category="code",
            confidence=0.85,
            embedding=[],
            reasoning="High confidence RAG"
        )
        mock_rag_classifier.classify.return_value = rag_result
        
        result = hybrid_classifier.classify("Python function")
        method = hybrid_classifier.get_last_classification_method()
        
        assert method == "rag"

    def test_get_classification_method_rule(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test getting the classification method used (rule-based)."""
        # Setup RAG classifier to return low confidence
        rag_result = PromptClassification(
            category="code",
            confidence=0.4,
            embedding=[],
            reasoning="Low confidence RAG"
        )
        mock_rag_classifier.classify.return_value = rag_result
        
        # Setup rule classifier to return higher confidence
        rule_result = PromptClassification(
            category="creative",
            confidence=0.6,
            embedding=[],
            reasoning="Rule-based classification"
        )
        mock_rule_classifier.classify.return_value = rule_result
        
        result = hybrid_classifier.classify("Write a story")
        method = hybrid_classifier.get_last_classification_method()
        
        assert method == "rule"

    def test_get_classification_method_fallback(self, hybrid_classifier, mock_rag_classifier, mock_rule_classifier):
        """Test getting the classification method used (fallback)."""
        # Setup both classifiers to return low confidence
        rag_result = PromptClassification(
            category="code",
            confidence=0.3,
            embedding=[],
            reasoning="Low confidence RAG"
        )
        mock_rag_classifier.classify.return_value = rag_result
        
        rule_result = PromptClassification(
            category="creative",
            confidence=0.4,
            embedding=[],
            reasoning="Low confidence rule"
        )
        mock_rule_classifier.classify.return_value = rule_result
        
        result = hybrid_classifier.classify("Ambiguous prompt")
        method = hybrid_classifier.get_last_classification_method()
        
        assert method == "fallback"

    def test_update_thresholds(self, hybrid_classifier):
        """Test updating confidence thresholds."""
        hybrid_classifier.set_rag_threshold(0.8)
        hybrid_classifier.set_rule_threshold(0.6)
        
        assert hybrid_classifier.rag_threshold == 0.8
        assert hybrid_classifier.rule_threshold == 0.6

    def test_invalid_threshold_updates(self, hybrid_classifier):
        """Test invalid threshold updates."""
        with pytest.raises(HybridClassificationError):
            hybrid_classifier.set_rag_threshold(1.5)
        
        with pytest.raises(HybridClassificationError):
            hybrid_classifier.set_rule_threshold(-0.1)
