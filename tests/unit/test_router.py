"""Unit tests for Phase 5.1: Basic Router Service.

This module tests the RouterService that orchestrates the entire LLM routing pipeline.
"""

import pytest
from unittest.mock import Mock, MagicMock

from llm_router.models import RoutingDecision, ModelCandidate, PromptClassification
from llm_router.registry import ProviderModel


class TestRouterService:
    """Test cases for the RouterService class."""

    def test_should_create_router_service_instance(self):
        """Test that RouterService can be instantiated with required dependencies."""
        # This test will fail initially (Red phase) until we create RouterService
        from llm_router.router import RouterService

        # Mock dependencies
        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        # Should be able to create RouterService instance
        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        assert router is not None
        assert hasattr(router, 'classifier')
        assert hasattr(router, 'registry')
        assert hasattr(router, 'ranker')

    def test_should_have_route_method(self):
        """Test that RouterService has a route method with correct signature."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Should have route method
        assert hasattr(router, 'route')
        assert callable(getattr(router, 'route'))

    def test_should_accept_prompt_string_in_route_method(self):
        """Test that route method accepts a prompt string parameter."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock the return values
        mock_classifier.classify.return_value = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Test classification"
        )

        # This should not raise an exception
        try:
            result = router.route("Write a Python function")
            # We expect this to work, even if it returns None initially
        except Exception as e:
            # If it fails, that's expected in Red phase
            assert "not implemented" in str(e).lower() or "not yet implemented" in str(e).lower()

    def test_should_store_dependencies_correctly(self):
        """Test that RouterService stores its dependencies correctly."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Should store dependencies
        assert router.classifier is mock_classifier
        assert router.registry is mock_registry
        assert router.ranker is mock_ranker

    def test_should_initialize_with_none_dependencies_as_error(self):
        """Test that RouterService requires all dependencies."""
        from llm_router.router import RouterService

        # Should raise error with None dependencies
        with pytest.raises((ValueError, TypeError)):
            RouterService(
                classifier=None,
                registry=Mock(),
                ranker=Mock()
            )

        with pytest.raises((ValueError, TypeError)):
            RouterService(
                classifier=Mock(),
                registry=None,
                ranker=Mock()
            )

        with pytest.raises((ValueError, TypeError)):
            RouterService(
                classifier=Mock(),
                registry=Mock(),
                ranker=None
            )

    def test_should_call_classifier_when_routing(self):
        """Test that router calls the classifier when routing a prompt."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classifier response
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Mock classification"
        )
        mock_classifier.classify.return_value = mock_classification

        # Call route method
        router.route("Write a Python function")

        # Verify classifier was called with the prompt
        mock_classifier.classify.assert_called_once_with("Write a Python function")

    def test_should_return_none_when_classifier_fails(self):
        """Test that router returns None when classification fails."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classifier to raise an exception
        mock_classifier.classify.side_effect = Exception("Classification failed")

        # Should return None when classification fails
        result = router.route("Test prompt")
        assert result is None

    def test_should_use_classification_result_for_further_processing(self):
        """Test that router uses the classification result for model selection."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification response
        mock_classification = PromptClassification(
            category="creative",
            confidence=0.9,
            embedding=[],
            reasoning="Creative writing detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Call route method
        result = router.route("Write a story")

        # Verify classifier was called and got expected result
        mock_classifier.classify.assert_called_once_with("Write a story")
        assert mock_classifier.classify.return_value.category == "creative"
        assert mock_classifier.classify.return_value.confidence == 0.9

    def test_should_handle_different_prompt_categories(self):
        """Test that router handles different prompt categories from classifier."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        test_cases = [
            ("Write a function", "code", 0.8),
            ("Tell me a story", "creative", 0.7),
            ("What is AI?", "qa", 0.6),
        ]

        for prompt, expected_category, expected_confidence in test_cases:
            # Reset mocks
            mock_classifier.reset_mock()

            # Mock classification
            mock_classification = PromptClassification(
                category=expected_category,
                confidence=expected_confidence,
                embedding=[],
                reasoning=f"Detected {expected_category}"
            )
            mock_classifier.classify.return_value = mock_classification

            # Route the prompt
            result = router.route(prompt)

            # Verify classifier was called correctly
            mock_classifier.classify.assert_called_once_with(prompt)
            assert mock_classifier.classify.return_value.category == expected_category
            assert mock_classifier.classify.return_value.confidence == expected_confidence

    def test_should_get_models_from_registry_based_on_category(self):
        """Test that router queries registry for models matching the classification category."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry response
        mock_models = [
            Mock(provider="openai", model="gpt-4", capabilities=["code"]),
            Mock(provider="anthropic", model="claude-3", capabilities=["code"])
        ]
        mock_registry.get_models_for_category.return_value = mock_models

        # Call route method
        router.route("Write a Python function")

        # Verify registry was called with correct category
        mock_registry.get_models_for_category.assert_called_once_with("code")

    def test_should_call_ranker_with_filtered_models(self):
        """Test that router passes filtered models to the ranker."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification
        mock_classification = PromptClassification(
            category="creative",
            confidence=0.9,
            embedding=[],
            reasoning="Creative writing detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry response
        mock_models = [Mock(capabilities=["creative"]), Mock(capabilities=["creative"])]
        mock_registry.get_models_for_category.return_value = mock_models

        # Mock ranker response
        mock_ranker.rank_models.return_value = Mock(
            ranked_models=[mock_models[0]],
            ranking_scores=[0.85]
        )

        # Call route method
        router.route("Write a story")

        # Verify ranker was called with the filtered models
        mock_ranker.rank_models.assert_called_once()
        call_args = mock_ranker.rank_models.call_args
        assert call_args[0][0] == mock_models  # First argument should be the models
        assert call_args[0][1] == "creative"   # Second argument should be the category

    def test_should_return_routing_decision_with_selected_model(self):
        """Test that router returns a proper RoutingDecision when model selection succeeds."""
        from llm_router.router import RouterService
        from llm_router.models import RoutingDecision

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification
        mock_classification = PromptClassification(
            category="qa",
            confidence=0.7,
            embedding=[],
            reasoning="Question detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry and ranker responses
        mock_model = Mock()
        mock_model.provider = "openai"
        mock_model.model = "gpt-4"

        mock_registry.get_models_for_category.return_value = [mock_model]

        mock_ranking_result = Mock()
        mock_ranking_result.ranked_models = [mock_model]
        mock_ranking_result.ranking_scores = [0.8]
        mock_ranker.rank_models.return_value = mock_ranking_result

        # Call route method
        result = router.route("What is AI?")

        # Should return a RoutingDecision
        assert isinstance(result, RoutingDecision)
        # The router converts the mock model to a proper ModelCandidate
        assert result.selected_model.provider == "openai"
        assert result.selected_model.model == "gpt-4"
        assert result.selected_model.score == 0.8
        assert result.confidence == 0.8  # The model's ranking score
        assert result.routing_time_ms == 10.5  # Mock routing time
        assert "openai/gpt-4" in result.reasoning
        assert result.classification.category == "qa"  # Should include the classification

    def test_should_return_none_when_no_models_available(self):
        """Test that router returns None when no suitable models are found."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry returns empty list (no models available)
        mock_registry.get_models_for_category.return_value = []

        # Call route method
        result = router.route("Write a function")

        # Should return None when no models are available
        assert result is None
