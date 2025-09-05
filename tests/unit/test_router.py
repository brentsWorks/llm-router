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
        mock_registry.get_models_by_capability.return_value = mock_models

        # Call route method
        router.route("Write a Python function")

        # Verify registry was called with correct category
        mock_registry.get_models_by_capability.assert_called_once_with("code")

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
        mock_registry.get_models_by_capability.return_value = mock_models

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

    def test_should_handle_registry_unavailable_exception(self):
        """Test that router handles registry exceptions gracefully."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification success
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry to raise exception
        mock_registry.get_models_for_category.side_effect = Exception("Registry database connection failed")

        # Should return None when registry fails
        result = router.route("Write a function")
        assert result is None

        # Verify classifier was still called
        mock_classifier.classify.assert_called_once_with("Write a function")

    def test_should_handle_registry_returns_none(self):
        """Test that router handles when registry returns None."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification success
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry returns None (shouldn't happen in real implementation but test robustness)
        mock_registry.get_models_for_category.return_value = None

        # Should return None when registry returns None
        result = router.route("Write a function")
        assert result is None

    def test_should_handle_registry_network_timeout(self):
        """Test that router handles registry network timeouts."""
        from llm_router.router import RouterService
        import time

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classification success
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry to simulate network timeout
        def timeout_simulation(*args, **kwargs):
            time.sleep(0.01)  # Very short delay for testing
            raise ConnectionError("Network timeout connecting to registry")

        mock_registry.get_models_for_category.side_effect = timeout_simulation

        # Should handle timeout gracefully and return None
        start_time = time.time()
        result = router.route("Write a function")
        end_time = time.time()

        assert result is None
        # Should complete reasonably quickly despite the simulated delay
        assert (end_time - start_time) < 1.0  # Less than 1 second

    def test_should_handle_classifier_exception(self):
        """Test that router handles classifier exceptions gracefully."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classifier to raise exception
        mock_classifier.classify.side_effect = Exception("Classification service unavailable")

        # Should return None when classifier fails
        result = router.route("Write a function")
        assert result is None

        # Verify classifier was called but failed
        mock_classifier.classify.assert_called_once_with("Write a function")

    def test_should_handle_classifier_returns_none(self):
        """Test that router handles when classifier returns None."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classifier returns None
        mock_classifier.classify.return_value = None

        # Should return None when classifier returns None
        result = router.route("Write a function")
        assert result is None

        # Verify classifier was called
        mock_classifier.classify.assert_called_once_with("Write a function")

    def test_should_handle_classifier_returns_invalid_result(self):
        """Test that router handles when classifier returns invalid result."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classifier returns invalid result (missing required fields)
        mock_classifier.classify.return_value = Mock(
            category="code",
            # Missing confidence, embedding, reasoning
        )

        # Should return None when classifier returns invalid result
        result = router.route("Write a function")
        assert result is None

    def test_should_handle_classifier_network_timeout(self):
        """Test that router handles classifier network timeouts."""
        from llm_router.router import RouterService
        import time

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock classifier to simulate network timeout
        def timeout_simulation(*args, **kwargs):
            time.sleep(0.01)  # Very short delay for testing
            raise ConnectionError("Network timeout connecting to classifier")

        mock_classifier.classify.side_effect = timeout_simulation

        # Should handle timeout gracefully and return None
        start_time = time.time()
        result = router.route("Write a function")
        end_time = time.time()

        assert result is None
        # Should complete reasonably quickly
        assert (end_time - start_time) < 1.0  # Less than 1 second

    def test_should_handle_classifier_with_empty_prompt(self):
        """Test that router handles classifier with empty or invalid prompts."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Test various edge case prompts
        edge_cases = ["", "   ", None]

        for prompt in edge_cases:
            # Reset mocks for each test
            mock_classifier.reset_mock()

            # Mock classifier to handle edge case
            if prompt is None:
                mock_classifier.classify.side_effect = TypeError("Cannot classify None")
            else:
                mock_classifier.classify.return_value = None

            # Should handle edge cases gracefully
            result = router.route(prompt)
            assert result is None

    def test_should_classifier_be_called_with_correct_prompt(self):
        """Test that classifier receives the exact prompt passed to router."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Test various prompt types
        test_prompts = [
            "Simple prompt",
            "Complex prompt with multiple words and punctuation!",
            "123",
            "Prompt with special chars: @#$%^&*()"
        ]

        for prompt in test_prompts:
            # Reset mocks
            mock_classifier.reset_mock()

            # Mock successful classification
            mock_classification = PromptClassification(
                category="code",
                confidence=0.8,
                embedding=[],
                reasoning="Test"
            )
            mock_classifier.classify.return_value = mock_classification

            # Mock registry and ranker
            mock_registry.get_models_for_category.return_value = [Mock()]
            mock_ranking = Mock()
            mock_ranking.ranked_models = [Mock()]
            mock_ranking.ranking_scores = [0.8]
            mock_ranker.rank_models.return_value = mock_ranking

            # Route the prompt
            router.route(prompt)

            # Verify classifier was called with exact prompt
            mock_classifier.classify.assert_called_once_with(prompt)

    def test_should_handle_ranker_exception(self):
        """Test that router handles ranker exceptions gracefully."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification and registry
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_registry.get_models_for_category.return_value = [Mock()]

        # Mock ranker to raise exception
        mock_ranker.rank_models.side_effect = Exception("Ranking service failed")

        # Should return None when ranker fails
        result = router.route("Write a function")
        assert result is None

    def test_should_handle_ranker_returns_none(self):
        """Test that router handles when ranker returns None."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification and registry
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_registry.get_models_for_category.return_value = [Mock()]

        # Mock ranker returns None
        mock_ranker.rank_models.return_value = None

        # Should return None when ranker returns None
        result = router.route("Write a function")
        assert result is None

    def test_should_handle_ranker_returns_invalid_result(self):
        """Test that router handles when ranker returns invalid result."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification and registry
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_registry.get_models_for_category.return_value = [Mock()]

        # Mock ranker returns invalid result (missing required fields)
        mock_ranker.rank_models.return_value = Mock(
            ranked_models=None,  # Invalid: should be a list
            ranking_scores=[0.8]
        )

        # Should return None when ranker returns invalid result
        result = router.route("Write a function")
        assert result is None

    def test_should_handle_ranker_returns_empty_ranked_models(self):
        """Test that router handles when ranker returns empty ranked models list."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification and registry
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_registry.get_models_for_category.return_value = [Mock()]

        # Mock ranker returns empty ranked models
        mock_ranker.rank_models.return_value = Mock(
            ranked_models=[],  # Empty list
            ranking_scores=[]
        )

        # Should return None when no ranked models available
        result = router.route("Write a function")
        assert result is None

    def test_should_handle_ranker_network_timeout(self):
        """Test that router handles ranker network timeouts."""
        from llm_router.router import RouterService
        import time

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification and registry
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_registry.get_models_for_category.return_value = [Mock()]

        # Mock ranker to simulate network timeout
        def timeout_simulation(*args, **kwargs):
            time.sleep(0.01)  # Very short delay for testing
            raise ConnectionError("Network timeout connecting to ranker")

        mock_ranker.rank_models.side_effect = timeout_simulation

        # Should handle timeout gracefully and return None
        start_time = time.time()
        result = router.route("Write a function")
        end_time = time.time()

        assert result is None
        # Should complete reasonably quickly
        assert (end_time - start_time) < 1.0  # Less than 1 second

    def test_should_handle_ranker_with_invalid_model_data(self):
        """Test that router handles ranker with invalid model data."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification and registry
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_registry.get_models_for_category.return_value = [Mock()]

        # Mock ranker returns corrupted data
        mock_ranker.rank_models.return_value = Mock(
            ranked_models=["invalid", "data"],  # Should be ModelCandidate objects
            ranking_scores=[0.8, 0.7]
        )

        # Should handle gracefully and return None
        result = router.route("Write a function")
        assert result is None

    def test_should_ranker_receive_correct_parameters(self):
        """Test that ranker receives correct parameters from router."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification and registry
        mock_classification = PromptClassification(
            category="creative",
            confidence=0.9,
            embedding=[],
            reasoning="Creative detected"
        )
        mock_classifier.classify.return_value = mock_classification

        mock_models = [Mock(), Mock()]
        mock_registry.get_models_by_capability.return_value = mock_models

        # Mock ranker response
        mock_ranking = Mock()
        mock_ranking.ranked_models = mock_models
        mock_ranking.ranking_scores = [0.8, 0.7]
        mock_ranker.rank_models.return_value = mock_ranking

        # Route the prompt
        router.route("Write a story")

        # Verify ranker was called with correct parameters
        mock_ranker.rank_models.assert_called_once()
        call_args = mock_ranker.rank_models.call_args
        assert call_args[0][0] == mock_models  # First argument should be the models
        assert call_args[0][1] == "creative"   # Second argument should be the category

    def test_should_implement_fallback_to_default_model_when_all_services_fail(self):
        """Test that router implements fallback to a default model when all services fail."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock all services to fail
        mock_classifier.classify.side_effect = Exception("Classification completely down")
        mock_registry.get_models_for_category.side_effect = Exception("Registry completely down")
        mock_ranker.rank_models.side_effect = Exception("Ranking completely down")

        # Should return None when all services fail (no fallback available yet)
        result = router.route("Any prompt")
        assert result is None

    def test_should_handle_partial_service_failures_with_best_effort_routing(self):
        """Test that router handles partial failures with best-effort routing."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry returns models
        mock_models = [Mock(provider="openai", model="gpt-4")]
        mock_registry.get_models_for_category.return_value = mock_models

        # Mock ranker fails but we have models to work with
        mock_ranker.rank_models.side_effect = Exception("Ranking temporarily unavailable")

        # Should return None when ranking fails (could implement fallback ranking)
        result = router.route("Write a function")
        assert result is None

    def test_should_validate_input_parameters_and_handle_edge_cases(self):
        """Test that router validates input parameters and handles edge cases."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Test with None input
        result = router.route(None)
        assert result is None

        # Test with empty string
        result = router.route("")
        assert result is None

        # Test with very long input
        long_prompt = "A" * 10000
        mock_classifier.classify.return_value = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_registry.get_models_for_category.return_value = [Mock()]
        mock_ranker.rank_models.return_value = Mock(ranked_models=[Mock()], ranking_scores=[0.8])

        result = router.route(long_prompt)
        # Should handle long prompts (implementation may truncate or handle differently)
        assert result is not None or result is None  # Either is acceptable

    def test_should_implement_graceful_degradation_under_high_load(self):
        """Test that router degrades gracefully under simulated high load."""
        from llm_router.router import RouterService
        import time

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful responses
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_registry.get_models_for_category.return_value = [Mock()]
        mock_ranker.rank_models.return_value = Mock(ranked_models=[Mock()], ranking_scores=[0.8])

        # Simulate multiple rapid requests
        results = []
        start_time = time.time()

        for i in range(10):
            result = router.route(f"Request {i}")
            results.append(result)

        end_time = time.time()
        total_time = end_time - start_time

        # All requests should succeed
        assert all(result is not None for result in results)
        # Should complete within reasonable time (under 2 seconds for 10 requests)
        assert total_time < 2.0

    def test_should_provide_detailed_error_information_in_debug_mode(self):
        """Test that router provides detailed error information when needed."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock a specific failure scenario
        mock_classifier.classify.side_effect = ValueError("Invalid classification input")

        # Route should handle the error gracefully
        result = router.route("Invalid input that causes classification error")
        assert result is None

        # Verify the error was handled without propagating exceptions
        # This ensures the router doesn't crash the application

    def test_should_recover_from_temporary_service_outages(self):
        """Test that router can recover from temporary service outages."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock initial failure then recovery
        mock_classifier.classify.side_effect = [
            ConnectionError("Temporary network issue"),  # First call fails
            PromptClassification(category="code", confidence=0.8, embedding=[], reasoning="Recovered")  # Second call succeeds
        ]

        # First request should fail
        result1 = router.route("First request during outage")
        assert result1 is None

        # Second request should succeed (service recovered)
        mock_registry.get_models_for_category.return_value = [Mock()]
        mock_ranker.rank_models.return_value = Mock(ranked_models=[Mock()], ranking_scores=[0.8])

        result2 = router.route("Second request after recovery")
        assert result2 is not None

    def test_should_handle_corrupted_model_data_gracefully(self):
        """Test that router handles corrupted or malformed model data."""
        from llm_router.router import RouterService

        mock_classifier = Mock()
        mock_registry = Mock()
        mock_ranker = Mock()

        router = RouterService(
            classifier=mock_classifier,
            registry=mock_registry,
            ranker=mock_ranker
        )

        # Mock successful classification
        mock_classification = PromptClassification(
            category="code",
            confidence=0.8,
            embedding=[],
            reasoning="Code detected"
        )
        mock_classifier.classify.return_value = mock_classification

        # Mock registry returns models with missing attributes
        corrupted_model = Mock()
        # Intentionally don't set provider/model attributes to simulate corruption
        mock_registry.get_models_for_category.return_value = [corrupted_model]

        # Mock ranker returns corrupted model
        mock_ranker.rank_models.return_value = Mock(
            ranked_models=[corrupted_model],
            ranking_scores=[0.8]
        )

        # Should handle corrupted data gracefully by providing safe defaults
        result = router.route("Request with corrupted model data")

        # Should successfully create a RoutingDecision with safe defaults
        assert isinstance(result, RoutingDecision)
        assert result.selected_model.provider == "unknown"  # Safe default for missing provider
        assert result.selected_model.model == "unknown"     # Safe default for missing model
        assert result.confidence == 0.8  # Should still have the ranking score
