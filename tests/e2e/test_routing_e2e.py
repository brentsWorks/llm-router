"""End-to-End tests for the complete LLM routing pipeline.

These tests validate the entire routing system from prompt input to final routing decision,
using real services to ensure the complete pipeline works correctly.
"""

import time
import pytest
from unittest.mock import Mock

from llm_router.router import RouterService
from llm_router.classification import KeywordClassifier
from llm_router.models import RoutingDecision, ModelCandidate


class TestEndToEndRouting:
    """End-to-end tests for the complete routing pipeline."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create real KeywordClassifier
        self.classifier = KeywordClassifier()

        # Create mock registry and ranker for controlled testing
        self.mock_registry = Mock()
        self.mock_ranker = Mock()

        # Create RouterService with real classifier but mocked registry/ranker
        self.router = RouterService(
            classifier=self.classifier,
            registry=self.mock_registry,
            ranker=self.mock_ranker
        )

    def test_should_route_code_prompt_end_to_end(self):
        """Test complete routing pipeline for a code-related prompt."""
        # Setup: Mock registry returns code-capable models
        code_models = [
            self._create_mock_model("gpt-4", "openai", ["code"]),
            self._create_mock_model("claude-3", "anthropic", ["code"])
        ]
        self.mock_registry.get_models_for_category.return_value = code_models

        # Mock ranker returns ranking results
        mock_ranking = Mock()
        mock_ranking.ranked_models = code_models
        mock_ranking.ranking_scores = [0.9, 0.8]
        self.mock_ranker.rank_models.return_value = mock_ranking

        # Test: Route a code prompt
        prompt = "Write a Python function to calculate fibonacci numbers"
        start_time = time.time()
        result = self.router.route(prompt)
        routing_time = time.time() - start_time

        # Verify: Complete routing decision
        assert isinstance(result, RoutingDecision)
        assert result.selected_model.provider == "openai"
        assert result.selected_model.model == "gpt-4"
        assert result.confidence == 0.9
        assert result.classification.category == "code"
        assert result.routing_time_ms > 0
        assert "openai/gpt-4" in result.reasoning
        assert "code" in result.reasoning.lower()
        assert routing_time < 1.0  # Should complete within 1 second

    def test_should_route_creative_prompt_end_to_end(self):
        """Test complete routing pipeline for a creative writing prompt."""
        # Setup: Mock registry returns creative-capable models
        creative_models = [
            self._create_mock_model("gpt-4", "openai", ["creative"]),
            self._create_mock_model("claude-3", "anthropic", ["creative"])
        ]
        self.mock_registry.get_models_for_category.return_value = creative_models

        # Mock ranker with creative-focused scoring
        mock_ranking = Mock()
        mock_ranking.ranked_models = creative_models
        mock_ranking.ranking_scores = [0.85, 0.75]
        self.mock_ranker.rank_models.return_value = mock_ranking

        # Test: Route a creative prompt
        prompt = "Write a short story about a brave knight"
        result = self.router.route(prompt)

        # Verify: Creative routing decision
        assert isinstance(result, RoutingDecision)
        assert result.selected_model.provider == "openai"
        assert result.selected_model.model == "gpt-4"
        assert result.classification.category == "creative"
        assert "creative" in result.reasoning.lower()
        assert "story" in result.reasoning.lower()

    def test_should_route_qa_prompt_end_to_end(self):
        """Test complete routing pipeline for a question-answering prompt."""
        # Setup: Mock registry returns QA-capable models
        qa_models = [
            self._create_mock_model("gpt-3.5-turbo", "openai", ["qa"]),
            self._create_mock_model("claude-2", "anthropic", ["qa"])
        ]
        self.mock_registry.get_models_for_category.return_value = qa_models

        # Mock ranker with QA-focused scoring
        mock_ranking = Mock()
        mock_ranking.ranked_models = qa_models
        mock_ranking.ranking_scores = [0.8, 0.7]
        self.mock_ranker.rank_models.return_value = mock_ranking

        # Test: Route a QA prompt
        prompt = "What is the capital of France?"
        result = self.router.route(prompt)

        # Verify: QA routing decision
        assert isinstance(result, RoutingDecision)
        assert result.selected_model.provider == "openai"
        assert result.selected_model.model == "gpt-3.5-turbo"
        assert result.classification.category == "qa"
        assert "qa" in result.reasoning.lower() or "question" in result.reasoning.lower()

    def test_should_handle_mixed_category_prompt_with_precedence(self):
        """Test routing when prompt contains keywords from multiple categories."""
        # Setup: Mock registry returns models for the winning category (code)
        code_models = [self._create_mock_model("gpt-4", "openai", ["code"])]
        self.mock_registry.get_models_for_category.return_value = code_models

        mock_ranking = Mock()
        mock_ranking.ranked_models = code_models
        mock_ranking.ranking_scores = [0.7]
        self.mock_ranker.rank_models.return_value = mock_ranking

        # Test: Route a mixed prompt (contains creative + code + qa keywords)
        prompt = "How do I write a Python function that tells a creative story?"
        result = self.router.route(prompt)

        # Verify: Should classify to one category (likely code due to "python function")
        assert isinstance(result, RoutingDecision)
        assert result.classification.category in ["code", "creative", "qa"]
        # Should still work even with mixed keywords
        assert result.confidence > 0

    def test_should_return_none_when_no_models_available(self):
        """Test that router returns None when no models are available for the category."""
        # Setup: Mock registry returns empty list
        self.mock_registry.get_models_for_category.return_value = []

        # Test: Route any prompt
        prompt = "Write a function"
        result = self.router.route(prompt)

        # Verify: Returns None when no models available
        assert result is None

    def test_should_measure_routing_performance(self):
        """Test that routing completes within acceptable time limits."""
        # Setup: Mock fast registry and ranker responses
        code_models = [self._create_mock_model("gpt-4", "openai", ["code"])]
        self.mock_registry.get_models_for_category.return_value = code_models

        mock_ranking = Mock()
        mock_ranking.ranked_models = code_models
        mock_ranking.ranking_scores = [0.8]
        self.mock_ranker.rank_models.return_value = mock_ranking

        # Test: Measure routing performance
        prompt = "Write a function"
        import time
        start_time = time.time()

        result = self.router.route(prompt)

        end_time = time.time()
        routing_duration = end_time - start_time

        # Verify: Routing completes successfully
        assert isinstance(result, RoutingDecision)

        # Performance: Should complete within reasonable time (under 1 second)
        assert routing_duration < 1.0, f"Routing took {routing_duration:.3f}s, expected < 1.0s"

        # The routing_time_ms in result should also be reasonable
        assert 0 < result.routing_time_ms < 1000  # Under 1 second in milliseconds

    def test_should_maintain_consistency_across_multiple_routes(self):
        """Test that routing maintains consistency across multiple sequential routes."""
        # Setup: Mock registry with consistent responses
        code_models = [self._create_mock_model("gpt-4", "openai", ["code"])]
        self.mock_registry.get_models_for_category.return_value = code_models

        mock_ranking = Mock()
        mock_ranking.ranked_models = code_models
        mock_ranking.ranking_scores = [0.8]
        self.mock_ranker.rank_models.return_value = mock_ranking

        # Test: Route multiple prompts sequentially
        prompts = [
            "Write a function",
            "Debug this code",
            "Create an algorithm"
        ]

        results = []
        for prompt in prompts:
            result = self.router.route(prompt)
            results.append(result)

        # Verify: All routes successful and consistent
        for i, result in enumerate(results):
            assert isinstance(result, RoutingDecision), f"Route {i} failed"
            assert result.selected_model.provider == "openai"
            assert result.selected_model.model == "gpt-4"
            assert result.classification.category == "code"

    def test_should_handle_real_classification_edge_cases(self):
        """Test routing with real classifier edge cases."""
        # Setup: Mock registry for any category
        self.mock_registry.get_models_for_category.return_value = [
            self._create_mock_model("gpt-4", "openai", ["code", "creative", "qa"])
        ]

        mock_ranking = Mock()
        mock_ranking.ranked_models = [self._create_mock_model("gpt-4", "openai", ["code"])]
        mock_ranking.ranking_scores = [0.6]
        self.mock_ranker.rank_models.return_value = mock_ranking

        # Test: Various edge case prompts that real classifier handles
        edge_cases = [
            "",  # Empty prompt
            "   ",  # Whitespace only
            "Hello world",  # Generic prompt
            "A" * 1000,  # Very long prompt
        ]

        for prompt in edge_cases:
            result = self.router.route(prompt)

            # Should handle gracefully (may return None for some cases)
            # The important thing is no crashes or exceptions
            if result is not None:
                assert isinstance(result, RoutingDecision)

    def _create_mock_model(self, model_name: str, provider: str, capabilities: list) -> Mock:
        """Helper to create mock model objects."""
        mock_model = Mock()
        mock_model.model = model_name
        mock_model.provider = provider
        mock_model.capabilities = capabilities
        mock_model.estimated_cost = 0.002
        mock_model.estimated_latency = 500.0
        mock_model.quality_match = 0.8
        return mock_model
