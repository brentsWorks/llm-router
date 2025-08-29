"""Tests for Phase 3.3: Model Ranking.

This module tests the model ranking system that sorts models by score
after constraint validation for optimal model selection.
"""

import pytest
from typing import List
from unittest.mock import Mock
from pydantic import ValidationError

from llm_router.ranking import ModelRanker, RankingResult, RankingStrategy
from llm_router.scoring import ScoringEngine, ScoringWeights, ScoringResult
from llm_router.constraints import ConstraintValidator, RoutingConstraints
from llm_router.registry import ProviderModel, PricingInfo, LimitsInfo, PerformanceInfo
from llm_router.models import Category
from llm_router.utils import format_validation_error


class TestRankingResult:
    """Test RankingResult data model."""

    def test_should_create_ranking_result(self):
        """Test that RankingResult can be created with ranking data."""
        from llm_router.ranking import RankingResult
        
        # Create actual ProviderModel instances for testing
        models = [
            self._create_test_model("model_a"),
            self._create_test_model("model_b"),
            self._create_test_model("model_c"),
        ]
        
        result = RankingResult(
            ranked_models=models,
            ranking_scores=[0.95, 0.87, 0.72],
            total_candidates=3,
            ranking_time_ms=15.5
        )
        
        assert len(result.ranked_models) == 3
        assert result.ranked_models[0].model == "model_a"
        assert result.ranked_models[1].model == "model_b"
        assert result.ranked_models[2].model == "model_c"
        assert result.ranking_scores == [0.95, 0.87, 0.72]
        assert result.total_candidates == 3
        assert result.ranking_time_ms == 15.5

    def test_should_validate_ranking_scores_match_models(self):
        """Test that ranking scores count matches ranked models count."""
        from llm_router.ranking import RankingResult
        
        # Valid: matching counts
        models = [
            self._create_test_model("model_a"),
            self._create_test_model("model_b"),
        ]
        
        RankingResult(
            ranked_models=models,
            ranking_scores=[0.95, 0.87],
            total_candidates=2,
            ranking_time_ms=10.0
        )
        
        # Invalid: mismatched counts
        with pytest.raises(ValidationError) as exc_info:
            RankingResult(
                ranked_models=models,
                ranking_scores=[0.95],  # Only 1 score for 2 models
                total_candidates=2,
                ranking_time_ms=10.0
            )
        
        # Should get clear error message using our utility
        error_message = format_validation_error(exc_info.value)
        assert "ranking_scores" in error_message or "Ranking data mismatch" in error_message

    def test_should_validate_total_candidates_consistency(self):
        """Test that total_candidates matches the actual number of models."""
        from llm_router.ranking import RankingResult
        
        models = [
            self._create_test_model("model_a"),
            self._create_test_model("model_b"),
        ]
        
        # Valid: matching total_candidates
        RankingResult(
            ranked_models=models,
            ranking_scores=[0.95, 0.87],
            total_candidates=2,
            ranking_time_ms=10.0
        )
        
        # Invalid: mismatched total_candidates
        with pytest.raises(ValidationError) as exc_info:
            RankingResult(
                ranked_models=models,
                ranking_scores=[0.95, 0.87],
                total_candidates=3,  # Wrong count
                ranking_time_ms=10.0
            )
        
        # Should get clear error message using our utility
        error_message = format_validation_error(exc_info.value)
        assert "total_candidates" in error_message or "Total candidates mismatch" in error_message

    def _create_test_model(self, model_name: str) -> ProviderModel:
        """Helper to create test ProviderModel instances."""
        return ProviderModel(
            provider="test_provider",
            model=model_name,
            capabilities=["code", "qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.005,
                output_tokens_per_1k=0.005
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="moderate"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=500,
                quality_scores={"code": 0.8, "qa": 0.8}
            )
        )


class TestRankingStrategy:
    """Test RankingStrategy enum values and usage."""

    def test_should_have_expected_strategy_values(self):
        """Test that RankingStrategy enum contains expected values."""
        assert RankingStrategy.SCORE_BASED == "score_based"
        assert RankingStrategy.COST_OPTIMIZED == "cost_optimized"
        assert RankingStrategy.LATENCY_OPTIMIZED == "latency_optimized"
        assert RankingStrategy.QUALITY_OPTIMIZED == "quality_optimized"

    def test_should_use_strategy_in_ranking(self):
        """Test that ranking strategy parameter is accepted and used."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        models = [
            self._create_test_model("model_a", cost=0.005, latency=500, quality=0.8),
            self._create_test_model("model_b", cost=0.005, latency=500, quality=0.8),
        ]
        
        # Test with different strategies
        result = ranker.rank_models(
            models, "code", strategy=RankingStrategy.SCORE_BASED
        )
        assert len(result.ranked_models) == 2
        
        result = ranker.rank_models(
            models, "code", strategy=RankingStrategy.COST_OPTIMIZED
        )
        assert len(result.ranked_models) == 2

    def _create_test_model(self, model_name: str, cost: float = 0.005, latency: int = 500, quality: float = 0.8) -> ProviderModel:
        """Helper to create test ProviderModel instances."""
        return ProviderModel(
            provider="test_provider",
            model=model_name,
            capabilities=["code", "qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=cost,
                output_tokens_per_1k=cost
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="moderate"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=latency,
                quality_scores={"code": quality, "qa": quality}
            )
        )


class TestModelRanker:
    """Test ModelRanker service."""

    def test_should_rank_models_by_score_descending(self):
        """Test that models are ranked from highest to lowest score."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Create test models with different characteristics that will result in different scores
        models = [
            self._create_test_model("model_a", cost=0.01, latency=1000, quality=0.7),  # Lower score
            self._create_test_model("model_b", cost=0.001, latency=100, quality=0.9),  # Higher score
            self._create_test_model("model_c", cost=0.005, latency=500, quality=0.8),  # Medium score
        ]
        
        # Rank models
        result = ranker.rank_models(models, "code")
        
        # Should be ranked by score (highest first)
        # model_b should have highest score due to low cost and latency
        assert result.ranked_models[0].model == "model_b"  # Highest score
        assert result.ranked_models[1].model == "model_c"  # Medium score  
        assert result.ranked_models[2].model == "model_a"  # Lowest score
        
        # Ranking scores should be in descending order
        assert result.ranking_scores[0] > result.ranking_scores[1]
        assert result.ranking_scores[1] > result.ranking_scores[2]
        assert result.total_candidates == 3

    def test_should_handle_models_with_identical_scores(self):
        """Test that models with identical scores are handled consistently."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Create models with very similar characteristics that might result in similar scores
        models = [
            self._create_test_model("model_a", cost=0.005, latency=500, quality=0.8),
            self._create_test_model("model_b", cost=0.005, latency=500, quality=0.8),
            self._create_test_model("model_c", cost=0.005, latency=500, quality=0.8),
        ]
        
        # Rank models
        result = ranker.rank_models(models, "code")
        
        # Should maintain consistent ordering for similar scores
        assert len(result.ranked_models) == 3
        assert len(result.ranking_scores) == 3
        
        # All models should be included (no duplicates or missing)
        model_names = {model.model for model in result.ranked_models}
        assert model_names == {"model_a", "model_b", "model_c"}

    def test_should_rank_single_model(self):
        """Test that ranking works with a single model."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Single model
        models = [self._create_test_model("single_model", cost=0.005, latency=500, quality=0.8)]
        
        result = ranker.rank_models(models, "code")
        
        assert len(result.ranked_models) == 1
        assert result.ranked_models[0].model == "single_model"
        assert result.total_candidates == 1
        # Score should be calculated based on the model's characteristics
        assert 0.0 <= result.ranking_scores[0] <= 1.0

    def test_should_handle_empty_model_list(self):
        """Test that ranking handles empty model lists gracefully."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Empty list
        models = []
        
        result = ranker.rank_models(models, "code")
        
        assert len(result.ranked_models) == 0
        assert len(result.ranking_scores) == 0
        assert result.total_candidates == 0

    def test_should_use_custom_scoring_weights(self):
        """Test that ranking respects custom scoring weights."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Create models with different characteristics
        models = [
            self._create_test_model("cheap_slow", cost=0.001, latency=1000, quality=0.7),
            self._create_test_model("expensive_fast", cost=0.01, latency=100, quality=0.9),
        ]
        
        # Cost-focused weights
        cost_weights = ScoringWeights(cost_weight=0.8, latency_weight=0.1, quality_weight=0.1)
        cost_result = ranker.rank_models(models, "code", weights=cost_weights)
        
        # Quality-focused weights
        quality_weights = ScoringWeights(cost_weight=0.1, latency_weight=0.1, quality_weight=0.8)
        quality_result = ranker.rank_models(models, "code", weights=quality_weights)
        
        # Should get different rankings based on weights
        assert cost_result.ranked_models[0].model == "cheap_slow"      # Lower cost
        assert quality_result.ranked_models[0].model == "expensive_fast"  # Higher quality

    def test_should_integrate_with_constraint_validation(self):
        """Test that ranking works with constraint-validated models."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        constraint_validator = ConstraintValidator()
        
        # Create models with different characteristics
        models = [
            self._create_test_model("valid_model", cost=0.005, latency=500, quality=0.8),
            self._create_test_model("expensive_model", cost=0.02, latency=500, quality=0.8),
            self._create_test_model("slow_model", cost=0.005, latency=2000, quality=0.8),
        ]
        
        # Apply constraints
        constraints = RoutingConstraints(
            max_cost_per_1k_tokens=0.01,
            max_latency_ms=1000
        )
        
        valid_models = constraint_validator.filter_valid_models(models, constraints)
        
        # Should only have 1 valid model (expensive and slow models filtered out)
        assert len(valid_models) == 1
        assert valid_models[0].model == "valid_model"
        
        # Rank the valid models
        result = ranker.rank_models(valid_models, "code")
        
        assert len(result.ranked_models) == 1
        assert result.ranked_models[0].model == "valid_model"
        assert result.total_candidates == 1

    def test_should_measure_ranking_performance(self):
        """Test that ranking performance is measured."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Create multiple models for performance testing
        models = [self._create_test_model(f"model_{i}", cost=0.005, latency=500, quality=0.8) for i in range(10)]
        
        result = ranker.rank_models(models, "code")
        
        # Should measure ranking time
        assert result.ranking_time_ms > 0
        assert result.ranking_time_ms < 1000  # Should be fast (< 1 second)
        assert result.total_candidates == 10

    def test_should_handle_large_model_lists(self):
        """Test that ranking handles large numbers of models efficiently."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Create a larger number of models
        models = [self._create_test_model(f"model_{i}", cost=0.005, latency=500, quality=0.8) for i in range(100)]
        
        result = ranker.rank_models(models, "code")
        
        # Should handle large lists efficiently
        assert result.total_candidates == 100
        assert len(result.ranked_models) == 100
        assert len(result.ranking_scores) == 100
        assert result.ranking_time_ms < 5000  # Should be reasonably fast even with 100 models

    def test_should_validate_input_parameters(self):
        """Test that ranking validates input parameters properly."""
        from llm_router.ranking import ModelRanker
        
        ranker = ModelRanker()
        
        # Test with invalid category
        models = [self._create_test_model("test_model")]
        
        with pytest.raises(ValueError) as exc_info:
            ranker.rank_models(models, "invalid_category")
        
        # Should get clear error message about invalid category
        assert "Invalid category" in str(exc_info.value)

    def _create_test_model(
        self,
        model_name: str,
        cost: float = 0.005,
        latency: int = 500,
        quality: float = 0.8
    ) -> ProviderModel:
        """Helper to create test ProviderModel instances."""
        return ProviderModel(
            provider="test_provider",
            model=model_name,
            capabilities=["code", "qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=cost,
                output_tokens_per_1k=cost
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="moderate"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=latency,
                quality_scores={"code": quality, "qa": quality}
            )
        )
