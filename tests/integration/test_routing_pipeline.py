"""Integration tests for the routing pipeline.

These tests verify that different components work together correctly:
- Constraint validation + ranking
- Scoring + ranking
- Complete routing flow
"""

import pytest
from typing import List

from llm_router.ranking import ModelRanker
from llm_router.constraints import ConstraintValidator, RoutingConstraints
from llm_router.scoring import ScoringWeights
from llm_router.registry import ProviderModel, PricingInfo, LimitsInfo, PerformanceInfo


class TestRoutingPipelineIntegration:
    """Test integration between routing components."""

    def test_constraint_filtering_then_ranking_integration(self):
        """Test that constraints properly filter models before ranking."""
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

    def test_scoring_then_ranking_integration(self):
        """Test that scoring engine integrates with ranking system."""
        ranker = ModelRanker()
        
        # Create models with different characteristics that will result in different scores
        models = [
            self._create_test_model("model_a", cost=0.01, latency=1000, quality=0.7),  # Lower score
            self._create_test_model("model_b", cost=0.001, latency=100, quality=0.9),  # Higher score
            self._create_test_model("model_c", cost=0.005, latency=500, quality=0.8),  # Medium score
        ]
        
        # Rank models (this internally uses ScoringEngine)
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

    def test_custom_weights_integration(self):
        """Test that custom scoring weights affect ranking through the pipeline."""
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

    def test_performance_measurement_integration(self):
        """Test that performance measurement works through the ranking pipeline."""
        ranker = ModelRanker()
        
        # Create multiple models for performance testing
        models = [self._create_test_model(f"model_{i}", cost=0.005, latency=500, quality=0.8) for i in range(10)]
        
        result = ranker.rank_models(models, "code")
        
        # Should measure ranking time
        assert result.ranking_time_ms > 0
        assert result.ranking_time_ms < 1000  # Should be fast (< 1 second)
        assert result.total_candidates == 10

    def test_large_model_list_integration(self):
        """Test that the pipeline handles large numbers of models efficiently."""
        ranker = ModelRanker()
        
        # Create a larger number of models
        models = [self._create_test_model(f"model_{i}", cost=0.005, latency=500, quality=0.8) for i in range(100)]
        
        result = ranker.rank_models(models, "code")
        
        # Should handle large lists efficiently
        assert result.total_candidates == 100
        assert len(result.ranked_models) == 100
        assert len(result.ranking_scores) == 100
        assert result.ranking_time_ms < 5000  # Should be reasonably fast even with 100 models

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
