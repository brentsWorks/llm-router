"""Tests for Phase 3.1: Basic Scoring Function.

This module tests the core scoring algorithm that balances cost, latency, and quality
for optimal model selection in LLM routing.
"""

import pytest
from unittest.mock import Mock
from typing import Dict, Any

from llm_router.scoring import ScoringEngine, ScoringWeights, ScoringResult
from llm_router.utils import format_validation_error
from llm_router.registry import ProviderModel, PricingInfo, LimitsInfo, PerformanceInfo
from llm_router.models import Category


class TestScoringWeights:
    """Test ScoringWeights configuration and validation."""

    def test_should_create_valid_scoring_weights(self):
        """Test that ScoringWeights can be created with valid values."""
        weights = ScoringWeights(
            cost_weight=0.3,
            latency_weight=0.4,
            quality_weight=0.3
        )
        
        assert weights.cost_weight == 0.3
        assert weights.latency_weight == 0.4
        assert weights.quality_weight == 0.3
        assert weights.total_weight == 1.0

    def test_should_validate_weights_sum_to_one(self):
        """Test that weights must sum to approximately 1.0."""
        # Valid weights
        ScoringWeights(cost_weight=0.5, latency_weight=0.3, quality_weight=0.2)
        
        # Invalid weights - should raise validation error
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ScoringWeights(cost_weight=0.5, latency_weight=0.5, quality_weight=0.5)

    def test_should_validate_individual_weight_bounds(self):
        """Test that individual weights must be between 0 and 1."""
        # Valid weights
        ScoringWeights(cost_weight=0.0, latency_weight=0.5, quality_weight=0.5)
        ScoringWeights(cost_weight=1.0, latency_weight=0.0, quality_weight=0.0)
        
        # Invalid weights - should raise validation error with clear messages
        with pytest.raises(Exception) as exc_info:
            ScoringWeights(cost_weight=-0.1, latency_weight=0.6, quality_weight=0.5)
        
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "cost_weight" in error_message
        assert "greater than or equal to" in error_message
        
        with pytest.raises(Exception) as exc_info:
            ScoringWeights(cost_weight=1.1, latency_weight=0.0, quality_weight=-0.1)
        
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "cost_weight" in error_message
        assert "less than or equal to" in error_message

    def test_should_use_default_weights_when_none_provided(self):
        """Test that default weights are used when none provided."""
        weights = ScoringWeights()
        
        # Should use equal weights by default
        assert weights.cost_weight == pytest.approx(1/3, rel=1e-2)
        assert weights.latency_weight == pytest.approx(1/3, rel=1e-2)
        assert weights.quality_weight == pytest.approx(1/3, rel=1e-2)
        assert weights.total_weight == pytest.approx(1.0, rel=1e-2)


class TestScoringResult:
    """Test ScoringResult data structure."""

    def test_should_create_valid_scoring_result(self):
        """Test that ScoringResult can be created with valid data."""
        result = ScoringResult(
            overall_score=0.85,
            cost_score=0.9,
            latency_score=0.8,
            quality_score=0.85,
            normalized_cost=0.02,
            normalized_latency=150.0,
            normalized_quality=0.85
        )
        
        assert result.overall_score == 0.85
        assert result.cost_score == 0.9
        assert result.latency_score == 0.8
        assert result.quality_score == 0.85
        assert result.normalized_cost == 0.02
        assert result.normalized_latency == 150.0
        assert result.normalized_quality == 0.85

    def test_should_validate_score_bounds(self):
        """Test that scores must be between 0 and 1."""
        # Valid scores
        ScoringResult(
            overall_score=0.0,
            cost_score=0.5,
            latency_score=0.5,
            quality_score=0.5,
            normalized_cost=0.01,
            normalized_latency=100.0,
            normalized_quality=0.5
        )
        
        # Invalid scores - should raise validation error with clear messages
        with pytest.raises(Exception) as exc_info:
            ScoringResult(
                overall_score=-0.1,
                cost_score=0.5,
                latency_score=0.5,
                quality_score=0.5,
                normalized_cost=0.01,
                normalized_latency=100.0,
                normalized_quality=0.5
            )
        
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "overall_score" in error_message
        assert "greater than or equal to" in error_message


class TestScoringEngine:
    """Test the core ScoringEngine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scoring_engine = ScoringEngine()
        
        # Create sample provider models for testing
        self.gpt4_model = ProviderModel(
            provider="openai",
            model="gpt-4",
            capabilities=["code", "reasoning", "creative"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.03,
                output_tokens_per_1k=0.06
            ),
            limits=LimitsInfo(
                context_length=8192,
                safety_level="high"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=200.0,
                quality_scores={
                    "code": 0.95,
                    "reasoning": 0.9,
                    "creative": 0.85
                }
            )
        )
        
        self.gpt35_model = ProviderModel(
            provider="openai",
            model="gpt-3.5-turbo",
            capabilities=["code", "qa", "summarization"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.0015,
                output_tokens_per_1k=0.002
            ),
            limits=LimitsInfo(
                context_length=4096,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=100.0,
                quality_scores={
                    "code": 0.7,
                    "qa": 0.8,
                    "summarization": 0.75
                }
            )
        )

    def test_should_calculate_score_with_different_weights(self):
        """Test scoring calculation with different weight configurations."""
        # Cost-focused weights
        cost_weights = ScoringWeights(cost_weight=0.6, latency_weight=0.2, quality_weight=0.2)
        cost_result = self.scoring_engine.calculate_score(
            self.gpt35_model, 
            category="code",
            weights=cost_weights
        )
        
        # Quality-focused weights
        quality_weights = ScoringWeights(cost_weight=0.2, latency_weight=0.2, quality_weight=0.6)
        quality_result = self.scoring_engine.calculate_score(
            self.gpt4_model,
            category="code", 
            weights=quality_weights
        )
        
        # Cost-focused should favor cheaper model
        assert cost_result.overall_score > quality_result.overall_score
        
        # Quality-focused should favor higher quality model
        assert quality_result.quality_score > cost_result.quality_score

    def test_should_handle_edge_case_zero_cost(self):
        """Test scoring with zero cost models (e.g., local models)."""
        zero_cost_model = ProviderModel(
            provider="local",
            model="llama-7b",
            capabilities=["code", "qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.0,
                output_tokens_per_1k=0.0
            ),
            limits=LimitsInfo(
                context_length=4096,
                safety_level="low"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=2000.0,  # Slower local inference
                quality_scores={
                    "code": 0.6,
                    "qa": 0.5
                }
            )
        )
        
        result = self.scoring_engine.calculate_score(
            zero_cost_model,
            category="code"
        )
        
        # Should handle zero cost gracefully
        assert result.cost_score == 1.0  # Perfect cost score
        assert result.normalized_cost == 0.0
        assert result.overall_score > 0.0

    def test_should_handle_edge_case_infinite_latency(self):
        """Test scoring with extremely high latency models."""
        high_latency_model = ProviderModel(
            provider="slow-provider",
            model="slow-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.001,
                output_tokens_per_1k=0.002
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=float('inf'),  # Infinite latency
                quality_scores={"code": 0.8}
            )
        )
        
        result = self.scoring_engine.calculate_score(
            high_latency_model,
            category="code"
        )
        
        # Should handle infinite latency gracefully
        assert result.latency_score == 0.0  # Worst latency score
        # Overall score depends on weights, but should be lower due to poor latency
        assert result.overall_score < 0.8  # Overall score should be lower than perfect

    def test_should_normalize_metrics_properly(self):
        """Test that metrics are properly normalized for comparison."""
        # Test with models that have very different scales
        cheap_fast_model = ProviderModel(
            provider="cheap",
            model="fast",
            capabilities=["qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.0001,  # Very cheap
                output_tokens_per_1k=0.0002
            ),
            limits=LimitsInfo(
                context_length=1024,
                safety_level="low"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=50.0,  # Very fast
                quality_scores={"qa": 0.6}  # Lower quality
            )
        )
        
        expensive_slow_model = ProviderModel(
            provider="expensive",
            model="slow",
            capabilities=["qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.1,  # Very expensive
                output_tokens_per_1k=0.2
            ),
            limits=LimitsInfo(
                context_length=8192,
                safety_level="high"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=1000.0,  # Very slow
                quality_scores={"qa": 0.95}  # Higher quality
            )
        )
        
        cheap_result = self.scoring_engine.calculate_score(
            cheap_fast_model,
            category="qa"
        )
        
        expensive_result = self.scoring_engine.calculate_score(
            expensive_slow_model,
            category="qa"
        )
        
        # Normalized values should be comparable
        assert 0.0 <= cheap_result.normalized_cost <= 1.0
        assert 0.0 <= expensive_result.normalized_cost <= 1.0
        assert 0.0 <= cheap_result.normalized_latency <= 1.0
        assert 0.0 <= expensive_result.normalized_latency <= 1.0

    def test_should_handle_missing_quality_scores(self):
        """Test scoring when quality scores are missing."""
        model_without_quality = ProviderModel(
            provider="basic",
            model="basic-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.01,
                output_tokens_per_1k=0.02
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=300.0,
                quality_scores=None  # No quality scores
            )
        )
        
        result = self.scoring_engine.calculate_score(
            model_without_quality,
            category="code"
        )
        
        # Should handle missing quality scores gracefully
        assert result.quality_score == 0.5  # Default neutral score
        assert result.overall_score > 0.0  # Should still calculate overall score

    def test_should_validate_input_parameters(self):
        """Test input validation for scoring function."""
        # Valid inputs should work
        self.scoring_engine.calculate_score(
            self.gpt4_model,
            category="code"
        )
        
        # Invalid category should raise error
        with pytest.raises(ValueError, match="Invalid category"):
            self.scoring_engine.calculate_score(
                self.gpt4_model,
                category="invalid_category"
            )
        
        # Invalid weights should raise error with clear messages
        with pytest.raises(Exception) as exc_info:
            invalid_weights = ScoringWeights(cost_weight=0.5, latency_weight=0.5, quality_weight=0.5)
            self.scoring_engine.calculate_score(
                self.gpt4_model,
                category="code",
                weights=invalid_weights
            )
        
        # Should get clear error message about weights not summing to 1.0
        error_message = format_validation_error(exc_info.value)
        assert "Weights must sum to 1.0" in str(exc_info.value)

    def test_should_handle_very_expensive_models(self):
        """Test scoring with extremely expensive models."""
        very_expensive_model = ProviderModel(
            provider="premium",
            model="ultra-premium",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=1.0,  # Very expensive
                output_tokens_per_1k=2.0
            ),
            limits=LimitsInfo(
                context_length=8192,
                safety_level="high"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=500.0,
                quality_scores={"code": 0.99}
            )
        )
        
        result = self.scoring_engine.calculate_score(
            very_expensive_model,
            category="code"
        )
        
        # Should get low cost score for very expensive models
        # Total cost = (1000/1000) * (1.0 + 2.0) = 3.0
        # This should fall into the expensive tier (0.4), not very expensive (0.2)
        assert result.cost_score == 0.4  # Expensive score
        assert result.overall_score < 0.9  # Overall score should be lower

    def test_should_handle_unacceptably_slow_models(self):
        """Test scoring with extremely slow models."""
        very_slow_model = ProviderModel(
            provider="slow",
            model="ultra-slow",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.001,
                output_tokens_per_1k=0.002
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=15000.0,  # 15 seconds - unacceptably slow
                quality_scores={"code": 0.8}
            )
        )
        
        result = self.scoring_engine.calculate_score(
            very_slow_model,
            category="code"
        )
        
        # Should get lowest latency score for unacceptably slow models
        assert result.latency_score == 0.1  # Unacceptably slow score

    def test_should_handle_missing_quality_scores_for_category(self):
        """Test scoring when quality scores are missing for a specific category."""
        model_with_missing_category = ProviderModel(
            provider="missing-category",
            model="test-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.01,
                output_tokens_per_1k=0.02
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=300.0,
                quality_scores={"qa": 0.8}  # Has qa but not code
            )
        )
        
        result = self.scoring_engine.calculate_score(
            model_with_missing_category,
            category="code"
        )
        
        # Should handle missing category gracefully
        assert result.quality_score == 0.5  # Default neutral score

    def test_should_handle_quality_scores_at_bounds(self):
        """Test scoring with quality scores at the boundary values."""
        boundary_quality_model = ProviderModel(
            provider="boundary",
            model="test-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.01,
                output_tokens_per_1k=0.02
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=300.0,
                quality_scores={"code": 0.0}  # At minimum boundary
            )
        )
        
        result = self.scoring_engine.calculate_score(
            boundary_quality_model,
            category="code"
        )
        
        # Should handle boundary quality scores correctly
        assert result.quality_score == 0.0  # Minimum quality score

    def test_should_handle_zero_cost_normalization(self):
        """Test cost normalization with zero cost models."""
        zero_cost_model = ProviderModel(
            provider="free",
            model="free-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.0,
                output_tokens_per_1k=0.0
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="low"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=1000.0,
                quality_scores={"code": 0.6}
            )
        )
        
        result = self.scoring_engine.calculate_score(
            zero_cost_model,
            category="code"
        )
        
        # Should handle zero cost normalization gracefully
        assert result.normalized_cost == 0.0

    def test_should_handle_infinite_latency_normalization(self):
        """Test latency normalization with infinite latency."""
        infinite_latency_model = ProviderModel(
            provider="infinite",
            model="test-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.001,
                output_tokens_per_1k=0.002
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=float('inf'),
                quality_scores={"code": 0.8}
            )
        )
        
        result = self.scoring_engine.calculate_score(
            infinite_latency_model,
            category="code"
        )
        
        # Should handle infinite latency normalization gracefully
        assert result.normalized_latency == 1.0

    def test_should_update_reference_values(self):
        """Test updating reference values based on model data."""
        # Create models with different pricing and latency
        cheap_fast_model = ProviderModel(
            provider="cheap",
            model="fast",
            capabilities=["qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.0001,  # Very cheap
                output_tokens_per_1k=0.0002
            ),
            limits=LimitsInfo(
                context_length=1024,
                safety_level="low"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=50.0,  # Very fast
                quality_scores={"qa": 0.6}
            )
        )
        
        expensive_slow_model = ProviderModel(
            provider="expensive",
            model="slow",
            capabilities=["qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.1,  # Very expensive
                output_tokens_per_1k=0.2
            ),
            limits=LimitsInfo(
                context_length=8192,
                safety_level="high"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=1000.0,  # Very slow
                quality_scores={"qa": 0.95}
            )
        )
        
        # Update reference values
        self.scoring_engine.update_reference_values([cheap_fast_model, expensive_slow_model])
        
        # Reference values should be updated to the best (cheapest/fastest) values
        # Total cost = input + output = 0.0001 + 0.0002 = 0.0003
        assert self.scoring_engine._reference_cost == pytest.approx(0.0003, rel=1e-10)  # Cheapest total cost
        assert self.scoring_engine._reference_latency == 50.0  # Fastest latency

    def test_should_handle_empty_models_list_in_update_reference_values(self):
        """Test update_reference_values with empty models list."""
        # Should not crash with empty list
        self.scoring_engine.update_reference_values([])
        
        # Reference values should remain unchanged
        assert self.scoring_engine._reference_cost == 0.01  # Default value
        assert self.scoring_engine._reference_latency == 100.0  # Default value

    def test_should_handle_models_with_zero_costs_in_update_reference_values(self):
        """Test update_reference_values with models that have zero costs."""
        zero_cost_model = ProviderModel(
            provider="free",
            model="free-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.0,
                output_tokens_per_1k=0.0
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="low"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=1000.0,
                quality_scores={"code": 0.6}
            )
        )
        
        # Update reference values
        self.scoring_engine.update_reference_values([zero_cost_model])
        
        # Should skip zero cost models when updating reference cost
        # Reference cost should remain unchanged since we only consider positive costs
        assert self.scoring_engine._reference_cost == 0.01  # Default value

    def test_should_handle_models_with_infinite_latency_in_update_reference_values(self):
        """Test update_reference_values with models that have infinite latency."""
        infinite_latency_model = ProviderModel(
            provider="infinite",
            model="test-model",
            capabilities=["code"],
            pricing=PricingInfo(
                input_tokens_per_1k=0.001,
                output_tokens_per_1k=0.002
            ),
            limits=LimitsInfo(
                context_length=2048,
                safety_level="medium"
            ),
            performance=PerformanceInfo(
                avg_latency_ms=float('inf'),
                quality_scores={"code": 0.8}
            )
        )
        
        # Update reference values
        self.scoring_engine.update_reference_values([infinite_latency_model])
        
        # Should skip infinite latency models when updating reference latency
        # Reference latency should remain unchanged since we only consider finite latencies
        assert self.scoring_engine._reference_latency == 100.0  # Default value
