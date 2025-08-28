"""Scoring Engine for Phase 3.1: Basic Scoring Function.

This module provides the core scoring algorithm that balances cost, latency, and quality
for optimal model selection in LLM routing.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from llm_router.registry import ProviderModel
from llm_router.models import VALID_CATEGORIES





class ScoringWeights(BaseModel):
    """Configuration for scoring weights to balance different factors."""

    cost_weight: float = Field(default=1/3, ge=0.0, le=1.0, description="Weight for cost optimization")
    latency_weight: float = Field(default=1/3, ge=0.0, le=1.0, description="Weight for latency optimization")
    quality_weight: float = Field(default=1/3, ge=0.0, le=1.0, description="Weight for quality optimization")

    @field_validator("cost_weight", "latency_weight", "quality_weight")
    @classmethod
    def validate_weight_bounds(cls, v: float) -> float:
        """Validate that individual weights are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "ScoringWeights":
        """Validate that weights sum to approximately 1.0."""
        total = self.cost_weight + self.latency_weight + self.quality_weight
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError("Weights must sum to 1.0")
        return self

    @property
    def total_weight(self) -> float:
        """Get the total weight (should always be 1.0)."""
        return self.cost_weight + self.latency_weight + self.quality_weight


class ScoringResult(BaseModel):
    """Result of scoring calculation for a model."""

    overall_score: float = Field(ge=0.0, le=1.0, description="Overall weighted score")
    cost_score: float = Field(ge=0.0, le=1.0, description="Cost optimization score (0=expensive, 1=cheap)")
    latency_score: float = Field(ge=0.0, le=1.0, description="Latency optimization score (0=slow, 1=fast)")
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality match score (0=poor, 1=excellent)")
    normalized_cost: float = Field(description="Normalized cost value for comparison")
    normalized_latency: float = Field(description="Normalized latency value for comparison")
    normalized_quality: float = Field(ge=0.0, le=1.0, description="Normalized quality value for comparison")

    @field_validator("overall_score", "cost_score", "latency_score", "quality_score")
    @classmethod
    def validate_score_bounds(cls, v: float) -> float:
        """Validate that scores are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v


class ScoringEngine:
    """Core scoring engine for model selection."""

    def __init__(self):
        """Initialize the scoring engine."""
        # Reference values for normalization (can be updated based on registry data)
        self._reference_cost = 0.01  # $0.01 per 1K tokens as baseline
        self._reference_latency = 100.0  # 100ms as baseline
        self._max_acceptable_latency = 10000.0  # 10 seconds as maximum

    def calculate_score(
        self,
        model: ProviderModel,
        category: str,
        weights: Optional[ScoringWeights] = None,
        estimated_tokens: int = 1000
    ) -> ScoringResult:
        """
        Calculate comprehensive score for a model based on cost, latency, and quality.
        
        Args:
            model: ProviderModel to score
            category: Task category for quality scoring
            weights: Optional scoring weights (uses defaults if None)
            estimated_tokens: Estimated token count for cost calculation
            
        Returns:
            ScoringResult with all scoring components
            
        Raises:
            ValueError: If category is invalid or weights are invalid
        """
        # Validate inputs
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Valid categories: {VALID_CATEGORIES}")
        
        if weights is None:
            weights = ScoringWeights()
        
        # Calculate individual scores
        cost_score = self._calculate_cost_score(model, estimated_tokens)
        latency_score = self._calculate_latency_score(model)
        quality_score = self._calculate_quality_score(model, category)
        
        # Normalize values for comparison
        normalized_cost = self._normalize_cost(model.pricing, estimated_tokens)
        normalized_latency = self._normalize_latency(model.performance.avg_latency_ms)
        normalized_quality = quality_score  # Already normalized 0-1
        
        # Calculate weighted overall score
        overall_score = (
            weights.cost_weight * cost_score +
            weights.latency_weight * latency_score +
            weights.quality_weight * quality_score
        )
        
        return ScoringResult(
            overall_score=overall_score,
            cost_score=cost_score,
            latency_score=latency_score,
            quality_score=quality_score,
            normalized_cost=normalized_cost,
            normalized_latency=normalized_latency,
            normalized_quality=normalized_quality
        )

    def _calculate_cost_score(self, model: ProviderModel, estimated_tokens: int) -> float:
        """Calculate cost score (0=expensive, 1=cheap)."""
        # Calculate total cost for estimated tokens
        input_cost = (estimated_tokens / 1000) * model.pricing.input_tokens_per_1k
        output_cost = (estimated_tokens / 1000) * model.pricing.output_tokens_per_1k
        total_cost = input_cost + output_cost
        
        # Handle zero cost (e.g., local models)
        if total_cost == 0.0:
            return 1.0
        
        # Score based on cost relative to reference
        if total_cost <= self._reference_cost:
            return 1.0  # Very cheap
        elif total_cost <= self._reference_cost * 10:
            return 0.8  # Cheap
        elif total_cost <= self._reference_cost * 100:
            return 0.6  # Moderate
        elif total_cost <= self._reference_cost * 1000:
            return 0.4  # Expensive
        else:
            return 0.2  # Very expensive

    def _calculate_latency_score(self, model: ProviderModel) -> float:
        """Calculate latency score (0=slow, 1=fast)."""
        latency = model.performance.avg_latency_ms
        
        # Handle edge cases
        if latency <= 0:
            return 1.0  # Instant (theoretical)
        elif latency == float('inf'):
            return 0.0  # Infinite latency
        
        # Score based on latency relative to reference
        if latency <= self._reference_latency:
            return 1.0  # Very fast
        elif latency <= self._reference_latency * 2:
            return 0.9  # Fast
        elif latency <= self._reference_latency * 5:
            return 0.7  # Moderate
        elif latency <= self._reference_latency * 10:
            return 0.5  # Slow
        elif latency <= self._max_acceptable_latency:
            return 0.3  # Very slow but acceptable
        else:
            return 0.1  # Unacceptably slow

    def _calculate_quality_score(self, model: ProviderModel, category: str) -> float:
        """Calculate quality score (0=poor, 1=excellent) for a specific category."""
        # Check if model has the required capability
        if category not in model.capabilities:
            return 0.0  # Cannot handle this category
        
        # Get quality score for the category
        if model.performance.quality_scores is None:
            return 0.5  # Default neutral score if no quality data
        
        quality_score = model.performance.quality_scores.get(category, 0.5)
        
        # Validate quality score bounds
        if not 0.0 <= quality_score <= 1.0:
            return 0.5  # Default if invalid
        
        return quality_score

    def _normalize_cost(self, pricing: Any, estimated_tokens: int) -> float:
        """Normalize cost to a 0-1 scale for comparison."""
        input_cost = (estimated_tokens / 1000) * pricing.input_tokens_per_1k
        output_cost = (estimated_tokens / 1000) * pricing.output_tokens_per_1k
        total_cost = input_cost + output_cost
        
        # Handle zero cost
        if total_cost == 0.0:
            return 0.0
        
        # Normalize relative to reference cost
        normalized = min(total_cost / self._reference_cost, 1.0)
        return normalized

    def _normalize_latency(self, latency_ms: float) -> float:
        """Normalize latency to a 0-1 scale for comparison."""
        if latency_ms <= 0:
            return 0.0
        elif latency_ms == float('inf'):
            return 1.0
        
        # Normalize relative to max acceptable latency
        normalized = min(latency_ms / self._max_acceptable_latency, 1.0)
        return normalized

    def update_reference_values(self, models: list[ProviderModel]) -> None:
        """
        Update reference values based on actual model data.
        
        Args:
            models: List of ProviderModel instances to analyze
        """
        if not models:
            return
        
        # Update reference cost based on actual pricing
        costs = []
        latencies = []
        
        for model in models:
            # Collect cost data
            total_cost = model.pricing.input_tokens_per_1k + model.pricing.output_tokens_per_1k
            if total_cost > 0:
                costs.append(total_cost)
            
            # Collect latency data
            if model.performance.avg_latency_ms > 0 and model.performance.avg_latency_ms != float('inf'):
                latencies.append(model.performance.avg_latency_ms)
        
        # Update reference values if we have data
        if costs:
            self._reference_cost = min(costs)  # Use cheapest as baseline
        
        if latencies:
            self._reference_latency = min(latencies)  # Use fastest as baseline
