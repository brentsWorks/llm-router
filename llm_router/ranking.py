"""Model ranking system for Phase 3.3.

This module provides intelligent model ranking based on scoring results
and integrates with the scoring engine and constraint validation.
"""

import time
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

from llm_router.scoring import ScoringEngine, ScoringWeights
from llm_router.constraints import RoutingConstraints
from llm_router.registry import ProviderModel


class RankingStrategy(str, Enum):
    """Available ranking strategies."""
    
    SCORE_BASED = "score_based"  # Rank by overall score
    COST_OPTIMIZED = "cost_optimized"  # Prioritize cost
    LATENCY_OPTIMIZED = "latency_optimized"  # Prioritize speed
    QUALITY_OPTIMIZED = "quality_optimized"  # Prioritize quality


class RankingResult(BaseModel):
    """Result of model ranking operation."""
    
    ranked_models: List[ProviderModel] = Field(
        description="Models ranked from best to worst"
    )
    ranking_scores: List[float] = Field(
        description="Scores corresponding to ranked models"
    )
    total_candidates: int = Field(
        description="Total number of models that were ranked"
    )
    ranking_time_ms: float = Field(
        description="Time taken for ranking operation in milliseconds"
    )
    
    @model_validator(mode="after")
    def validate_ranking_data_consistency(self) -> "RankingResult":
        """Validate that ranking data is consistent."""
        if len(self.ranked_models) != len(self.ranking_scores):
            raise ValueError(
                f"Ranking data mismatch: {len(self.ranked_models)} models "
                f"but {len(self.ranking_scores)} scores"
            )
        
        if len(self.ranked_models) != self.total_candidates:
            raise ValueError(
                f"Total candidates mismatch: {len(self.ranked_models)} models "
                f"but total_candidates is {self.total_candidates}"
            )
        
        return self


class ModelRanker:
    """Service for ranking models by score."""
    
    def __init__(self):
        """Initialize the model ranker."""
        self.scoring_engine = ScoringEngine()
    
    def rank_models(
        self,
        models: List[ProviderModel],
        category: str,
        weights: Optional[ScoringWeights] = None,
        strategy: RankingStrategy = RankingStrategy.SCORE_BASED,
        estimated_tokens: int = 1000
    ) -> RankingResult:
        """
        Rank models by their scores for a specific category.
        
        Args:
            models: List of ProviderModel instances to rank
            category: Task category for scoring
            weights: Optional scoring weights (uses defaults if None)
            strategy: Ranking strategy to use
            estimated_tokens: Estimated token count for cost calculation
            
        Returns:
            RankingResult with ranked models and scores
            
        Raises:
            ValueError: If models list is empty or invalid
        """
        start_time = time.time()
        
        if not models:
            return RankingResult(
                ranked_models=[],
                ranking_scores=[],
                total_candidates=0,
                ranking_time_ms=0.0
            )
        
        # Score all models
        scored_models = []
        for model in models:
            score_result = self.scoring_engine.calculate_score(
                model, category, weights, estimated_tokens
            )
            scored_models.append((model, score_result.overall_score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Extract ranked models and scores
        ranked_models = [model for model, _ in scored_models]
        ranking_scores = [score for _, score in scored_models]
        
        # Calculate ranking time
        ranking_time_ms = (time.time() - start_time) * 1000
        
        return RankingResult(
            ranked_models=ranked_models,
            ranking_scores=ranking_scores,
            total_candidates=len(models),
            ranking_time_ms=ranking_time_ms
        )
    
    def rank_models_with_constraints(
        self,
        models: List[ProviderModel],
        category: str,
        constraints: RoutingConstraints,
        weights: Optional[ScoringWeights] = None,
        estimated_tokens: int = 1000
    ) -> RankingResult:
        """
        Rank models after applying constraints.
        
        This method combines constraint validation with ranking for a
        complete model selection pipeline.
        
        Args:
            models: List of ProviderModel instances to rank
            category: Task category for scoring
            constraints: RoutingConstraints to apply
            weights: Optional scoring weights
            estimated_tokens: Estimated token count
            
        Returns:
            RankingResult with constraint-validated and ranked models
        """
        from llm_router.constraints import ConstraintValidator
        
        # Apply constraints first
        validator = ConstraintValidator()
        valid_models = validator.filter_valid_models(models, constraints)
        
        # Rank the valid models
        return self.rank_models(valid_models, category, weights, estimated_tokens=estimated_tokens)
