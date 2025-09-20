"""Constraint validation system for Phase 3.2.

This module provides hard constraint validation to ensure models meet
requirements before they can be considered for selection.
"""

from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator

from .registry import ProviderModel


class ConstraintType(str, Enum):
    """Types of constraints that can be violated."""
    
    CONTEXT_LENGTH = "context_length"
    SAFETY_LEVEL = "safety_level"
    PROVIDER_EXCLUSION = "provider_exclusion"
    MODEL_EXCLUSION = "model_exclusion"
    COST = "cost"
    LATENCY = "latency"


class RoutingConstraints(BaseModel):
    """Configuration for routing constraints."""
    
    max_context_length: Optional[int] = Field(
        default=None, 
        gt=0, 
        description="Maximum allowed context length in tokens"
    )
    min_safety_level: Optional[str] = Field(
        default=None,
        description="Minimum required safety level"
    )
    excluded_providers: List[str] = Field(
        default_factory=list,
        description="Providers to exclude from selection"
    )
    excluded_models: List[str] = Field(
        default_factory=list,
        description="Specific models to exclude from selection"
    )
    max_cost_per_1k_tokens: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Maximum allowed cost per 1K tokens"
    )
    max_latency_ms: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum allowed latency in milliseconds"
    )
    
    @field_validator("min_safety_level")
    @classmethod
    def validate_safety_level(cls, v: Optional[str]) -> Optional[str]:
        """Validate safety level is from allowed values."""
        if v is not None:
            valid_levels = ["low", "moderate", "high", "strict"]
            if v not in valid_levels:
                raise ValueError(f"Invalid safety level: {v}. Valid levels: {valid_levels}")
        return v


class ConstraintViolation(BaseModel):
    """Details about a constraint violation."""
    
    constraint_type: ConstraintType = Field(description="Type of constraint violated")
    field: str = Field(description="Field that caused the violation")
    value: Any = Field(description="Actual value that violated the constraint")
    limit: Any = Field(description="Limit that was exceeded")
    message: str = Field(description="Human-readable violation message")


class ConstraintValidator:
    """Service to validate models against routing constraints."""
    
    def validate_model(self, model: ProviderModel, constraints: RoutingConstraints) -> List[ConstraintViolation]:
        """
        Validate a single model against constraints.
        
        Args:
            model: ProviderModel to validate
            constraints: RoutingConstraints to validate against
            
        Returns:
            List of ConstraintViolation instances (empty if no violations)
        """
        violations = []
        
        # Check context length constraint
        if constraints.max_context_length is not None:
            if model.limits.context_length > constraints.max_context_length:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CONTEXT_LENGTH,
                    field="max_context_length",
                    value=model.limits.context_length,
                    limit=constraints.max_context_length,
                    message=f"Model context length ({model.limits.context_length}) exceeds limit ({constraints.max_context_length})"
                ))
        
        # Check safety level constraint
        if constraints.min_safety_level is not None:
            if not self._meets_safety_level(model.limits.safety_level, constraints.min_safety_level):
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.SAFETY_LEVEL,
                    field="min_safety_level",
                    value=model.limits.safety_level,
                    limit=constraints.min_safety_level,
                    message=f"Model safety level ({model.limits.safety_level}) below required ({constraints.min_safety_level})"
                ))
        
        # Check provider exclusion constraint
        if constraints.excluded_providers and model.provider in constraints.excluded_providers:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.PROVIDER_EXCLUSION,
                field="excluded_providers",
                value=model.provider,
                limit=constraints.excluded_providers,
                message=f"Provider '{model.provider}' is excluded"
            ))
        
        # Check model exclusion constraint
        if constraints.excluded_models and model.model in constraints.excluded_models:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.MODEL_EXCLUSION,
                field="excluded_models",
                value=model.model,
                limit=constraints.excluded_models,
                message=f"Model '{model.model}' is excluded"
            ))
        
        # Check cost constraint
        if constraints.max_cost_per_1k_tokens is not None:
            total_cost = model.pricing.input_tokens_per_1k + model.pricing.output_tokens_per_1k
            if total_cost > constraints.max_cost_per_1k_tokens:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.COST,
                    field="max_cost_per_1k_tokens",
                    value=total_cost,
                    limit=constraints.max_cost_per_1k_tokens,
                    message=f"Model cost ({total_cost:.4f}) exceeds limit ({constraints.max_cost_per_1k_tokens:.4f})"
                ))
        
        # Check latency constraint
        if constraints.max_latency_ms is not None:
            if model.performance.avg_latency_ms > constraints.max_latency_ms:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.LATENCY,
                    field="max_latency_ms",
                    value=model.performance.avg_latency_ms,
                    limit=constraints.max_latency_ms,
                    message=f"Model latency ({model.performance.avg_latency_ms}ms) exceeds limit ({constraints.max_latency_ms}ms)"
                ))
        
        return violations
    
    def filter_valid_models(self, models: List[ProviderModel], constraints: RoutingConstraints) -> List[ProviderModel]:
        """
        Filter models to only those that meet all constraints.
        
        Args:
            models: List of ProviderModel instances to filter
            constraints: RoutingConstraints to filter against
            
        Returns:
            List of models that pass all constraints
        """
        valid_models = []
        
        for model in models:
            violations = self.validate_model(model, constraints)
            if not violations:
                valid_models.append(model)
        
        return valid_models
    
    def _meets_safety_level(self, model_level: str, required_level: str) -> bool:
        """Check if model safety level meets or exceeds required level."""
        safety_hierarchy = {
            "low": 1,
            "moderate": 2,
            "high": 3,
            "strict": 4
        }
        
        model_score = safety_hierarchy.get(model_level, 0)
        required_score = safety_hierarchy.get(required_level, 0)
        
        return model_score >= required_score
