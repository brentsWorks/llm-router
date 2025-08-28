"""Tests for Phase 3.2: Constraint Validation.

This module tests the constraint validation system that ensures models meet
hard requirements before they can be considered for selection.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock

from llm_router.constraints import (
    ConstraintValidator, 
    RoutingConstraints, 
    ConstraintViolation,
    ConstraintType
)
from llm_router.registry import ProviderModel, PricingInfo, LimitsInfo, PerformanceInfo
from llm_router.models import Category
from llm_router.utils import format_validation_error


class TestRoutingConstraints:
    """Test RoutingConstraints data model and validation."""

    def test_should_create_valid_constraints_with_defaults(self):
        """Test that RoutingConstraints can be created with sensible defaults."""
        constraints = RoutingConstraints()
        
        # Should have sensible defaults
        assert constraints.max_context_length is None  # No limit by default
        assert constraints.min_safety_level is None   # No safety requirement by default
        assert constraints.excluded_providers == []   # No exclusions by default
        assert constraints.excluded_models == []      # No model exclusions by default
        assert constraints.max_cost_per_1k_tokens is None  # No cost limit by default
        assert constraints.max_latency_ms is None     # No latency limit by default

    def test_should_create_constraints_with_specific_values(self):
        """Test that RoutingConstraints can be created with specific values."""
        constraints = RoutingConstraints(
            max_context_length=4096,
            min_safety_level="high",
            excluded_providers=["provider_a"],
            excluded_models=["model_x"],
            max_cost_per_1k_tokens=0.01,
            max_latency_ms=1000
        )
        
        assert constraints.max_context_length == 4096
        assert constraints.min_safety_level == "high"
        assert constraints.excluded_providers == ["provider_a"]
        assert constraints.excluded_models == ["model_x"]
        assert constraints.max_cost_per_1k_tokens == 0.01
        assert constraints.max_latency_ms == 1000

    def test_should_validate_context_length_bounds(self):
        """Test that context length must be positive if specified."""
        # Valid context length
        RoutingConstraints(max_context_length=1024)
        
        # Invalid context length - should raise validation error
        with pytest.raises(Exception) as exc_info:
            RoutingConstraints(max_context_length=0)
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "max_context_length" in error_message
        assert "greater than" in error_message
        
        with pytest.raises(Exception) as exc_info:
            RoutingConstraints(max_context_length=-100)
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "max_context_length" in error_message
        assert "greater than" in error_message

    def test_should_validate_safety_level_values(self):
        """Test that safety level must be from allowed values if specified."""
        valid_levels = ["low", "moderate", "high", "strict"]
        
        # Valid safety levels should work
        for level in valid_levels:
            constraints = RoutingConstraints(min_safety_level=level)
            assert constraints.min_safety_level == level
        
        # Invalid safety level should raise error
        with pytest.raises(Exception) as exc_info:
            RoutingConstraints(min_safety_level="invalid_level")
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "min_safety_level" in error_message

    def test_should_validate_cost_constraints(self):
        """Test that cost constraints must be positive if specified."""
        # Valid cost constraint
        RoutingConstraints(max_cost_per_1k_tokens=0.05)
        
        # Invalid cost constraint - should raise validation error
        with pytest.raises(Exception) as exc_info:
            RoutingConstraints(max_cost_per_1k_tokens=0)
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "max_cost_per_1k_tokens" in error_message
        assert "greater than" in error_message
        
        with pytest.raises(Exception) as exc_info:
            RoutingConstraints(max_cost_per_1k_tokens=-0.01)
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "max_cost_per_1k_tokens" in error_message
        assert "greater than" in error_message

    def test_should_validate_latency_constraints(self):
        """Test that latency constraints must be positive if specified."""
        # Valid latency constraint
        RoutingConstraints(max_latency_ms=500)
        
        # Invalid latency constraint - should raise validation error
        with pytest.raises(Exception) as exc_info:
            RoutingConstraints(max_latency_ms=0)
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "max_latency_ms" in error_message
        assert "greater than" in error_message
        
        with pytest.raises(Exception) as exc_info:
            RoutingConstraints(max_latency_ms=-100)
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "max_latency_ms" in error_message
        assert "greater than" in error_message


class TestConstraintViolation:
    """Test ConstraintViolation data model."""

    def test_should_create_constraint_violation(self):
        """Test that ConstraintViolation can be created with violation details."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.CONTEXT_LENGTH,
            field="max_context_length",
            value=8192,
            limit=4096,
            message="Model context length (8192) exceeds limit (4096)"
        )
        
        assert violation.constraint_type == ConstraintType.CONTEXT_LENGTH
        assert violation.field == "max_context_length"
        assert violation.value == 8192
        assert violation.limit == 4096
        assert violation.message == "Model context length (8192) exceeds limit (4096)"

    def test_should_serialize_violation_to_dict(self):
        """Test that ConstraintViolation can be serialized."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.SAFETY_LEVEL,
            field="min_safety_level",
            value="low",
            limit="high",
            message="Model safety level (low) below required (high)"
        )
        
        result = violation.model_dump()
        
        assert result["constraint_type"] == "safety_level"
        assert result["field"] == "min_safety_level"
        assert result["value"] == "low"
        assert result["limit"] == "high"
        assert "below required" in result["message"]


class TestConstraintValidator:
    """Test ConstraintValidator service."""

    def test_should_validate_model_with_no_constraints(self):
        """Test that model passes validation when no constraints are specified."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints()  # No constraints
        
        model = self._create_test_model()
        
        # Should pass validation with no constraints
        violations = validator.validate_model(model, constraints)
        assert len(violations) == 0

    def test_should_validate_context_length_constraint(self):
        """Test that context length constraints are properly validated."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(max_context_length=2048)
        
        # Model with acceptable context length
        model_acceptable = self._create_test_model(context_length=1024)
        violations = validator.validate_model(model_acceptable, constraints)
        assert len(violations) == 0
        
        # Model with context length at limit
        model_at_limit = self._create_test_model(context_length=2048)
        violations = validator.validate_model(model_at_limit, constraints)
        assert len(violations) == 0
        
        # Model with context length exceeding limit
        model_too_long = self._create_test_model(context_length=4096)
        violations = validator.validate_model(model_too_long, constraints)
        assert len(violations) == 1
        assert violations[0].constraint_type == ConstraintType.CONTEXT_LENGTH
        assert violations[0].field == "max_context_length"
        assert violations[0].value == 4096
        assert violations[0].limit == 2048

    def test_should_validate_safety_level_constraint(self):
        """Test that safety level constraints are properly validated."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(min_safety_level="high")
        
        # Model with acceptable safety level
        model_acceptable = self._create_test_model(safety_level="high")
        violations = validator.validate_model(model_acceptable, constraints)
        assert len(violations) == 0
        
        # Model with safety level at limit
        model_at_limit = self._create_test_model(safety_level="strict")
        violations = validator.validate_model(model_at_limit, constraints)
        assert len(violations) == 0
        
        # Model with safety level below limit
        model_too_low = self._create_test_model(safety_level="moderate")
        violations = validator.validate_model(model_too_low, constraints)
        assert len(violations) == 1
        assert violations[0].constraint_type == ConstraintType.SAFETY_LEVEL
        assert violations[0].field == "min_safety_level"
        assert violations[0].value == "moderate"
        assert violations[0].limit == "high"

    def test_should_validate_provider_exclusion_constraint(self):
        """Test that provider exclusion constraints are properly validated."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(excluded_providers=["blocked_provider"])
        
        # Model from allowed provider
        model_allowed = self._create_test_model(provider="allowed_provider")
        violations = validator.validate_model(model_allowed, constraints)
        assert len(violations) == 0
        
        # Model from blocked provider
        model_blocked = self._create_test_model(provider="blocked_provider")
        violations = validator.validate_model(model_blocked, constraints)
        assert len(violations) == 1
        assert violations[0].constraint_type == ConstraintType.PROVIDER_EXCLUSION
        assert violations[0].field == "excluded_providers"
        assert violations[0].value == "blocked_provider"

    def test_should_validate_model_exclusion_constraint(self):
        """Test that model exclusion constraints are properly validated."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(excluded_models=["blocked_model"])
        
        # Allowed model
        model_allowed = self._create_test_model(model="allowed_model")
        violations = validator.validate_model(model_allowed, constraints)
        assert len(violations) == 0
        
        # Blocked model
        model_blocked = self._create_test_model(model="blocked_model")
        violations = validator.validate_model(model_blocked, constraints)
        assert len(violations) == 1
        assert violations[0].constraint_type == ConstraintType.MODEL_EXCLUSION
        assert violations[0].field == "excluded_models"
        assert violations[0].value == "blocked_model"

    def test_should_validate_cost_constraint(self):
        """Test that cost constraints are properly validated."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(max_cost_per_1k_tokens=0.01)
        
        # Model with acceptable cost
        model_acceptable = self._create_test_model(
            input_cost=0.005, output_cost=0.005
        )
        violations = validator.validate_model(model_acceptable, constraints)
        assert len(violations) == 0
        
        # Model with cost at limit
        model_at_limit = self._create_test_model(
            input_cost=0.01, output_cost=0.0
        )
        violations = validator.validate_model(model_at_limit, constraints)
        assert len(violations) == 0
        
        # Model with cost exceeding limit
        model_too_expensive = self._create_test_model(
            input_cost=0.015, output_cost=0.005
        )
        violations = validator.validate_model(model_too_expensive, constraints)
        assert len(violations) == 1
        assert violations[0].constraint_type == ConstraintType.COST
        assert violations[0].field == "max_cost_per_1k_tokens"
        assert violations[0].value == 0.02  # Total cost
        assert violations[0].limit == 0.01

    def test_should_validate_latency_constraint(self):
        """Test that latency constraints are properly validated."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(max_latency_ms=500)
        
        # Model with acceptable latency
        model_acceptable = self._create_test_model(latency=300)
        violations = validator.validate_model(model_acceptable, constraints)
        assert len(violations) == 0
        
        # Model with latency at limit
        model_at_limit = self._create_test_model(latency=500)
        violations = validator.validate_model(model_at_limit, constraints)
        assert len(violations) == 0
        
        # Model with latency exceeding limit
        model_too_slow = self._create_test_model(latency=800)
        violations = validator.validate_model(model_too_slow, constraints)
        assert len(violations) == 1
        assert violations[0].constraint_type == ConstraintType.LATENCY
        assert violations[0].field == "max_latency_ms"
        assert violations[0].value == 800
        assert violations[0].limit == 500

    def test_should_validate_multiple_constraints(self):
        """Test that multiple constraints are validated together."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(
            max_context_length=2048,
            min_safety_level="high",
            max_cost_per_1k_tokens=0.01
        )
        
        # Model violating multiple constraints
        model_violations = self._create_test_model(
            context_length=4096,
            safety_level="moderate",
            input_cost=0.015,
            output_cost=0.005
        )
        
        violations = validator.validate_model(model_violations, constraints)
        assert len(violations) == 3
        
        # Check all violation types are present
        violation_types = {v.constraint_type for v in violations}
        expected_types = {
            ConstraintType.CONTEXT_LENGTH,
            ConstraintType.SAFETY_LEVEL,
            ConstraintType.COST
        }
        assert violation_types == expected_types

    def test_should_filter_models_by_constraints(self):
        """Test that models can be filtered based on constraints."""
        from llm_router.constraints import ConstraintValidator
        
        validator = ConstraintValidator()
        constraints = RoutingConstraints(
            max_context_length=2048,
            min_safety_level="high"
        )
        
        # Create multiple models with different characteristics
        models = [
            self._create_test_model(context_length=1024, safety_level="high"),      # Passes
            self._create_test_model(context_length=4096, safety_level="high"),      # Fails context
            self._create_test_model(context_length=1024, safety_level="moderate"),  # Fails safety
            self._create_test_model(context_length=4096, safety_level="moderate"),  # Fails both
        ]
        
        # Filter models by constraints
        valid_models = validator.filter_valid_models(models, constraints)
        
        # Should only return the model that passes all constraints
        assert len(valid_models) == 1
        assert valid_models[0].limits.context_length == 1024
        assert valid_models[0].limits.safety_level == "high"

    def _create_test_model(
        self,
        provider: str = "test_provider",
        model: str = "test_model",
        context_length: int = 2048,
        safety_level: str = "moderate",
        input_cost: float = 0.005,
        output_cost: float = 0.005,
        latency: int = 300
    ) -> ProviderModel:
        """Helper to create test ProviderModel instances."""
        return ProviderModel(
            provider=provider,
            model=model,
            capabilities=["code", "qa"],
            pricing=PricingInfo(
                input_tokens_per_1k=input_cost,
                output_tokens_per_1k=output_cost
            ),
            limits=LimitsInfo(
                context_length=context_length,
                safety_level=safety_level
            ),
            performance=PerformanceInfo(
                avg_latency_ms=latency,
                quality_scores={"code": 0.8, "qa": 0.9}
            )
        )
