"""Tests for Phase 2.2: Model Capability Definitions.

This module contains comprehensive tests for the ModelCapabilities system,
capability matching algorithms, and integration with the ProviderModel.

Following TDD approach - these tests will initially FAIL and drive implementation.
"""

import pytest
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from llm_router.models import Category
from llm_router.capabilities import (
    TaskType, SafetyLevel, ModelCapabilities, CapabilityRequirement,
    CapabilityMatcher, CapabilityScore
)
from llm_router.utils import format_validation_error
from llm_router.registry import ProviderModel, PricingInfo, LimitsInfo, PerformanceInfo


class TestTaskType:
    """Test the TaskType enum for different LLM task categories."""
    
    def test_task_type_values(self):
        """Test that TaskType enum has expected values."""
        assert TaskType.CODE.value == "code"
        assert TaskType.CREATIVE.value == "creative" 
        assert TaskType.QA.value == "qa"
        assert TaskType.SUMMARIZATION.value == "summarization"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.TOOL_USE.value == "tool_use"
        assert TaskType.TRANSLATION.value == "translation"
        assert TaskType.ANALYSIS.value == "analysis"
    
    def test_task_type_compatibility_with_category(self):
        """Test that TaskType values align with existing Category values."""
        # Ensure backward compatibility
        assert TaskType.CODE.value == Category.CODE.value
        assert TaskType.CREATIVE.value == Category.CREATIVE.value
        assert TaskType.QA.value == Category.QA.value
        assert TaskType.SUMMARIZATION.value == Category.SUMMARIZATION.value


class TestSafetyLevel:
    """Test the SafetyLevel enum for model safety classifications."""
    
    def test_safety_level_values(self):
        """Test that SafetyLevel enum has expected values."""
        assert SafetyLevel.LOW.value == "low"
        assert SafetyLevel.MEDIUM.value == "medium"
        assert SafetyLevel.HIGH.value == "high"
        assert SafetyLevel.STRICT.value == "strict"
    
    def test_safety_level_ordering(self):
        """Test that safety levels can be compared for ordering."""
        assert SafetyLevel.LOW < SafetyLevel.MEDIUM
        assert SafetyLevel.MEDIUM < SafetyLevel.HIGH
        assert SafetyLevel.HIGH < SafetyLevel.STRICT


class TestModelCapabilities:
    """Test the ModelCapabilities data model."""
    
    def test_model_capabilities_creation(self):
        """Test creating ModelCapabilities with valid data."""
        capabilities = ModelCapabilities(
            task_expertise={
                TaskType.CODE: 0.9,
                TaskType.QA: 0.8
            },
            context_length_optimal=4000,
            context_length_max=8000,
            safety_level=SafetyLevel.MEDIUM,
            supports_streaming=True,
            supports_function_calling=True,
            languages_supported=["python", "javascript", "typescript"],
            specialized_domains=["web_development", "data_science"]
        )
        
        assert capabilities.task_expertise[TaskType.CODE] == 0.9
        assert capabilities.context_length_optimal == 4000
        assert capabilities.context_length_max == 8000
        assert capabilities.safety_level == SafetyLevel.MEDIUM
        assert capabilities.supports_streaming is True
        assert "python" in capabilities.languages_supported
    
    def test_model_capabilities_validation_task_expertise_scores(self):
        """Test that task expertise scores must be between 0 and 1."""
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(
                task_expertise={TaskType.CODE: 1.5},  # Invalid score > 1
                context_length_max=4000
            )
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "task_expertise" in error_message or "Task expertise score" in error_message
        
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(
                task_expertise={TaskType.CODE: -0.1},  # Invalid score < 0
                context_length_max=4000
            )
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "task_expertise" in error_message or "Task expertise score" in error_message
    
    def test_model_capabilities_validation_context_lengths(self):
        """Test context length validation rules."""
        # context_length_max is required
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(task_expertise={TaskType.CODE: 0.8})
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "context_length_max" in error_message
        
        # context_length_optimal cannot exceed context_length_max
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(
                task_expertise={TaskType.CODE: 0.8},
                context_length_optimal=8000,
                context_length_max=4000  # Max < Optimal
            )
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "optimal cannot exceed max" in error_message.lower()
    
    def test_model_capabilities_defaults(self):
        """Test default values for optional fields."""
        capabilities = ModelCapabilities(
            task_expertise={TaskType.CODE: 0.8},
            context_length_max=4000
        )
        
        assert capabilities.context_length_optimal is None
        assert capabilities.safety_level == SafetyLevel.MEDIUM  # Default
        assert capabilities.supports_streaming is False  # Default
        assert capabilities.supports_function_calling is False  # Default
        assert capabilities.languages_supported == []  # Default
        assert capabilities.specialized_domains == []  # Default
    
    def test_get_task_score_method(self):
        """Test getting task expertise scores."""
        capabilities = ModelCapabilities(
            task_expertise={
                TaskType.CODE: 0.9,
                TaskType.QA: 0.7
            },
            context_length_max=4000
        )
        
        assert capabilities.get_task_score(TaskType.CODE) == 0.9
        assert capabilities.get_task_score(TaskType.QA) == 0.7
        assert capabilities.get_task_score(TaskType.CREATIVE) == 0.0  # Default for missing
    
    def test_has_capability_method(self):
        """Test checking if model has specific capabilities."""
        capabilities = ModelCapabilities(
            task_expertise={TaskType.CODE: 0.8},
            context_length_max=4000,
            supports_streaming=True,
            languages_supported=["python"]
        )
        
        assert capabilities.has_capability(TaskType.CODE) is True
        assert capabilities.has_capability(TaskType.CREATIVE) is False
        assert capabilities.supports_feature("streaming") is True
        assert capabilities.supports_feature("function_calling") is False
        assert capabilities.supports_language("python") is True
        assert capabilities.supports_language("java") is False


class TestCapabilityRequirement:
    """Test the CapabilityRequirement model for specifying routing needs."""
    
    def test_capability_requirement_creation(self):
        """Test creating capability requirements."""
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            min_score=0.8,
            context_length_needed=2000,
            required_safety_level=SafetyLevel.HIGH,
            requires_streaming=True,
            preferred_languages=["python", "typescript"]
        )
        
        assert req.primary_task == TaskType.CODE
        assert req.min_score == 0.8
        assert req.context_length_needed == 2000
        assert req.required_safety_level == SafetyLevel.HIGH
    
    def test_capability_requirement_validation(self):
        """Test validation of capability requirements."""
        # min_score must be between 0 and 1
        with pytest.raises(ValidationError) as exc_info:
            CapabilityRequirement(
                primary_task=TaskType.CODE,
                min_score=1.5
            )
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "min_score" in error_message
        
        # context_length_needed must be positive
        with pytest.raises(ValidationError) as exc_info:
            CapabilityRequirement(
                primary_task=TaskType.CODE,
                context_length_needed=-100
            )
        # Should get clear error message
        error_message = format_validation_error(exc_info.value)
        assert "context_length_needed" in error_message
    
    def test_capability_requirement_defaults(self):
        """Test default values for capability requirements."""
        req = CapabilityRequirement(primary_task=TaskType.CODE)
        
        assert req.min_score == 0.0  # Default
        assert req.context_length_needed is None  # Optional
        assert req.required_safety_level is None  # Optional
        assert req.requires_streaming is False  # Default
        assert req.preferred_languages == []  # Default


class TestCapabilityScore:
    """Test the CapabilityScore model for matching results."""
    
    def test_capability_score_creation(self):
        """Test creating capability match scores."""
        score = CapabilityScore(
            overall_score=0.85,
            task_match_score=0.9,
            context_length_score=1.0,
            safety_score=0.8,
            feature_score=0.7,
            language_score=0.9,
            penalties=0.05
        )
        
        assert score.overall_score == 0.85
        assert score.task_match_score == 0.9
        assert score.penalties == 0.05
    
    def test_capability_score_validation(self):
        """Test that capability scores are properly validated."""
        # All scores must be between 0 and 1
        with pytest.raises(ValidationError):
            CapabilityScore(
                overall_score=1.2,  # Invalid
                task_match_score=0.9
            )
    
    def test_capability_score_calculation_method(self):
        """Test the score calculation method."""
        score = CapabilityScore(
            overall_score=0.0,  # Will be calculated
            task_match_score=0.9,
            context_length_score=1.0,
            safety_score=0.8,
            feature_score=0.7,
            language_score=0.9,
            penalties=0.1
        )
        
        calculated_score = score.calculate_overall_score()
        # Expected: weighted average minus penalties
        # Implementation should define the exact formula
        assert 0.0 <= calculated_score <= 1.0
        assert calculated_score < 0.9  # Should be less due to penalties


class TestCapabilityMatcher:
    """Test the CapabilityMatcher for scoring model-requirement matches."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = CapabilityMatcher()
        
        self.capabilities = ModelCapabilities(
            task_expertise={
                TaskType.CODE: 0.9,
                TaskType.QA: 0.7,
                TaskType.CREATIVE: 0.3
            },
            context_length_optimal=4000,
            context_length_max=8000,
            safety_level=SafetyLevel.HIGH,
            supports_streaming=True,
            supports_function_calling=True,
            languages_supported=["python", "javascript", "typescript"],
            specialized_domains=["web_development"]
        )
    
    def test_capability_matcher_initialization(self):
        """Test CapabilityMatcher initialization."""
        matcher = CapabilityMatcher()
        assert matcher is not None
        
        # Test with custom weights
        custom_weights = {
            "task_weight": 0.4,
            "context_weight": 0.2,
            "safety_weight": 0.2,
            "feature_weight": 0.1,
            "language_weight": 0.1
        }
        matcher = CapabilityMatcher(weights=custom_weights)
        assert matcher.weights["task_weight"] == 0.4
    
    def test_score_task_match(self):
        """Test scoring task expertise match."""
        # Perfect match
        req = CapabilityRequirement(primary_task=TaskType.CODE, min_score=0.8)
        score = self.matcher.score_task_match(self.capabilities, req)
        assert score == 0.9  # Model has 0.9 expertise in CODE
        
        # Below minimum threshold
        req = CapabilityRequirement(primary_task=TaskType.CODE, min_score=0.95)
        score = self.matcher.score_task_match(self.capabilities, req)
        assert score < 0.5  # Should be penalized for not meeting minimum
        
        # Task model doesn't support
        req = CapabilityRequirement(primary_task=TaskType.TRANSLATION)
        score = self.matcher.score_task_match(self.capabilities, req)
        assert score == 0.0  # No expertise in translation
    
    def test_score_context_length_match(self):
        """Test scoring context length compatibility."""
        # Within optimal range
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            context_length_needed=3000
        )
        score = self.matcher.score_context_length_match(self.capabilities, req)
        assert score == 1.0  # Perfect score for within optimal
        
        # Beyond optimal but within max
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            context_length_needed=6000
        )
        score = self.matcher.score_context_length_match(self.capabilities, req)
        assert 0.5 <= score < 1.0  # Reduced but acceptable
        
        # Beyond maximum capacity
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            context_length_needed=10000
        )
        score = self.matcher.score_context_length_match(self.capabilities, req)
        assert score == 0.0  # Cannot handle
    
    def test_score_safety_match(self):
        """Test scoring safety level compatibility."""
        # Exact match
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            required_safety_level=SafetyLevel.HIGH
        )
        score = self.matcher.score_safety_match(self.capabilities, req)
        assert score == 1.0
        
        # Model exceeds requirement (good)
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            required_safety_level=SafetyLevel.MEDIUM
        )
        score = self.matcher.score_safety_match(self.capabilities, req)
        assert score == 1.0  # High safety meets medium requirement
        
        # Model below requirement (bad)
        low_safety_capabilities = ModelCapabilities(
            task_expertise={TaskType.CODE: 0.9},
            context_length_max=4000,
            safety_level=SafetyLevel.LOW
        )
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            required_safety_level=SafetyLevel.HIGH
        )
        score = self.matcher.score_safety_match(low_safety_capabilities, req)
        assert score == 0.0  # Unacceptable
    
    def test_score_feature_match(self):
        """Test scoring feature compatibility."""
        # Requires streaming - model supports it
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            requires_streaming=True
        )
        score = self.matcher.score_feature_match(self.capabilities, req)
        assert score >= 0.8  # High score for supporting required feature
        
        # Requires function calling - model supports it
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            requires_function_calling=True
        )
        score = self.matcher.score_feature_match(self.capabilities, req)
        assert score >= 0.8
        
        # No special requirements
        req = CapabilityRequirement(primary_task=TaskType.CODE)
        score = self.matcher.score_feature_match(self.capabilities, req)
        assert score == 1.0  # Perfect when no requirements
    
    def test_score_language_match(self):
        """Test scoring programming language support."""
        # Supported language
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            preferred_languages=["python"]
        )
        score = self.matcher.score_language_match(self.capabilities, req)
        assert score == 1.0  # Perfect match
        
        # Partially supported languages
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            preferred_languages=["python", "java", "rust"]  # Only python supported
        )
        score = self.matcher.score_language_match(self.capabilities, req)
        assert 0.3 <= score < 1.0  # Partial match
        
        # No supported languages
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            preferred_languages=["cobol", "fortran"]
        )
        score = self.matcher.score_language_match(self.capabilities, req)
        assert score == 0.0  # No match
    
    def test_calculate_overall_score(self):
        """Test overall capability score calculation."""
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            min_score=0.8,
            context_length_needed=3000,
            required_safety_level=SafetyLevel.HIGH,
            requires_streaming=True,
            preferred_languages=["python"]
        )
        
        score = self.matcher.calculate_score(self.capabilities, req)
        
        assert isinstance(score, CapabilityScore)
        assert 0.8 <= score.overall_score <= 1.0  # Should be high for good match
        assert score.task_match_score == 0.9  # CODE expertise
        assert score.context_length_score == 1.0  # Within optimal
        assert score.safety_score == 1.0  # Exact match
        assert score.feature_score >= 0.8  # Supports streaming
        assert score.language_score == 1.0  # Supports Python
    
    def test_calculate_score_with_penalties(self):
        """Test score calculation with various penalties."""
        # Below minimum task score
        req = CapabilityRequirement(
            primary_task=TaskType.CREATIVE,  # Model only has 0.3 expertise
            min_score=0.8
        )
        
        score = self.matcher.calculate_score(self.capabilities, req)
        assert score.overall_score < 0.5  # Heavy penalty for below minimum
        assert score.penalties > 0.0
    
    def test_find_best_matches(self):
        """Test finding best capability matches from multiple models."""
        # Create multiple model capabilities for comparison
        models_capabilities = [
            ("openai/gpt-4", self.capabilities),
            ("openai/gpt-3.5", ModelCapabilities(
                task_expertise={TaskType.CODE: 0.7, TaskType.QA: 0.8},
                context_length_max=4000,
                safety_level=SafetyLevel.MEDIUM
            )),
            ("anthropic/claude", ModelCapabilities(
                task_expertise={TaskType.CREATIVE: 0.9, TaskType.QA: 0.8},
                context_length_max=100000,
                safety_level=SafetyLevel.HIGH
            ))
        ]
        
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            min_score=0.8
        )
        
        matches = self.matcher.find_best_matches(models_capabilities, req, top_k=2)
        
        assert len(matches) <= 2
        assert len(matches) > 0
        
        # Results should be sorted by overall score (descending)
        for i in range(len(matches) - 1):
            assert matches[i][1].overall_score >= matches[i + 1][1].overall_score
        
        # Best match should be the one with high CODE expertise
        best_model, best_score = matches[0]
        assert "gpt-4" in best_model  # Should prefer the model with 0.9 CODE expertise


class TestProviderModelIntegration:
    """Test integration of ModelCapabilities with ProviderModel."""
    
    def test_provider_model_with_capabilities(self):
        """Test ProviderModel with new ModelCapabilities field."""
        capabilities = ModelCapabilities(
            task_expertise={TaskType.CODE: 0.9},
            context_length_max=8000,
            safety_level=SafetyLevel.HIGH
        )
        
        model = ProviderModel(
            provider="openai",
            model="gpt-4",
            capabilities=["code", "qa"],  # Legacy format still supported
            detailed_capabilities=capabilities,  # New field
            pricing=PricingInfo(input_tokens_per_1k=0.03, output_tokens_per_1k=0.06),
            limits=LimitsInfo(context_length=8000),
            performance=PerformanceInfo(avg_latency_ms=2000.0)
        )
        
        assert model.detailed_capabilities is not None
        assert model.detailed_capabilities.get_task_score(TaskType.CODE) == 0.9
        assert model.has_capability("code")  # Legacy method still works
    
    def test_provider_model_backward_compatibility(self):
        """Test that existing ProviderModel usage still works."""
        # Should work without detailed_capabilities field
        model = ProviderModel(
            provider="openai",
            model="gpt-3.5-turbo",
            capabilities=["code", "qa"],
            pricing=PricingInfo(input_tokens_per_1k=0.002, output_tokens_per_1k=0.002),
            limits=LimitsInfo(context_length=4000),
            performance=PerformanceInfo(avg_latency_ms=1500.0)
        )
        
        assert model.detailed_capabilities is None  # Optional field
        assert model.has_capability("code")  # Legacy functionality
    
    def test_provider_model_capability_sync_validation(self):
        """Test validation between legacy capabilities and detailed_capabilities."""
        capabilities = ModelCapabilities(
            task_expertise={TaskType.CODE: 0.9},
            context_length_max=8000
        )
        
        # Should validate that legacy and detailed capabilities are consistent
        with pytest.raises(ValidationError) as exc_info:
            ProviderModel(
                provider="openai",
                model="gpt-4",
                capabilities=["creative", "qa"],  # Doesn't include "code"
                detailed_capabilities=capabilities,  # Has CODE expertise
                pricing=PricingInfo(input_tokens_per_1k=0.03, output_tokens_per_1k=0.06),
                limits=LimitsInfo(context_length=8000),
                performance=PerformanceInfo(avg_latency_ms=2000.0)
            )
        assert "mismatch" in str(exc_info.value).lower()
    
    def test_get_capability_score_method(self):
        """Test new capability scoring method on ProviderModel."""
        capabilities = ModelCapabilities(
            task_expertise={TaskType.CODE: 0.9, TaskType.QA: 0.7},
            context_length_max=8000,
            safety_level=SafetyLevel.HIGH
        )
        
        model = ProviderModel(
            provider="openai",
            model="gpt-4",
            capabilities=["code", "qa"],
            detailed_capabilities=capabilities,
            pricing=PricingInfo(input_tokens_per_1k=0.03, output_tokens_per_1k=0.06),
            limits=LimitsInfo(context_length=8000),
            performance=PerformanceInfo(avg_latency_ms=2000.0)
        )
        
        req = CapabilityRequirement(primary_task=TaskType.CODE)
        score = model.calculate_capability_score(req)
        
        assert isinstance(score, CapabilityScore)
        assert score.overall_score > 0.8  # Should be high for good match


class TestRegistryCapabilityIntegration:
    """Test ProviderRegistry integration with new capability system."""
    
    def test_registry_filter_by_capability_requirements(self):
        """Test filtering registry by capability requirements."""
        from llm_router.registry import ProviderRegistry
        
        registry = ProviderRegistry()
        
        # Add models with different capabilities
        gpt4_capabilities = ModelCapabilities(
            task_expertise={TaskType.CODE: 0.9, TaskType.REASONING: 0.8},
            context_length_max=8000,
            safety_level=SafetyLevel.HIGH
        )
        
        claude_capabilities = ModelCapabilities(
            task_expertise={TaskType.CREATIVE: 0.9, TaskType.QA: 0.8},
            context_length_max=100000,
            safety_level=SafetyLevel.HIGH
        )
        
        gpt4_model = ProviderModel(
            provider="openai",
            model="gpt-4",
            capabilities=["code", "reasoning"],
            detailed_capabilities=gpt4_capabilities,
            pricing=PricingInfo(input_tokens_per_1k=0.03, output_tokens_per_1k=0.06),
            limits=LimitsInfo(context_length=8000),
            performance=PerformanceInfo(avg_latency_ms=2000.0)
        )
        
        claude_model = ProviderModel(
            provider="anthropic",
            model="claude-3",
            capabilities=["creative", "qa"],
            detailed_capabilities=claude_capabilities,
            pricing=PricingInfo(input_tokens_per_1k=0.025, output_tokens_per_1k=0.125),
            limits=LimitsInfo(context_length=100000),
            performance=PerformanceInfo(avg_latency_ms=3000.0)
        )
        
        registry.add_model(gpt4_model)
        registry.add_model(claude_model)
        
        # Filter by capability requirements
        req = CapabilityRequirement(
            primary_task=TaskType.CODE,
            min_score=0.8
        )
        
        matches = registry.find_models_by_requirements(req, min_score=0.7)
        
        assert len(matches) == 1
        assert matches[0][0].model == "gpt-4"  # Should find GPT-4 for code tasks
        assert matches[0][1].overall_score > 0.7
    
    def test_registry_ranking_by_capability_score(self):
        """Test ranking models by capability scores."""
        from llm_router.registry import ProviderRegistry
        
        registry = ProviderRegistry()
        # Implementation should add models and test ranking
        # This tests the registry's ability to score and rank models
        assert True  # Placeholder - will be implemented
