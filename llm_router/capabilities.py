"""Model Capability Definitions for Phase 2.2.

This module provides comprehensive capability modeling for LLM routing decisions,
including task expertise, safety levels, and matching algorithms.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(Enum):
    """Enumeration of LLM task types for capability matching."""

    CODE = "code"
    CREATIVE = "creative"
    QA = "qa"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    TRANSLATION = "translation"
    ANALYSIS = "analysis"


class SafetyLevel(Enum):
    """Enumeration of model safety levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRICT = "strict"

    def __lt__(self, other):
        """Enable ordering of safety levels."""
        return self._get_order() < other._get_order()

    def __ge__(self, other):
        """Enable >= comparison of safety levels."""
        return self._get_order() >= other._get_order()

    def _get_order(self) -> int:
        """Get numeric order for comparison."""
        order = {
            SafetyLevel.LOW: 0,
            SafetyLevel.MEDIUM: 1,
            SafetyLevel.HIGH: 2,
            SafetyLevel.STRICT: 3,
        }
        return order[self]


class ModelCapabilities(BaseModel):
    """Comprehensive model capability definition."""

    task_expertise: Dict[TaskType, float] = Field(
        default_factory=dict, description="Task expertise scores (0.0 to 1.0)"
    )
    context_length_optimal: Optional[int] = Field(
        None, description="Optimal context length for this model"
    )
    context_length_max: int = Field(
        gt=0, description="Maximum context length supported"
    )
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.MEDIUM, description="Model safety level"
    )
    supports_streaming: bool = Field(
        default=False, description="Whether model supports streaming"
    )
    supports_function_calling: bool = Field(
        default=False, description="Whether model supports function calling"
    )
    languages_supported: List[str] = Field(
        default_factory=list, description="Programming languages supported"
    )
    specialized_domains: List[str] = Field(
        default_factory=list, description="Specialized domains of expertise"
    )

    @field_validator("task_expertise")
    @classmethod
    def validate_task_expertise_scores(cls, v):
        """Validate task expertise scores are between 0 and 1."""
        for task_type, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Task expertise score for {task_type} must be between 0.0 and 1.0"
                )
        return v

    @model_validator(mode="after")
    def validate_context_lengths(self):
        """Validate that optimal context length doesn't exceed max."""
        if (
            self.context_length_optimal is not None
            and self.context_length_optimal > self.context_length_max
        ):
            raise ValueError("optimal cannot exceed max")
        return self

    def get_task_score(self, task_type: TaskType) -> float:
        """Get expertise score for a specific task type."""
        return self.task_expertise.get(task_type, 0.0)

    def has_capability(self, task_type: TaskType) -> bool:
        """Check if model has capability for a task type."""
        return self.get_task_score(task_type) > 0.0

    def supports_feature(self, feature: str) -> bool:
        """Check if model supports a specific feature."""
        if feature == "streaming":
            return self.supports_streaming
        elif feature == "function_calling":
            return self.supports_function_calling
        return False

    def supports_language(self, language: str) -> bool:
        """Check if model supports a programming language."""
        return language.lower() in [lang.lower() for lang in self.languages_supported]


class CapabilityRequirement(BaseModel):
    """Specification of capability requirements for routing."""

    primary_task: TaskType = Field(description="Primary task type required")
    min_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum task expertise score"
    )
    context_length_needed: Optional[int] = Field(
        None, gt=0, description="Required context length"
    )
    required_safety_level: Optional[SafetyLevel] = Field(
        None, description="Required safety level"
    )
    requires_streaming: bool = Field(
        default=False, description="Whether streaming is required"
    )
    requires_function_calling: bool = Field(
        default=False, description="Whether function calling is required"
    )
    preferred_languages: List[str] = Field(
        default_factory=list, description="Preferred programming languages"
    )


class CapabilityScore(BaseModel):
    """Score representing capability match quality."""

    overall_score: float = Field(
        ge=0.0, le=1.0, description="Overall capability match score"
    )
    task_match_score: float = Field(
        ge=0.0, le=1.0, description="Task expertise match score"
    )
    context_length_score: float = Field(
        ge=0.0, le=1.0, description="Context length compatibility score"
    )
    safety_score: float = Field(
        ge=0.0, le=1.0, description="Safety level compatibility score"
    )
    feature_score: float = Field(
        ge=0.0, le=1.0, description="Feature compatibility score"
    )
    language_score: float = Field(ge=0.0, le=1.0, description="Language support score")
    penalties: float = Field(
        default=0.0, ge=0.0, description="Penalties applied to score"
    )

    def calculate_overall_score(self) -> float:
        """Calculate overall score from component scores minus penalties."""
        # Simple weighted average for now - can be enhanced later
        weighted_score = (
            self.task_match_score * 0.4
            + self.context_length_score * 0.2
            + self.safety_score * 0.2
            + self.feature_score * 0.1
            + self.language_score * 0.1
        )
        return max(0.0, weighted_score - self.penalties)


class CapabilityMatcher:
    """Matches model capabilities against requirements."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize matcher with optional custom weights."""
        self.weights = weights or {
            "task_weight": 0.4,
            "context_weight": 0.2,
            "safety_weight": 0.2,
            "feature_weight": 0.1,
            "language_weight": 0.1,
        }

    def score_task_match(
        self, capabilities: ModelCapabilities, requirement: CapabilityRequirement
    ) -> float:
        """Score task expertise match."""
        task_score = capabilities.get_task_score(requirement.primary_task)

        if task_score < requirement.min_score:
            # Heavy penalty for not meeting minimum
            return task_score * 0.1  # 90% penalty for not meeting minimum

        return task_score

    def score_context_length_match(
        self, capabilities: ModelCapabilities, requirement: CapabilityRequirement
    ) -> float:
        """Score context length compatibility."""
        if requirement.context_length_needed is None:
            return 1.0

        needed = requirement.context_length_needed
        max_capacity = capabilities.context_length_max

        if needed > max_capacity:
            return 0.0

        if capabilities.context_length_optimal is None:
            # No optimal specified, linear scoring
            return needed / max_capacity

        optimal = capabilities.context_length_optimal

        if needed <= optimal:
            return 1.0
        else:
            # Linear degradation from optimal to max
            degradation = (needed - optimal) / (max_capacity - optimal) * 0.5
            return 1.0 - degradation

    def score_safety_match(
        self, capabilities: ModelCapabilities, requirement: CapabilityRequirement
    ) -> float:
        """Score safety level compatibility."""
        if requirement.required_safety_level is None:
            return 1.0

        model_safety = capabilities.safety_level
        required_safety = requirement.required_safety_level

        return 1.0 if model_safety >= required_safety else 0.0

    def score_feature_match(
        self, capabilities: ModelCapabilities, requirement: CapabilityRequirement
    ) -> float:
        """Score feature compatibility."""
        score = 1.0

        if requirement.requires_streaming and not capabilities.supports_streaming:
            score -= 0.5

        if (
            requirement.requires_function_calling
            and not capabilities.supports_function_calling
        ):
            score -= 0.5

        return max(0.0, score)

    def score_language_match(
        self, capabilities: ModelCapabilities, requirement: CapabilityRequirement
    ) -> float:
        """Score programming language support."""
        if not requirement.preferred_languages:
            return 1.0

        supported_count = sum(
            1
            for lang in requirement.preferred_languages
            if capabilities.supports_language(lang)
        )

        return supported_count / len(requirement.preferred_languages)

    def calculate_score(
        self, capabilities: ModelCapabilities, requirement: CapabilityRequirement
    ) -> CapabilityScore:
        """Calculate comprehensive capability match score."""
        task_score = self.score_task_match(capabilities, requirement)
        context_score = self.score_context_length_match(capabilities, requirement)
        safety_score = self.score_safety_match(capabilities, requirement)
        feature_score = self.score_feature_match(capabilities, requirement)
        language_score = self.score_language_match(capabilities, requirement)

        # Calculate penalties
        penalties = 0.0
        if task_score < requirement.min_score:
            penalties += 0.3

        # Create score object
        score = CapabilityScore(
            overall_score=0.0,  # Will be calculated
            task_match_score=task_score,
            context_length_score=context_score,
            safety_score=safety_score,
            feature_score=feature_score,
            language_score=language_score,
            penalties=penalties,
        )

        # Calculate overall score
        score.overall_score = score.calculate_overall_score()

        return score

    def find_best_matches(
        self,
        models_capabilities: List[Tuple[str, ModelCapabilities]],
        requirement: CapabilityRequirement,
        top_k: int = 5,
    ) -> List[Tuple[str, CapabilityScore]]:
        """Find best matching models for given requirements."""
        matches = []

        for model_name, capabilities in models_capabilities:
            score = self.calculate_score(capabilities, requirement)
            matches.append((model_name, score))

        # Sort by overall score (descending) and return top_k
        matches.sort(key=lambda x: x[1].overall_score, reverse=True)
        return matches[:top_k]
