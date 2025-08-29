"""Core data models for LLM routing."""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# Import TaskType from capabilities to maintain consistency
from llm_router.capabilities import TaskType


# Use TaskType as the single source of truth for categories
Category = TaskType

# Export valid categories for tests (using TaskType values)
VALID_CATEGORIES = [task_type.value for task_type in TaskType]


class PromptClassification(BaseModel):
    """Classification result for a user prompt."""

    model_config = ConfigDict(use_enum_values=True)

    category: Category
    subcategory: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: List[float]
    reasoning: Optional[str] = None


class ModelCandidate(BaseModel):
    """A candidate model for routing with scoring information."""

    provider: str
    model: str
    score: float = Field(ge=0.0)
    estimated_cost: float = Field(ge=0.0)
    estimated_latency: float = Field(ge=0.0)
    quality_match: float
    constraint_violations: List[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    """Complete routing decision with selected model and alternatives."""

    model_config = ConfigDict(use_enum_values=True)

    selected_model: ModelCandidate
    classification: PromptClassification
    alternatives: List[ModelCandidate] = Field(default_factory=list)
    routing_time_ms: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
