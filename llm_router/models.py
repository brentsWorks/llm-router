"""Core data models for LLM routing."""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class Category(str, Enum):
    """Valid prompt categories for classification."""
    CODE = "code"
    CREATIVE = "creative"
    QA = "qa"
    SUMMARIZATION = "summarization"
    TOOL_USE = "tool-use"


# Export valid categories for tests
VALID_CATEGORIES = [category.value for category in Category]


class PromptClassification(BaseModel):
    """Classification result for a user prompt."""
    
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
    
    selected_model: ModelCandidate
    classification: PromptClassification
    alternatives: List[ModelCandidate] = Field(default_factory=list)
    routing_time_ms: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
