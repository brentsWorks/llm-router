"""
API Request/Response Models
==========================

Pydantic models for FastAPI request and response validation.
Keeps main.py focused on application setup and routing logic.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from ..scoring import ScoringWeights
from ..constraints import RoutingConstraints


class RouteRequest(BaseModel):
    """Request model for the /route endpoint."""
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to route")
    preferences: Optional[ScoringWeights] = Field(
        default=None, 
        description="Scoring weights for cost/latency/quality optimization"
    )
    constraints: Optional[RoutingConstraints] = Field(
        default=None, 
        description="Routing constraints for filtering models"
    )


class RouteResponse(BaseModel):
    """Response model for the /route endpoint."""
    selected_model: Dict[str, Any] = Field(..., description="The selected model information")
    classification: Dict[str, Any] = Field(..., description="Prompt classification details")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    routing_time_ms: float = Field(..., ge=0.0, description="Time taken for routing decision")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the routing decision")
    # Client-side execution guidance
    provider_info: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="API endpoints and model parameters for client-side execution"
    )
    fallback_models: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Backup model options for fallback"
    )


class ExecuteResponse(BaseModel):
    """Response model for the /execute endpoint with LLM execution."""
    selected_model: Dict[str, Any] = Field(..., description="The selected model information")
    classification: Dict[str, Any] = Field(..., description="Prompt classification details")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    routing_time_ms: float = Field(..., ge=0.0, description="Time taken for routing decision")
    reasoning: Optional[str] = Field(default=None, description="Reasoning for the routing decision")
    # LLM execution results
    llm_response: str = Field(..., description="The actual LLM response content")
    model_used: str = Field(..., description="The model that was actually used for execution")
    execution_time_ms: float = Field(..., ge=0.0, description="Time taken for LLM execution")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage information")
    finish_reason: Optional[str] = Field(default=None, description="Reason the LLM finished generating")


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    total_models: int = Field(..., ge=0, description="Number of available models")
    total_requests: int = Field(..., ge=0, description="Total requests processed")


class ModelInfo(BaseModel):
    """Model information for the /models endpoint."""
    provider: str = Field(..., description="Model provider name")
    model: str = Field(..., description="Model identifier")
    capabilities: List[str] = Field(..., description="Model capabilities")
    pricing: Dict[str, float] = Field(..., description="Pricing information")
    limits: Dict[str, Any] = Field(..., description="Model limits and constraints")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")


class ModelsResponse(BaseModel):
    """Response model for the /models endpoint."""
    models: List[ModelInfo] = Field(..., description="Available models")
    total_count: int = Field(..., ge=0, description="Total number of models")


class ClassifyRequest(BaseModel):
    """Request model for the /classify endpoint."""
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to classify")


class ClassifyResponse(BaseModel):
    """Response model for the /classify endpoint."""
    category: str = Field(..., description="Classified category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: Optional[str] = Field(default=None, description="Classification reasoning")
    classification_time_ms: float = Field(..., ge=0.0, description="Time taken for classification")


class MetricsResponse(BaseModel):
    """Response model for the /metrics endpoint."""
    total_requests: int = Field(..., ge=0, description="Total requests processed")
    average_response_time_ms: float = Field(..., ge=0.0, description="Average response time")
    classification_stats: Dict[str, int] = Field(..., description="Classification category counts")
    model_selection_stats: Dict[str, int] = Field(..., description="Model selection counts")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds")


# Error Response Models
class ErrorDetail(BaseModel):
    """Individual error detail for validation errors."""
    field: str = Field(..., description="Field that caused the error")
    message: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type identifier")
    value: Optional[Any] = Field(None, description="Invalid value that caused the error")


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str = Field(..., description="Error type/category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed validation errors")
    request_id: str = Field(..., description="Unique request identifier for tracking")
    timestamp: str = Field(..., description="ISO timestamp when error occurred")
    path: Optional[str] = Field(None, description="Request path where error occurred")
