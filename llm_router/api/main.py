"""
LLM Router API - Main FastAPI Application
==========================================

This is the main API entry point for the LLM Router service.
Start here for manual exploration, then refactor to modules as needed.

Current Structure:
├── main.py (this file) - FastAPI app, routes, dependencies
├── models.py (future) - Pydantic request/response models
├── dependencies.py (future) - FastAPI dependencies and middleware
└── routers/ (future) - Route-specific modules
    ├── health.py
    ├── routing.py
    └── ...
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time
import logging
import uuid

# Import our existing services
from ..router import RouterService
from ..classification import KeywordClassifier
from ..registry import ProviderRegistry
from ..ranking import ModelRanker
from ..models import RoutingDecision

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API (start simple, expand later)
class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RouteResponse(BaseModel):
    selected_model: Dict[str, Any]
    classification: Dict[str, Any]
    confidence: float
    routing_time_ms: float
    reasoning: Optional[str] = None
    # Client-side execution guidance
    provider_info: Optional[Dict[str, Any]] = None  # API endpoints, model params
    fallback_models: Optional[List[Dict[str, Any]]] = None  # Backup options

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    total_models: int
    total_requests: Optional[int] = None

class MetricsResponse(BaseModel):
    total_requests: int
    avg_response_time_ms: float
    classification_stats: Dict[str, int]
    model_selection_stats: Dict[str, int]
    uptime_seconds: float

# Global service instances (later move to dependency injection)
classifier = KeywordClassifier()
registry = ProviderRegistry()
ranker = ModelRanker()

# Add some mock models for Phase 6.1 testing
from ..registry import ProviderModel, PricingInfo, LimitsInfo, PerformanceInfo

# Mock OpenAI GPT-3.5 Turbo
gpt35_model = ProviderModel(
    provider="openai",
    model="gpt-3.5-turbo",
    capabilities=["code", "creative", "qa"],
    pricing=PricingInfo(input_tokens_per_1k=0.001, output_tokens_per_1k=0.002),
    limits=LimitsInfo(context_length=4096, rate_limit=3500, safety_level="moderate"),
    performance=PerformanceInfo(
        avg_latency_ms=800,
        quality_scores={"code": 0.85, "creative": 0.90, "qa": 0.88}
    )
)

# Mock OpenAI GPT-4
gpt4_model = ProviderModel(
    provider="openai", 
    model="gpt-4",
    capabilities=["code", "creative", "qa"],
    pricing=PricingInfo(input_tokens_per_1k=0.03, output_tokens_per_1k=0.06),
    limits=LimitsInfo(context_length=8192, rate_limit=500, safety_level="high"),
    performance=PerformanceInfo(
        avg_latency_ms=1200,
        quality_scores={"code": 0.95, "creative": 0.92, "qa": 0.94}
    )
)

# Mock Anthropic Claude
claude_model = ProviderModel(
    provider="anthropic",
    model="claude-3-haiku",
    capabilities=["code", "creative", "qa"],
    pricing=PricingInfo(input_tokens_per_1k=0.00025, output_tokens_per_1k=0.00125),
    limits=LimitsInfo(context_length=200000, rate_limit=1000, safety_level="high"),
    performance=PerformanceInfo(
        avg_latency_ms=600,
        quality_scores={"code": 0.82, "creative": 0.95, "qa": 0.90}
    )
)

# Add models to registry
registry.add_model(gpt35_model)
registry.add_model(gpt4_model)
registry.add_model(claude_model)

router_service = RouterService(classifier, registry, ranker)

# FastAPI app
app = FastAPI(
    title="LLM Router API",
    description="Intelligent model selection for optimal LLM routing",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests with timing and request ID."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Started")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global startup time for health checks
START_TIME = time.time()

# Simple metrics tracking (in production, use proper metrics system)
REQUEST_METRICS = {
    "total_requests": 0,
    "total_response_time": 0.0,
    "classification_counts": {"code": 0, "creative": 0, "qa": 0},
    "model_selection_counts": {}
}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy."""
    return HealthResponse(
        status="healthy",
        service="llm-router",
        version="0.1.0",
        uptime_seconds=time.time() - START_TIME,
        total_models=len(registry.get_all_models()),
        total_requests=REQUEST_METRICS["total_requests"]
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API performance metrics."""
    avg_response_time = (
        REQUEST_METRICS["total_response_time"] / REQUEST_METRICS["total_requests"] 
        if REQUEST_METRICS["total_requests"] > 0 else 0.0
    )
    
    return MetricsResponse(
        total_requests=REQUEST_METRICS["total_requests"],
        avg_response_time_ms=avg_response_time * 1000,  # Convert to ms
        classification_stats=REQUEST_METRICS["classification_counts"].copy(),
        model_selection_stats=REQUEST_METRICS["model_selection_counts"].copy(),
        uptime_seconds=time.time() - START_TIME
    )

# Main routing endpoint
@app.post("/route", response_model=RouteResponse)
async def route_prompt(request: RouteRequest):
    """
    Route a prompt to the optimal LLM model.

    This endpoint analyzes the prompt, classifies it, and selects the best model
    based on capabilities, cost, latency, and quality preferences.
    """
    try:
        start_time = time.time()
        logger.info(f"Routing prompt: {request.prompt[:50]}...")

        # Route the prompt using our service
        result = router_service.route(request.prompt)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to route prompt - no suitable model found"
            )

        # Convert to response format
        response = RouteResponse(
            selected_model=result.selected_model.model_dump(),
            classification=result.classification.model_dump(),
            confidence=result.confidence,
            routing_time_ms=result.routing_time_ms,
            reasoning=getattr(result, 'reasoning', None)
        )

        # Update metrics
        routing_duration = time.time() - start_time
        REQUEST_METRICS["total_requests"] += 1
        REQUEST_METRICS["total_response_time"] += routing_duration
        
        # Track classification stats
        category = result.classification.category
        if category in REQUEST_METRICS["classification_counts"]:
            REQUEST_METRICS["classification_counts"][category] += 1
        
        # Track model selection stats
        model_key = f"{result.selected_model.provider}/{result.selected_model.model}"
        REQUEST_METRICS["model_selection_counts"][model_key] = REQUEST_METRICS["model_selection_counts"].get(model_key, 0) + 1

        logger.info(f"Routed to {response.selected_model['provider']}/{response.selected_model['model']} ({routing_duration:.3f}s)")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal routing error: {str(e)}"
        )

# Optional: Add a simple route listing endpoint
@app.get("/models")
async def list_available_models():
    """List all available models (for debugging/exploration)."""
    try:
        models = registry.get_all_models()
        return {
            "total_models": len(models),
            "models": [
                {
                    "provider": model.provider,
                    "model": model.model,
                    "capabilities": model.capabilities,
                    "avg_latency_ms": model.performance.avg_latency_ms,
                    "cost_per_1k_input": model.pricing.input_tokens_per_1k,
                    "quality_scores": model.performance.quality_scores
                }
                for model in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a classification testing endpoint
@app.post("/classify")
async def classify_prompt(request: RouteRequest):
    """Classify a prompt without full routing (for testing)."""
    try:
        classification = classifier.classify(request.prompt)
        if classification is None:
            raise HTTPException(status_code=500, detail="Classification failed")

        return classification.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # For manual testing
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
