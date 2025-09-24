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
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time
import logging
import uuid
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our existing services
from ..router import RouterService
from ..classification import KeywordClassifier
from ..ranking import ModelRanker
from ..models import RoutingDecision
from ..scoring import ScoringWeights
from ..constraints import RoutingConstraints
from ..classifier_factory import create_classifier, get_classifier_info

# Import configuration
from ..config import create_configured_registry

# Import API models
from .models import (
    RouteRequest, RouteResponse, ExecuteResponse, HealthResponse, MetricsResponse,
    ModelsResponse, ModelInfo, ClassifyRequest, ClassifyResponse,
    ErrorResponse, ErrorDetail
)

# Import our enhanced logger
from .logger import get_api_logger

# Import OpenRouter service
from ..openrouter_service import OpenRouterService, LLMExecutionRequest, LLMExecutionResponse

# Get the API logger instance
api_logger = get_api_logger()
logger = api_logger.logger  # For backwards compatibility

# API models are now imported from .models module

# Global service instances (later move to dependency injection)
classifier = create_classifier("hybrid")  # Try hybrid, fallback to keyword if needed
registry = create_configured_registry()  # Load models from configuration
ranker = ModelRanker()

router_service = RouterService(classifier, registry, ranker)

# Initialize OpenRouter service
from ..openrouter_client import OpenRouterClient
openrouter_client = OpenRouterClient()
openrouter_service = OpenRouterService(openrouter_client)

# FastAPI app
app = FastAPI(
    title="LLM Router API",
    description="""
    **Intelligent Model Selection for Optimal LLM Routing**
    
    The LLM Router API automatically selects the best language model for your specific use case 
    based on your preferences and constraints.
    
    ## 🎯 Key Features
    - **Intelligent Classification**: Automatically categorizes prompts (code, creative, Q&A)
    - **Flexible Preferences**: Optimize for cost, latency, or quality with custom weights
    - **Smart Constraints**: Filter models by cost, speed, safety, providers, and more
    - **Multiple Providers**: Support for OpenAI, Anthropic, and other LLM providers
    - **Real-time Routing**: Fast model selection with sub-100ms response times
    
    ## 🚀 Getting Started
    1. Check available models: `GET /models`
    2. Route your first prompt: `POST /route`
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "LLM Router",
        "url": "https://github.com/brentsworks/llm-router",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Custom Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with custom format and monitoring."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    # Convert Pydantic errors to our custom format
    details = []
    field_names = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"] if loc != "body")
        field_names.append(field_path or "request")
        # Handle non-serializable values (like bytes)
        input_value = error.get("input")
        if isinstance(input_value, bytes):
            input_value = f"<bytes: {len(input_value)} bytes>"
        elif not isinstance(input_value, (str, int, float, bool, list, dict, type(None))):
            input_value = str(input_value)
        
        details.append(ErrorDetail(
            field=field_path or "request",
            message=error["msg"],
            type=error["type"],
            value=input_value
        ))
    
    # Create a more descriptive message
    fields_str = ", ".join(field_names) if field_names else "request"
    message = f"Validation failed for field(s): {fields_str}"
    
    # Log validation error with enhanced context
    api_logger.log_validation_error(
        request_id=request_id,
        path=str(request.url.path),
        field_errors=field_names
    )
    
    error_response = ErrorResponse(
        error="Validation Error",
        message=message,
        details=details,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(request.url.path)
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with custom format."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    # Map status codes to error types
    error_types = {
        400: "Bad Request",
        401: "Unauthorized", 
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable"
    }
    
    error_response = ErrorResponse(
        error=error_types.get(exc.status_code, "HTTP Error"),
        message=exc.detail,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(request.url.path)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(json.JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    """Handle malformed JSON requests."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    error_response = ErrorResponse(
        error="Bad Request",
        message=f"Invalid JSON format: {str(exc)}",
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(request.url.path)
    )
    
    return JSONResponse(
        status_code=400,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions with secure error responses and monitoring."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())[:8])
    
    # Log the actual error with enhanced monitoring
    api_logger.log_internal_error(
        request_id=request_id,
        path=str(request.url.path),
        error=exc
    )
    
    # Return a generic error response that doesn't expose internal details
    error_response = ErrorResponse(
        error="Internal Server Error",
        message="An internal server error occurred. Please try again later.",
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(request.url.path)
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


# Enhanced request validation and logging middleware
@app.middleware("http")
async def enhanced_request_middleware(request: Request, call_next):
    """Enhanced middleware for request validation, logging, and error context."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Store request ID in request state for error handlers
    request.state.request_id = request_id
    
    # Get client context
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Log request start with enhanced context
    api_logger.log_request_start(
        request_id=request_id,
        method=request.method,
        path=str(request.url.path),
        client_ip=client_ip,
        user_agent=user_agent
    )
    
    # Request validation and sanitization
    try:
        # Validate request size (basic protection against large payloads)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
            api_logger.log_large_request_blocked(request_id, content_length)
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Request Too Large",
                    "message": "Request payload exceeds maximum size limit (1MB)",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "path": str(request.url.path)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration and log response
        duration = time.time() - start_time
        
        # Log request completion
        api_logger.log_request_end(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration=duration
        )
        
        # Add enhanced headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        # Add monitoring metrics if available
        if hasattr(request.state, 'metrics'):
            response.headers["X-Classification-Time"] = f"{request.state.metrics.get('classification_time', 0):.3f}ms"
            response.headers["X-Routing-Time"] = f"{request.state.metrics.get('routing_time', 0):.3f}ms"
        
        return response
        
    except Exception as e:
        # Log middleware errors
        duration = time.time() - start_time
        api_logger.log_middleware_error(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            error=e,
            duration=duration
        )
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An error occurred processing your request",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(request.url.path)
            }
        )

# Add CORS middleware with debugging
allowed_origins = [
    "https://cooperative-reflection-production-89f2.up.railway.app",
    "https://cooperative-reflection-production-89f2.up.railway.app/",
    "http://localhost:3000",
    "http://localhost:5173"
]

# Log CORS configuration at startup
logger.info(f"CORS configuration - Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to log CORS debugging (after CORS middleware)
@app.middleware("http")
async def cors_debug_middleware(request: Request, call_next):
    """Debug CORS issues by logging request origins."""
    origin = request.headers.get("origin")
    if origin:
        logger.info(f"Request from origin: {origin}")

    response = await call_next(request)

    # Log CORS headers in response
    if "access-control-allow-origin" in response.headers:
        logger.info(f"CORS Allow-Origin set to: {response.headers['access-control-allow-origin']}")
    else:
        logger.warning(f"No CORS Allow-Origin header set for origin: {origin}")

    return response

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
        version="0.1.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        total_models=len(registry.get_all_models()),
        total_requests=REQUEST_METRICS["total_requests"]
    )

@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches (for development)."""
    try:
        # Clear embedding cache if available
        if hasattr(classifier, 'embedding_service') and hasattr(classifier.embedding_service, 'clear_cache'):
            classifier.embedding_service.clear_cache()
        
        return {
            "status": "success",
            "message": "Caches cleared successfully",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to clear cache: {str(e)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API performance metrics."""
    avg_response_time = (
        REQUEST_METRICS["total_response_time"] / REQUEST_METRICS["total_requests"] 
        if REQUEST_METRICS["total_requests"] > 0 else 0.0
    )
    
    return MetricsResponse(
        total_requests=REQUEST_METRICS["total_requests"],
        average_response_time_ms=avg_response_time * 1000,  # Convert to ms
        classification_stats=REQUEST_METRICS["classification_counts"].copy(),
        model_selection_stats=REQUEST_METRICS["model_selection_counts"].copy(),
        uptime_seconds=time.time() - START_TIME
    )


@app.get("/monitoring/errors")
async def get_error_metrics():
    """Get detailed error metrics for monitoring and alerting."""
    return api_logger.get_error_metrics()


@app.get("/monitoring/health")
async def get_health_status():
    """Get health status based on error rates and system status."""
    health_data = api_logger.get_health_status()
    
    # Add additional health checks
    health_data.update({
        "uptime_seconds": time.time() - START_TIME,
        "total_models": len(registry.get_all_models()),
        "total_requests": REQUEST_METRICS["total_requests"]
    })
    
    return health_data

# Main routing endpoint
@app.post("/route", response_model=RouteResponse)
async def route_prompt(request: RouteRequest):
    """
    Route a prompt to the optimal LLM model.

    This endpoint intelligently analyzes your prompt and selects the best available model
    based on your specified preferences and constraints.

    ## Request Parameters
    - **prompt** (required): The text prompt to route
    - **preferences** (optional): Scoring weights for optimization
      - `cost_weight`: Weight for cost optimization (0.0-1.0)
      - `latency_weight`: Weight for latency optimization (0.0-1.0) 
      - `quality_weight`: Weight for quality optimization (0.0-1.0)
      - Note: Weights must sum to 1.0
    - **constraints** (optional): Filtering constraints
      - `max_cost_per_1k_tokens`: Maximum cost per 1K tokens
      - `max_latency_ms`: Maximum acceptable latency in milliseconds
      - `max_context_length`: Maximum context length required
      - `min_safety_level`: Minimum safety level ("low", "moderate", "high")
      - `excluded_providers`: List of providers to exclude
      - `excluded_models`: List of specific models to exclude

    ## Response
    Returns detailed information about the selected model, classification results,
    confidence scores, and routing performance metrics.

    ## Examples
    See `examples/api_usage_examples.py` for comprehensive usage examples.
    """
    try:
        start_time = time.time()
        logger.info(f"Routing prompt: {request.prompt[:50]}...")

        # Route the prompt using our service
        result = router_service.route(request.prompt, preferences=request.preferences, constraints=request.constraints)
        
        # DEBUG: Log the actual selection
        if result:
            logger.info(f"DEBUG: Backend selected {result.selected_model.provider}/{result.selected_model.model}")
            print(f"DEBUG: Backend selected {result.selected_model.provider}/{result.selected_model.model}")  # Also print to console

        if result is None:
            # Log routing error
            api_logger.log_routing_error(
                request_id=getattr(request, 'state', {}).get('request_id', 'unknown'),
                error_message="Failed to route prompt - no suitable model found"
            )
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
        # Let the general exception handler deal with this
        # It will log appropriately and return a secure error response
        raise

# Execute endpoint with OpenRouter integration
@app.post("/execute", response_model=ExecuteResponse)
async def execute_prompt(request: RouteRequest):
    """
    Route a prompt to the optimal LLM model and execute it via OpenRouter.

    This endpoint provides end-to-end LLM execution:
    1. Classifies your prompt using RAG/LLM fallback
    2. Selects the best model based on preferences/constraints
    3. Executes the prompt via OpenRouter API
    4. Returns both routing decision AND actual LLM response

    ## Request Parameters
    Same as `/route` endpoint:
    - **prompt** (required): The text prompt to route and execute
    - **preferences** (optional): Scoring weights for optimization
    - **constraints** (optional): Filtering constraints

    ## Response
    Returns the routing decision plus the actual LLM response content,
    execution time, and usage information.

    ## Examples
    ```json
    {
      "prompt": "Write a Python function to calculate fibonacci numbers",
      "preferences": {"quality_weight": 0.8, "cost_weight": 0.2}
    }
    ```
    """
    try:
        start_time = time.time()
        logger.info(f"Executing prompt: {request.prompt[:50]}...")

        # First, route the prompt using our service
        routing_result = router_service.route(
            request.prompt, 
            preferences=request.preferences, 
            constraints=request.constraints
        )

        if routing_result is None:
            # Log routing error
            api_logger.log_routing_error(
                request_id=getattr(request, 'state', {}).get('request_id', 'unknown'),
                error_message="Failed to route prompt - no suitable model found"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to route prompt - no suitable model found"
            )

        # Create execution request for OpenRouter
        execution_request = LLMExecutionRequest(
            prompt=request.prompt,
            routing_decision=routing_result,
            temperature=0.7,
            max_tokens=1000
        )

        # Execute via OpenRouter
        execution_result = openrouter_service.execute_prompt(execution_request)

        # Convert to response format
        response = ExecuteResponse(
            selected_model=routing_result.selected_model.model_dump(),
            classification=routing_result.classification.model_dump(),
            confidence=routing_result.confidence,
            routing_time_ms=routing_result.routing_time_ms,
            reasoning=getattr(routing_result, 'reasoning', None),
            llm_response=execution_result.content,
            model_used=execution_result.model_used,
            execution_time_ms=execution_result.execution_time_ms,
            usage=execution_result.usage,
            finish_reason=execution_result.finish_reason
        )

        # Update metrics
        total_duration = time.time() - start_time
        REQUEST_METRICS["total_requests"] += 1
        REQUEST_METRICS["total_response_time"] += total_duration
        
        # Track classification stats
        category = routing_result.classification.category
        if category in REQUEST_METRICS["classification_counts"]:
            REQUEST_METRICS["classification_counts"][category] += 1
        
        # Track model selection stats
        model_key = f"{routing_result.selected_model.provider}/{routing_result.selected_model.model}"
        REQUEST_METRICS["model_selection_counts"][model_key] = REQUEST_METRICS["model_selection_counts"].get(model_key, 0) + 1

        logger.info(f"Executed with {response.model_used} ({total_duration:.3f}s total, {execution_result.execution_time_ms:.1f}ms LLM)")
        return response

    except HTTPException:
        raise
    except Exception as e:
        # Let the general exception handler deal with this
        # It will log appropriately and return a secure error response
        raise

# Optional: Add a simple route listing endpoint
@app.get("/classifier")
async def get_classifier_info_endpoint() -> Dict[str, Any]:
    """Get information about the currently active classifier."""
    try:
        classifier_info = get_classifier_info(classifier)
        return {
            "classifier": classifier_info,
            "status": "active",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting classifier info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get classifier information")

@app.get("/models", response_model=ModelsResponse)
async def list_available_models():
    """List all available models (for debugging/exploration)."""
    try:
        models = registry.get_all_models()
        model_info_list = []
        
        for model in models:
            model_info = ModelInfo(
                provider=model.provider,
                model=model.model,
                capabilities=model.capabilities,
                pricing={
                    "input_tokens_per_1k": model.pricing.input_tokens_per_1k,
                    "output_tokens_per_1k": model.pricing.output_tokens_per_1k
                },
                limits={
                    "context_length": model.limits.context_length,
                    "rate_limit": model.limits.rate_limit,
                    "safety_level": model.limits.safety_level
                },
                performance={
                    "avg_latency_ms": model.performance.avg_latency_ms,
                    "quality_scores": model.performance.quality_scores or {}
                }
            )
            model_info_list.append(model_info)
        
        return ModelsResponse(
            models=model_info_list,
            total_count=len(models)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a classification testing endpoint
@app.post("/classify", response_model=ClassifyResponse)
async def classify_prompt(request: ClassifyRequest):
    """Classify a prompt without full routing (for testing)."""
    try:
        start_time = time.time()
        classification = classifier.classify(request.prompt)
        classification_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if classification is None:
            # Log classification error
            api_logger.log_classification_error(
                request_id=getattr(request, 'state', {}).get('request_id', 'unknown'),
                error_message="Classification failed - no result returned"
            )
            raise HTTPException(status_code=500, detail="Classification failed")

        return ClassifyResponse(
            category=classification.category,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            classification_time_ms=classification_time
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # For manual testing
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
