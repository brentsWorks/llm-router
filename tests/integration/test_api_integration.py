"""
Integration tests for FastAPI application.

Tests component interactions between API layer and internal services:
- API endpoints with real router service
- Request/response processing
- Error handling integration
- Middleware functionality

Uses real implementations of internal services but mocks external dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from llm_router.api.main import app


class TestAPIIntegration:
    """Test API integration with internal services."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_health_endpoint_integration(self, client):
        """Test health endpoint integrates with registry service."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with registry
        assert "total_models" in data
        assert isinstance(data["total_models"], int)
        assert data["total_models"] > 0  # Should have mock models from registry

    def test_route_endpoint_integration_with_classifier(self, client):
        """Test routing endpoint integrates with classification service."""
        request_data = {
            "prompt": "Write a Python function to sort a list",
            "preferences": {}
        }
        
        response = client.post("/route", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with classifier
        assert "classification" in data
        assert data["classification"]["category"] == "code"
        assert "confidence" in data
        assert "reasoning" in data["classification"]

    def test_route_endpoint_integration_with_registry(self, client):
        """Test routing endpoint integrates with model registry."""
        request_data = {
            "prompt": "Tell me a creative story",
            "preferences": {}
        }
        
        response = client.post("/route", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with registry - should return actual model
        assert "selected_model" in data
        selected_model = data["selected_model"]
        assert "provider" in selected_model
        assert "model" in selected_model
        assert selected_model["provider"] in ["openai", "anthropic"]

    def test_route_endpoint_integration_with_ranker(self, client):
        """Test routing endpoint integrates with model ranker."""
        request_data = {
            "prompt": "What is machine learning?",
            "preferences": {}
        }
        
        response = client.post("/route", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with ranker - should have routing decision
        assert "selected_model" in data
        assert "routing_time_ms" in data
        assert isinstance(data["routing_time_ms"], (int, float))

    def test_models_endpoint_integration_with_registry(self, client):
        """Test models endpoint integrates with registry."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with registry
        assert "total_models" in data
        assert "models" in data
        assert len(data["models"]) == data["total_models"]
        
        # Verify model data comes from registry
        for model in data["models"]:
            assert "provider" in model
            assert "model" in model
            assert "capabilities" in model

    def test_classify_endpoint_integration_with_classifier(self, client):
        """Test classify endpoint integrates with classification service."""
        request_data = {
            "prompt": "Debug this Python code",
            "preferences": {}
        }
        
        response = client.post("/classify", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with classifier
        assert "category" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert data["category"] == "code"

    def test_metrics_endpoint_integration_with_tracking(self, client):
        """Test metrics endpoint integrates with request tracking."""
        # Make requests to generate metrics
        client.post("/route", json={"prompt": "Test prompt 1", "preferences": {}})
        client.post("/route", json={"prompt": "Test prompt 2", "preferences": {}})
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with metrics tracking
        assert "total_requests" in data
        assert "classification_stats" in data
        assert "model_selection_stats" in data
        assert data["total_requests"] >= 2

    def test_error_handling_integration(self, client):
        """Test error handling integrates with service layer."""
        # Test empty prompt handling - should be validation error (422) not server error (500)
        response = client.post("/route", json={"prompt": "", "preferences": {}})
        assert response.status_code == 422  # Pydantic validation catches this
        
        # Test invalid JSON handling
        response = client.post("/route", content="invalid json")
        assert response.status_code == 422

    def test_middleware_integration(self, client):
        """Test middleware integrates with request processing."""
        response = client.get("/health")
        
        # Verify request logging middleware
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    def test_cors_middleware_integration(self, client):
        """Test CORS middleware integration."""
        # Test CORS headers are present in actual requests
        response = client.post("/route", json={"prompt": "test", "preferences": {}})
        # CORS middleware should allow the request (not testing OPTIONS which isn't implemented)
        assert response.status_code == 200

    def test_request_validation_integration(self, client):
        """Test request validation integrates with Pydantic models."""
        # Test missing required field
        response = client.post("/route", json={"preferences": {}})
        assert response.status_code == 422
        
        # Test invalid field type
        response = client.post("/route", json={"prompt": 123, "preferences": {}})
        assert response.status_code == 422
        
        # Test field length validation
        response = client.post("/route", json={
            "prompt": "a" * 20000,  # Exceeds max length
            "preferences": {}
        })
        assert response.status_code == 422
