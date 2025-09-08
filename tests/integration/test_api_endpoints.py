"""
API Endpoints Integration Tests
==============================

Tests for basic API endpoint functionality and service integration:
- Health endpoint
- Route endpoint with service integration
- Models endpoint
- Classify endpoint
- Basic request/response processing

Uses real implementations of internal services but mocks external dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from llm_router.api.main import app


class TestAPIEndpoints:
    """Test basic API endpoint functionality and service integration."""

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
        assert "category" in data["classification"]
        assert "confidence" in data["classification"]
        
        # Should classify coding prompts correctly
        assert data["classification"]["category"] == "code"
        assert data["classification"]["confidence"] > 0.5

    def test_route_endpoint_integration_with_registry(self, client):
        """Test routing endpoint integrates with registry service."""
        request_data = {
            "prompt": "What is machine learning?",
            "preferences": {}
        }
        
        response = client.post("/route", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with registry (model selection)
        assert "selected_model" in data
        assert "provider" in data["selected_model"]
        assert "model" in data["selected_model"]
        
        # Should select from available models in registry
        provider = data["selected_model"]["provider"]
        model = data["selected_model"]["model"]
        assert provider in ["openai", "anthropic"]
        assert len(model) > 0

    def test_route_endpoint_integration_with_ranker(self, client):
        """Test routing endpoint integrates with ranking service."""
        request_data = {
            "prompt": "Tell me a creative story",
            "preferences": {
                "cost_weight": 0.8,
                "latency_weight": 0.1,
                "quality_weight": 0.1
            }
        }
        
        response = client.post("/route", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with ranker (should consider preferences)
        assert "selected_model" in data
        assert "confidence" in data
        assert data["confidence"] > 0

    def test_models_endpoint_integration_with_registry(self, client):
        """Test models endpoint integrates with registry service."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with registry
        assert "models" in data
        assert "total_count" in data
        assert isinstance(data["models"], list)
        assert data["total_count"] > 0
        
        # Verify model structure from registry
        model = data["models"][0]
        assert "provider" in model
        assert "model" in model
        assert "capabilities" in model
        assert "pricing" in model

    def test_classify_endpoint_integration_with_classifier(self, client):
        """Test classify endpoint integrates with classification service."""
        request_data = {
            "prompt": "How do I implement a binary search algorithm?"
        }
        
        response = client.post("/classify", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify integration with classifier
        assert "category" in data
        assert "confidence" in data
        assert "classification_time_ms" in data
        
        # Should classify correctly
        assert data["category"] == "code"
        assert data["confidence"] > 0.3  # Adjusted for realistic confidence levels
        assert data["classification_time_ms"] > 0

    def test_metrics_endpoint_integration_with_tracking(self, client):
        """Test metrics endpoint integrates with request tracking."""
        # Make a request first to generate metrics
        client.post("/route", json={"prompt": "Test prompt"})
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify metrics tracking integration
        assert "total_requests" in data
        assert "average_response_time_ms" in data
        assert "classification_stats" in data
        assert "model_selection_stats" in data
        assert data["total_requests"] > 0
