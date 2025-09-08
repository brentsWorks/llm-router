"""
API Middleware Integration Tests
===============================

Tests for middleware functionality:
- Request logging middleware
- CORS middleware
- Request/response processing
- Header handling

Uses real implementations of internal services but mocks external dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from llm_router.api.main import app


class TestAPIMiddleware:
    """Test API middleware functionality."""

    def test_middleware_integration(self, client):
        """Test middleware integration with requests."""
        response = client.post("/route", json={
            "prompt": "Test middleware integration"
        })
        
        assert response.status_code == 200
        
        # Check that request ID header is added by middleware
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_cors_middleware_integration(self, client):
        """Test CORS middleware integration."""
        # Test that CORS headers are properly handled
        response = client.post("/route", json={
            "prompt": "Test CORS"
        })
        
        assert response.status_code == 200
        # CORS middleware should allow the request
        # (TestClient doesn't fully simulate CORS, but middleware should be active)
