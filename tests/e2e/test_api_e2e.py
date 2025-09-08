"""
End-to-end tests for the complete LLM Router API workflow.

Tests complete user workflows through API endpoints:
- Full routing pipeline from prompt to model selection
- Real user scenarios and use cases
- Performance and concurrency testing
- API documentation accessibility

These tests use the complete system with realistic scenarios.
"""

import pytest
import concurrent.futures
import time
from fastapi.testclient import TestClient
from llm_router.api.main import app


class TestAPIEndToEnd:
    """Test complete end-to-end API workflows."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_complete_routing_workflow_e2e(self, client):
        """Test complete user workflow: health -> models -> route -> metrics."""
        # Step 1: User checks if service is healthy
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: User explores available models
        models_response = client.get("/models")
        assert models_response.status_code == 200
        models_data = models_response.json()
        assert len(models_data["models"]) > 0
        available_providers = {model["provider"] for model in models_data["models"]}
        assert "openai" in available_providers or "anthropic" in available_providers
        
        # Step 3: User routes a coding task
        code_request = {
            "prompt": "Write a Python function to implement binary search",
            "preferences": {"quality_priority": "high"}
        }
        route_response = client.post("/route", json=code_request)
        assert route_response.status_code == 200
        route_data = route_response.json()
        
        # Verify routing worked correctly for code task
        assert route_data["classification"]["category"] == "code"
        assert route_data["confidence"] > 0.0
        assert "selected_model" in route_data
        
        # Step 4: User checks service metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        assert metrics_data["total_requests"] >= 1
        assert "code" in metrics_data["classification_stats"]

    def test_different_task_types_e2e(self, client):
        """Test end-to-end workflow for different types of tasks."""
        test_cases = [
            {
                "prompt": "Write a Python class for a binary tree",
                "expected_category": "code",
                "description": "coding task"
            },
            {
                "prompt": "Write a short story about a time traveler",
                "expected_category": "creative", 
                "description": "creative writing task"
            },
            {
                "prompt": "What are the main causes of climate change?",
                "expected_category": "qa",
                "description": "question answering task"
            }
        ]
        
        for test_case in test_cases:
            response = client.post("/route", json={
                "prompt": test_case["prompt"],
                "preferences": {}
            })
            
            assert response.status_code == 200, f"Failed for {test_case['description']}"
            data = response.json()
            
            assert data["classification"]["category"] == test_case["expected_category"]
            assert data["confidence"] > 0.0
            assert "selected_model" in data

    def test_user_preferences_impact_e2e(self, client):
        """Test that user preferences actually impact model selection."""
        base_prompt = "Explain artificial intelligence concepts"
        
        # Request with cost preference
        cost_focused_request = {
            "prompt": base_prompt,
            "preferences": {"cost_priority": "high", "quality_priority": "low"}
        }
        
        # Request with quality preference  
        quality_focused_request = {
            "prompt": base_prompt,
            "preferences": {"cost_priority": "low", "quality_priority": "high"}
        }
        
        cost_response = client.post("/route", json=cost_focused_request)
        quality_response = client.post("/route", json=quality_focused_request)
        
        assert cost_response.status_code == 200
        assert quality_response.status_code == 200
        
        # Both should work (preferences are hints, not requirements)
        cost_data = cost_response.json()
        quality_data = quality_response.json()
        
        assert "selected_model" in cost_data
        assert "selected_model" in quality_data

    def test_concurrent_users_e2e(self, client):
        """Test API handles multiple concurrent users properly."""
        def simulate_user_workflow():
            # Each user does a complete workflow
            health_check = client.get("/health")
            if health_check.status_code != 200:
                return False
                
            route_request = client.post("/route", json={
                "prompt": "What is quantum computing?",
                "preferences": {}
            })
            return route_request.status_code == 200
        
        # Simulate 10 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(simulate_user_workflow) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All users should succeed
        assert all(results), "Some concurrent users failed"
        
        # Check metrics reflect all requests
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        assert metrics_data["total_requests"] >= 10

    def test_api_performance_e2e(self, client):
        """Test API performance meets requirements."""
        test_prompts = [
            "Write a function to sort data",
            "Create a story about robots", 
            "Explain machine learning",
            "Debug this code snippet",
            "What is the meaning of life?"
        ]
        
        response_times = []
        
        for prompt in test_prompts:
            start_time = time.time()
            response = client.post("/route", json={
                "prompt": prompt,
                "preferences": {}
            })
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Each request should be reasonably fast
            assert response_time < 1.0, f"Request took too long: {response_time:.2f}s"
        
        # Average response time should be good
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.5, f"Average response time too slow: {avg_response_time:.2f}s"

    def test_api_documentation_accessibility_e2e(self, client):
        """Test that API documentation is accessible to users."""
        # Test Swagger UI
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
        
        # Test ReDoc
        redoc_response = client.get("/redoc")
        assert redoc_response.status_code == 200
        
        # Test OpenAPI schema
        schema_response = client.get("/openapi.json")
        assert schema_response.status_code == 200
        schema_data = schema_response.json()
        
        # Verify schema contains our endpoints
        paths = schema_data.get("paths", {})
        assert "/health" in paths
        assert "/route" in paths
        assert "/models" in paths
        assert "/classify" in paths
        assert "/metrics" in paths

    def test_error_scenarios_e2e(self, client):
        """Test complete error handling workflows."""
        error_scenarios = [
            {
                "request": {"prompt": "", "preferences": {}},
                "description": "empty prompt",
                "expected_status": 422  # Caught by Pydantic validation
            },
            {
                "request": {"prompt": "   \n\t   ", "preferences": {}},
                "description": "whitespace-only prompt", 
                "expected_status": 500  # Caught by business logic
            },
            {
                "request": {"preferences": {}},
                "description": "missing prompt field",
                "expected_status": 422  # Caught by Pydantic validation
            }
        ]
        
        for scenario in error_scenarios:
            response = client.post("/route", json=scenario["request"])
            assert response.status_code == scenario["expected_status"], \
                f"Wrong status for {scenario['description']}"
            
            # Should have error details (using our custom error format)
            error_data = response.json()
            assert "error" in error_data or "detail" in error_data, f"Missing error info in response: {error_data}"

    def test_service_resilience_e2e(self, client):
        """Test service resilience under various conditions."""
        # Test with very long prompt (but within limits)
        long_prompt = "Explain this concept: " + "detail " * 100
        response = client.post("/route", json={
            "prompt": long_prompt,
            "preferences": {}
        })
        assert response.status_code == 200
        
        # Test with complex preferences
        complex_preferences = {
            "cost_priority": "medium",
            "quality_priority": "high", 
            "speed_priority": "low",
            "custom_field": "should be ignored"
        }
        response = client.post("/route", json={
            "prompt": "Simple test prompt",
            "preferences": complex_preferences
        })
        assert response.status_code == 200
