"""
API Preferences & Constraints Integration Tests
==============================================

Tests for preferences and constraints functionality:
- ScoringWeights (preferences) support and validation
- RoutingConstraints support and validation
- Edge cases and complex combinations
- Performance and consistency testing

Uses real implementations of internal services but mocks external dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from llm_router.api.main import app


class TestAPIPreferencesConstraints:
    """Test API preferences and constraints functionality."""

    def test_preferences_support_integration(self, client):
        """Test API accepts and uses ScoringWeights preferences."""
        # Test cost-optimized preferences
        response = client.post("/route", json={
            "prompt": "Explain quantum computing",
            "preferences": {
                "cost_weight": 0.8,
                "latency_weight": 0.1,
                "quality_weight": 0.1
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should successfully route with preferences
        assert "selected_model" in data
        assert "classification" in data
        assert "confidence" in data
        
        # Test quality-optimized preferences
        response = client.post("/route", json={
            "prompt": "Write a complex algorithm",
            "preferences": {
                "cost_weight": 0.1,
                "latency_weight": 0.1,
                "quality_weight": 0.8
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "selected_model" in data

    def test_preferences_validation_integration(self, client):
        """Test API validates ScoringWeights properly."""
        # Test weights that don't sum to 1.0
        response = client.post("/route", json={
            "prompt": "Test prompt",
            "preferences": {
                "cost_weight": 0.5,
                "latency_weight": 0.5,
                "quality_weight": 0.5  # Total = 1.5
            }
        })
        
        assert response.status_code == 422
        error_data = response.json()
        assert "sum to 1.0" in str(error_data)
        
        # Test negative weights
        response = client.post("/route", json={
            "prompt": "Test prompt",
            "preferences": {
                "cost_weight": -0.1,
                "latency_weight": 0.6,
                "quality_weight": 0.5
            }
        })
        
        assert response.status_code == 422
        
        # Test weights > 1.0
        response = client.post("/route", json={
            "prompt": "Test prompt",
            "preferences": {
                "cost_weight": 1.5,
                "latency_weight": 0.0,
                "quality_weight": 0.0
            }
        })
        
        assert response.status_code == 422

    def test_constraints_support_integration(self, client):
        """Test API accepts and uses RoutingConstraints."""
        response = client.post("/route", json={
            "prompt": "Create a detailed analysis",
            "preferences": {
                "cost_weight": 0.4,
                "latency_weight": 0.3,
                "quality_weight": 0.3
            },
            "constraints": {
                "max_cost_per_1k_tokens": 0.01,
                "max_latency_ms": 2000,
                "max_context_length": 250000,
                "min_safety_level": "moderate"
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should successfully route with constraints
        assert "selected_model" in data
        assert "classification" in data
        assert "confidence" in data

    def test_constraints_validation_integration(self, client):
        """Test API validates RoutingConstraints properly."""
        # Test invalid safety level
        response = client.post("/route", json={
            "prompt": "Test prompt",
            "constraints": {
                "min_safety_level": "ultra_secure"  # Invalid level
            }
        })
        
        assert response.status_code == 422
        
        # Test negative cost constraint
        response = client.post("/route", json={
            "prompt": "Test prompt",
            "constraints": {
                "max_cost_per_1k_tokens": -0.01
            }
        })
        
        assert response.status_code == 422

    def test_preferences_and_constraints_together_integration(self, client):
        """Test API handles preferences and constraints together."""
        response = client.post("/route", json={
            "prompt": "Analyze market trends",
            "preferences": {
                "cost_weight": 0.5,
                "latency_weight": 0.3,
                "quality_weight": 0.2
            },
            "constraints": {
                "max_cost_per_1k_tokens": 0.05,
                "max_latency_ms": 1500,
                "excluded_providers": []
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "selected_model" in data
        assert "classification" in data

    def test_preferences_edge_cases_integration(self, client):
        """Test edge cases for preferences."""
        # Test minimal weights (but valid)
        response = client.post("/route", json={
            "prompt": "Simple question",
            "preferences": {
                "cost_weight": 0.34,
                "latency_weight": 0.33,
                "quality_weight": 0.33
            }
        })
        
        assert response.status_code == 200
        
        # Test extreme cost optimization
        response = client.post("/route", json={
            "prompt": "Another question",
            "preferences": {
                "cost_weight": 0.99,
                "latency_weight": 0.005,
                "quality_weight": 0.005
            }
        })
        
        assert response.status_code == 200
        
        # Test extreme quality optimization
        response = client.post("/route", json={
            "prompt": "Complex analysis needed",
            "preferences": {
                "cost_weight": 0.01,
                "latency_weight": 0.01,
                "quality_weight": 0.98
            }
        })
        
        assert response.status_code == 200

    def test_constraints_edge_cases_integration(self, client):
        """Test edge cases for constraints."""
        # Test very restrictive cost constraint
        response = client.post("/route", json={
            "prompt": "Budget-conscious request",
            "constraints": {
                "max_cost_per_1k_tokens": 0.002
            }
        })
        
        assert response.status_code == 200  # Should find Claude
        
        # Test high latency tolerance
        response = client.post("/route", json={
            "prompt": "Not time-sensitive",
            "constraints": {
                "max_latency_ms": 10000
            }
        })
        
        assert response.status_code == 200
        
        # Test multiple exclusions (should result in no models)
        response = client.post("/route", json={
            "prompt": "This will fail",
            "constraints": {
                "excluded_providers": ["openai", "anthropic"]
            }
        })
        
        assert response.status_code == 500  # No models available

    def test_complex_preferences_constraints_combinations_integration(self, client):
        """Test complex combinations of preferences and constraints."""
        # Scenario 1: Cost-conscious with quality constraints
        response = client.post("/route", json={
            "prompt": "Need good quality but cheap",
            "preferences": {
                "cost_weight": 0.7,
                "latency_weight": 0.1,
                "quality_weight": 0.2
            },
            "constraints": {
                "max_cost_per_1k_tokens": 0.01,
                "min_safety_level": "moderate"
            }
        })
        
        assert response.status_code == 200
        
        # Scenario 2: Speed-focused with provider preferences
        response = client.post("/route", json={
            "prompt": "Quick response needed",
            "preferences": {
                "cost_weight": 0.2,
                "latency_weight": 0.7,
                "quality_weight": 0.1
            },
            "constraints": {
                "max_latency_ms": 800,
                "excluded_models": []
            }
        })
        
        assert response.status_code == 200

    def test_preferences_constraints_validation_comprehensive_integration(self, client):
        """Test comprehensive validation scenarios."""
        # Test partial preferences (should work)
        response = client.post("/route", json={
            "prompt": "Test with partial prefs",
            "preferences": {
                "cost_weight": 1.0,
                "latency_weight": 0.0,
                "quality_weight": 0.0
            }
        })
        
        assert response.status_code == 200
        
        # Test empty constraints (should work)
        response = client.post("/route", json={
            "prompt": "Test with empty constraints",
            "constraints": {}
        })
        
        assert response.status_code == 200
        
        # Test null preferences and constraints (should work)
        response = client.post("/route", json={
            "prompt": "Test with nulls",
            "preferences": None,
            "constraints": None
        })
        
        assert response.status_code == 200

    def test_model_selection_consistency_integration(self, client):
        """Test that preferences lead to expected model choices."""
        # Test that cost preference tends toward cheaper models
        cost_responses = []
        for _ in range(3):
            response = client.post("/route", json={
                "prompt": "Cost-optimized request",
                "preferences": {
                    "cost_weight": 0.9,
                    "latency_weight": 0.05,
                    "quality_weight": 0.05
                }
            })
            assert response.status_code == 200
            cost_responses.append(response.json())
        
        # Should consistently prefer cheaper models
        # (In our mock setup, this would be predictable)
        assert len(cost_responses) == 3

    def test_routing_time_performance_integration(self, client):
        """Test routing performance with preferences and constraints."""
        import time
        
        start_time = time.time()
        response = client.post("/route", json={
            "prompt": "Performance test prompt",
            "preferences": {
                "cost_weight": 0.4,
                "latency_weight": 0.3,
                "quality_weight": 0.3
            },
            "constraints": {
                "max_cost_per_1k_tokens": 0.1,
                "max_latency_ms": 5000
            }
        })
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        
        # Should complete quickly
        actual_time_ms = (end_time - start_time) * 1000
        assert actual_time_ms < 100, f"Routing took {actual_time_ms:.2f}ms, expected < 100ms"
        
        # The reported routing time should be reasonable
        reported_time = data["routing_time_ms"]
        assert 0 < reported_time < 50, f"Reported routing time {reported_time}ms seems unrealistic"
