"""
Shared test configuration for integration tests.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from llm_router.api.main import app
from llm_router.classification import KeywordClassifier


@pytest.fixture(scope="session")
def test_app():
    """Create the FastAPI app for testing with mocked classifier."""
    # Mock the classifier factory to return a KeywordClassifier instead of HybridClassifier
    with patch('llm_router.api.main.create_classifier') as mock_create_classifier:
        mock_create_classifier.return_value = KeywordClassifier()
        
        # Also mock the classifier info to avoid API key checks
        with patch('llm_router.api.main.get_classifier_info') as mock_get_info:
            mock_get_info.return_value = {
                "type": "keyword",
                "description": "Rule-based keyword classifier (test mode)",
                "capabilities": ["fast_classification", "keyword_matching"],
                "fallback_available": False
            }
            yield app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)
