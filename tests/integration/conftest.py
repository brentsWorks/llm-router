"""
Shared test configuration for integration tests.
"""

import pytest
from fastapi.testclient import TestClient
from llm_router.api.main import app


@pytest.fixture(scope="session")
def test_app():
    """Create the FastAPI app for testing."""
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)
