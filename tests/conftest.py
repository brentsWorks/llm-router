"""Shared test configuration and fixtures."""

import pytest
from llm_router import Router
from llm_router.config import create_configured_registry


@pytest.fixture
def router():
    """Create a Router instance for testing."""
    return Router()


@pytest.fixture
def configured_registry():
    """Create a fresh registry with models loaded from config for testing."""
    return create_configured_registry()
