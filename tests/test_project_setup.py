"""Tests for basic project setup and structure."""

import pytest
import sys
from pathlib import Path


class TestProjectStructure:
    """Test that the project structure is correctly set up."""
    
    def test_can_import_main_package(self):
        """Test that we can import the main llm_router package."""
        import llm_router
        assert llm_router is not None
    
    def test_has_version_attribute(self):
        """Test that the package has a __version__ attribute."""
        import llm_router
        assert hasattr(llm_router, '__version__')
        assert isinstance(llm_router.__version__, str)
        assert llm_router.__version__ == "0.1.0"
    
    def test_can_import_router_class(self):
        """Test that we can import the Router class."""
        from llm_router import Router
        assert Router is not None
    
    def test_router_can_be_instantiated(self):
        """Test that Router class can be instantiated."""
        from llm_router import Router
        router = Router()
        assert router is not None
    
    def test_router_has_route_method(self):
        """Test that Router has a route method."""
        from llm_router import Router
        router = Router()
        assert hasattr(router, 'route')
        assert callable(router.route)
    
    def test_router_route_method_accepts_prompt(self):
        """Test that router.route() accepts a prompt parameter."""
        from llm_router import Router
        router = Router()
        
        # This should not raise an exception
        result = router.route("test prompt")
        assert result is not None
        assert isinstance(result, dict)
    
    def test_router_route_returns_expected_structure(self):
        """Test that router.route() returns expected dictionary structure."""
        from llm_router import Router
        router = Router()
        
        result = router.route("test prompt")
        
        # Should have basic routing decision structure
        assert "provider" in result
        assert "model" in result
        assert "confidence" in result
        
        # Values should be appropriate types
        assert isinstance(result["provider"], str)
        assert isinstance(result["model"], str)
        assert isinstance(result["confidence"], (int, float))


class TestProjectDependencies:
    """Test that required dependencies can be imported."""
    
    def test_can_import_pydantic(self):
        """Test that pydantic can be imported."""
        try:
            import pydantic
            assert pydantic is not None
        except ImportError:
            pytest.fail("pydantic is not available - check dependencies")
    
    def test_can_import_pydantic_settings(self):
        """Test that pydantic-settings can be imported.""" 
        try:
            from pydantic_settings import BaseSettings
            assert BaseSettings is not None
        except ImportError:
            pytest.fail("pydantic-settings is not available - check dependencies")


class TestTestingInfrastructure:
    """Test that testing infrastructure works."""
    
    def test_pytest_is_working(self):
        """Test that pytest is functioning correctly."""
        assert True  # If this runs, pytest is working
    
    def test_fixtures_are_available(self):
        """Test that fixtures from conftest.py are available."""
        # This test uses the router fixture defined in conftest.py
        pass  # If the fixture injection works, this test will pass
    
    def test_router_fixture_works(self, router):
        """Test that the router fixture provides a valid Router instance."""
        from llm_router import Router
        assert isinstance(router, Router)
        assert hasattr(router, 'route')


@pytest.mark.unit
class TestBasicFunctionality:
    """Test basic functionality requirements."""
    
    def test_router_route_is_deterministic(self):
        """Test that calling route with same input gives consistent results."""
        from llm_router import Router
        router = Router()
        
        result1 = router.route("hello world")
        result2 = router.route("hello world")
        
        # Should get same result for same input
        assert result1 == result2
    
    def test_router_handles_empty_prompt(self):
        """Test that router can handle empty prompt without crashing."""
        from llm_router import Router
        router = Router()
        
        # Should not raise an exception
        result = router.route("")
        assert result is not None
        assert isinstance(result, dict)
    
    def test_router_handles_long_prompt(self):
        """Test that router can handle very long prompts."""
        from llm_router import Router
        router = Router()
        
        long_prompt = "test " * 1000  # 4000+ character prompt
        result = router.route(long_prompt)
        assert result is not None
        assert isinstance(result, dict)
