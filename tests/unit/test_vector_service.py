"""
Unit tests for vector service using Pinecone.

This module tests the main vector service that uses Pinecone
for production vector similarity search.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import numpy as np

from llm_router.vector_service import VectorService, create_vector_service
from llm_router.dataset import ExamplePrompt, PromptCategory


class TestVectorService:
    """Test vector service functionality."""
    
    def test_should_require_pinecone_api_key(self):
        """Test that API key is required."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                VectorService()
            assert "API key required" in str(exc_info.value)
    
    def test_should_use_environment_variables(self):
        """Test that service uses environment variables for config."""
        with patch.dict(os.environ, {
            "PINECONE_API_KEY": "test-key",
            "PINECONE_ENVIRONMENT": "test-env"
        }):
            with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
                with patch('llm_router.vector_service.EmbeddingService'):
                    mock_store = MagicMock()
                    mock_factory.return_value = mock_store
                    
                    service = VectorService()
                    
                    # Should use env vars
                    mock_factory.assert_called_once()
                    call_kwargs = mock_factory.call_args[1]
                    assert call_kwargs['api_key'] == "test-key"
                    assert call_kwargs['environment'] == "test-env"
    
    def test_should_add_example_with_metadata(self):
        """Test adding an example to the vector service."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
            with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
                with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                    
                    # Setup mocks
                    mock_store = MagicMock()
                    mock_factory.return_value = mock_store
                    
                    mock_embedder = MagicMock()
                    mock_embedder.embed.return_value = np.array([0.1, 0.2, 0.3])
                    mock_embedding_service.return_value = mock_embedder
                    
                    service = VectorService()
                    
                    # Create example
                    example = ExamplePrompt(
                        text="Write a Python function",
                        category=PromptCategory.CODE,
                        preferred_models=["codex", "gpt-4"]
                    )
                    
                    # Add example
                    example_id = service.add_example(example)
                    
                    # Verify embedding was generated
                    mock_embedder.embed.assert_called_once_with("Write a Python function")
                    
                    # Verify vector was added to store
                    mock_store.add_vector.assert_called_once()
                    call_args = mock_store.add_vector.call_args
                    
                    assert call_args[0][0] == example_id  # ID
                    np.testing.assert_array_equal(call_args[0][1], np.array([0.1, 0.2, 0.3]))  # Vector
                    
                    # Check metadata
                    metadata = call_args[0][2]
                    assert metadata["text"] == "Write a Python function"
                    assert metadata["category"] == "code"
                    assert metadata["preferred_models"] == ["codex", "gpt-4"]
    
    def test_should_find_similar_examples(self):
        """Test finding similar examples."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
            with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
                with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                    
                    # Setup mocks
                    mock_store = MagicMock()
                    mock_store.search.return_value = [
                        MagicMock(similarity=0.9, metadata={"text": "Similar example"})
                    ]
                    mock_factory.return_value = mock_store
                    
                    mock_embedder = MagicMock()
                    mock_embedder.embed.return_value = np.array([0.1, 0.2, 0.3])
                    mock_embedding_service.return_value = mock_embedder
                    
                    service = VectorService()
                    
                    # Find similar examples
                    results = service.find_similar_examples("Test query", k=3)
                    
                    # Verify embedding was generated for query
                    mock_embedder.embed.assert_called_with("Test query")
                    
                    # Verify search was called
                    mock_store.search.assert_called_once()
                    search_args = mock_store.search.call_args[1]
                    assert search_args['k'] == 3
                    np.testing.assert_array_equal(search_args['query_vector'], np.array([0.1, 0.2, 0.3]))
                    
                    # Verify results
                    assert len(results) == 1
                    assert results[0].similarity == 0.9
    
    def test_should_get_routing_recommendation(self):
        """Test getting routing recommendations."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
            with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
                with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                    
                    # Setup mocks
                    mock_store = MagicMock()
                    mock_store.search.return_value = [
                        MagicMock(
                            similarity=0.9, 
                            metadata={
                                "text": "Code example",
                                "category": "code_generation",
                                "preferred_models": ["codex", "gpt-4"]
                            }
                        ),
                        MagicMock(
                            similarity=0.7, 
                            metadata={
                                "text": "Another code example",
                                "category": "code_generation", 
                                "preferred_models": ["gpt-4"]
                            }
                        )
                    ]
                    mock_store.calculate_confidence.return_value = 0.85
                    mock_factory.return_value = mock_store
                    
                    mock_embedder = MagicMock()
                    mock_embedder.embed.return_value = np.array([0.1, 0.2, 0.3])
                    mock_embedding_service.return_value = mock_embedder
                    
                    service = VectorService()
                    
                    # Get routing recommendation
                    recommendation = service.get_routing_recommendation("Write Python code")
                    
                    # Verify recommendation structure
                    assert "recommended_models" in recommendation
                    assert "confidence" in recommendation
                    assert "reasoning" in recommendation
                    assert "similar_examples" in recommendation
                    
                    # Verify model recommendations (gpt-4 should be top due to higher combined votes)
                    assert "gpt-4" in recommendation["recommended_models"]
                    assert "codex" in recommendation["recommended_models"]
                    
                    # Verify confidence
                    assert recommendation["confidence"] == 0.85
                    
                    # Verify similar examples
                    assert len(recommendation["similar_examples"]) == 2
    
    def test_should_handle_no_similar_examples(self):
        """Test handling when no similar examples are found."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
            with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
                with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                    
                    # Setup mocks
                    mock_store = MagicMock()
                    mock_store.search.return_value = []  # No results
                    mock_factory.return_value = mock_store
                    
                    mock_embedder = MagicMock()
                    mock_embedding_service.return_value = mock_embedder
                    
                    service = VectorService()
                    
                    # Get routing recommendation
                    recommendation = service.get_routing_recommendation("Unknown query")
                    
                    # Verify fallback response
                    assert recommendation["recommended_models"] == []
                    assert recommendation["confidence"] == 0.0
                    assert "No similar examples found" in recommendation["reasoning"]
                    assert recommendation["similar_examples"] == []
    
    def test_should_create_service_via_factory(self):
        """Test creating service via factory function."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
            with patch('llm_router.vector_service.VectorStoreFactory.create_store'):
                with patch('llm_router.vector_service.EmbeddingService'):
                    
                    service = create_vector_service(index_name="test-index")
                    
                    assert isinstance(service, VectorService)
                    assert service.index_name == "test-index"
