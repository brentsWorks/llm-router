"""
Integration tests for VectorService with Pinecone backend.

These tests verify that VectorService integrates properly with:
- PineconeVectorStore backend
- EmbeddingService for vector generation
- ExampleDataset for bulk operations
- Search and recommendation functionality
"""

import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from llm_router.vector_service import VectorService, create_vector_service
from llm_router.dataset import ExamplePrompt, PromptCategory, ExampleDataset
from llm_router.vector_stores import SearchResult
from llm_router.embeddings import EmbeddingService


@pytest.fixture
def mock_pinecone_environment():
    """Setup mock environment variables for Pinecone."""
    with patch.dict(os.environ, {
        "PINECONE_API_KEY": "test-api-key",
        "PINECONE_ENVIRONMENT": "test-env"
    }):
        yield


class TestVectorServicePineconeIntegration:
    """Test VectorService integration with Pinecone backend."""

    @pytest.fixture
    def mock_pinecone_store(self):
        """Create a mock Pinecone vector store."""
        mock_store = MagicMock()
        mock_store.add_vector.return_value = None
        mock_store.count.return_value = 0
        mock_store.get_dimension.return_value = 384
        mock_store.calculate_confidence.return_value = 0.8
        return mock_store

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        mock_service = MagicMock(spec=EmbeddingService)
        mock_service.use_cache = True
        mock_service.embed.return_value = np.random.random(384)
        mock_service.embed_batch.return_value = np.random.random((3, 384))
        return mock_service

    @pytest.fixture
    def sample_examples(self):
        """Create sample examples for testing."""
        return [
            ExamplePrompt(
                text="Write a Python function to calculate factorial",
                category=PromptCategory.CODE_GENERATION,
                preferred_models=["codex", "gpt-4"],
                description="Math function example",
                difficulty="medium",
                expected_length="short",
                domain="mathematics",
                tags=["python", "function", "math"]
            ),
            ExamplePrompt(
                text="Write a creative story about space exploration",
                category=PromptCategory.CREATIVE_WRITING,
                preferred_models=["gpt-4", "claude"],
                description="Creative writing example",
                difficulty="easy",
                expected_length="long",
                domain="fiction",
                tags=["story", "space", "creative"]
            ),
            ExamplePrompt(
                text="Explain quantum computing principles",
                category=PromptCategory.QUESTION_ANSWERING,
                preferred_models=["gpt-4", "claude"],
                description="Technical explanation",
                difficulty="hard",
                expected_length="medium",
                domain="physics",
                tags=["quantum", "computing", "science"]
            )
        ]

    def test_vector_service_initializes_with_pinecone_backend(self, mock_pinecone_environment):
        """Test that VectorService properly initializes with Pinecone backend."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                mock_store = MagicMock()
                mock_factory.return_value = mock_store
                mock_embedding_service.return_value = MagicMock()

                service = VectorService(
                    api_key="test-key",
                    environment="test-env",
                    index_name="test-index",
                    dimension=384
                )

                # Verify factory was called with correct parameters
                mock_factory.assert_called_once()
                call_kwargs = mock_factory.call_args[1]
                assert call_kwargs['api_key'] == "test-key"
                assert call_kwargs['environment'] == "test-env"
                assert call_kwargs['index_name'] == "test-index"
                assert call_kwargs['dimension'] == 384

                # Verify embedding service was initialized
                mock_embedding_service.assert_called_once_with(use_cache=True)

    def test_add_example_integrates_embedding_and_storage(self, mock_pinecone_environment, sample_examples):
        """Test that adding an example integrates embedding generation and vector storage."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                mock_store = MagicMock()
                mock_factory.return_value = mock_store
                
                mock_embedder = MagicMock()
                test_embedding = np.array([0.1] * 384)
                mock_embedder.embed.return_value = test_embedding
                mock_embedding_service.return_value = mock_embedder

                service = VectorService()
                example = sample_examples[0]

                # Add example
                example_id = service.add_example(example, "test_id")

                # Verify embedding was generated
                mock_embedder.embed.assert_called_once_with(example.text)

                # Verify vector was stored with correct metadata
                mock_store.add_vector.assert_called_once()
                call_args = mock_store.add_vector.call_args[0]
                
                assert call_args[0] == "test_id"  # ID
                np.testing.assert_array_equal(call_args[1], test_embedding)  # Vector
                
                # Check metadata structure
                metadata = call_args[2]
                assert metadata["text"] == example.text
                assert metadata["category"] == example.category.value
                assert metadata["preferred_models"] == example.preferred_models
                assert metadata["description"] == example.description
                assert metadata["difficulty"] == example.difficulty
                assert metadata["expected_length"] == example.expected_length
                assert metadata["domain"] == example.domain
                assert metadata["tags"] == example.tags

    def test_add_dataset_bulk_operation_integration(self, mock_pinecone_environment, sample_examples):
        """Test that adding a dataset integrates bulk operations correctly."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                mock_store = MagicMock()
                mock_factory.return_value = mock_store
                
                mock_embedder = MagicMock()
                mock_embedder.embed.side_effect = [np.array([0.1] * 384), np.array([0.2] * 384), np.array([0.3] * 384)]
                mock_embedding_service.return_value = mock_embedder

                service = VectorService()
                dataset = ExampleDataset(sample_examples)

                # Add dataset
                example_ids = service.add_dataset(dataset)

                # Verify all examples were processed
                assert len(example_ids) == 3
                assert mock_embedder.embed.call_count == 3
                assert mock_store.add_vector.call_count == 3

                # Verify IDs were generated correctly
                expected_ids = ["dataset_example_0", "dataset_example_1", "dataset_example_2"]
                assert example_ids == expected_ids

    def test_find_similar_examples_integration(self, mock_pinecone_environment, sample_examples):
        """Test that finding similar examples integrates search and filtering."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                # Setup mock search results
                mock_results = [
                    SearchResult(
                        vector=np.array([0.1] * 384),
                        similarity=0.9,
                        index=0,
                        metadata={
                            "id": "example_1",
                            "text": "Similar code example",
                            "category": "code_generation",
                            "preferred_models": ["codex"]
                        }
                    ),
                    SearchResult(
                        vector=np.array([0.2] * 384),
                        similarity=0.7,
                        index=1,
                        metadata={
                            "id": "example_2",
                            "text": "Another code example",
                            "category": "code_generation",
                            "preferred_models": ["gpt-4"]
                        }
                    )
                ]
                
                mock_store = MagicMock()
                mock_store.search.return_value = mock_results
                mock_factory.return_value = mock_store
                
                mock_embedder = MagicMock()
                query_embedding = np.array([0.5] * 384)
                mock_embedder.embed.return_value = query_embedding
                mock_embedding_service.return_value = mock_embedder

                service = VectorService()

                # Find similar examples with filters
                results = service.find_similar_examples(
                    query_text="Write a function",
                    k=5,
                    category_filter="code_generation",
                    model_filter=["codex", "gpt-4"]
                )

                # Verify embedding was generated for query
                mock_embedder.embed.assert_called_once_with("Write a function")

                # Verify search was called with correct parameters
                mock_store.search.assert_called_once()
                search_kwargs = mock_store.search.call_args[1]
                np.testing.assert_array_equal(search_kwargs['query_vector'], query_embedding)
                assert search_kwargs['k'] == 5
                assert search_kwargs['filter_metadata']['category'] == "code_generation"
                assert search_kwargs['filter_metadata']['preferred_models'] == ["codex", "gpt-4"]

                # Verify results
                assert len(results) == 2
                assert results[0].similarity == 0.9
                assert results[1].similarity == 0.7

    def test_routing_recommendation_integration(self, mock_pinecone_environment):
        """Test that routing recommendation integrates search and confidence calculation."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                # Setup mock search results with model preferences
                mock_results = [
                    SearchResult(
                        vector=np.array([0.1] * 384),
                        similarity=0.9,
                        index=0,
                        metadata={
                            "text": "Code example 1",
                            "category": "code_generation",
                            "preferred_models": ["codex", "gpt-4"]
                        }
                    ),
                    SearchResult(
                        vector=np.array([0.2] * 384),
                        similarity=0.7,
                        index=1,
                        metadata={
                            "text": "Code example 2", 
                            "category": "code_generation",
                            "preferred_models": ["gpt-4", "claude"]
                        }
                    ),
                    SearchResult(
                        vector=np.array([0.3] * 384),
                        similarity=0.6,
                        index=2,
                        metadata={
                            "text": "Code example 3",
                            "category": "code_generation",
                            "preferred_models": ["codex"]
                        }
                    )
                ]
                
                mock_store = MagicMock()
                mock_store.search.return_value = mock_results
                mock_store.calculate_confidence.return_value = 0.85
                mock_factory.return_value = mock_store
                
                mock_embedder = MagicMock()
                mock_embedder.embed.return_value = np.array([0.5] * 384)
                mock_embedding_service.return_value = mock_embedder

                service = VectorService()

                # Get routing recommendation
                recommendation = service.get_routing_recommendation("Write Python code", k=3)

                # Verify search was called
                mock_store.search.assert_called_once()
                
                # Verify confidence calculation was called
                mock_store.calculate_confidence.assert_called_once_with(mock_results)

                # Verify recommendation structure
                assert "recommended_models" in recommendation
                assert "confidence" in recommendation
                assert "reasoning" in recommendation
                assert "similar_examples" in recommendation

                # Verify model recommendations (weighted by similarity)
                # gpt-4: 0.9 + 0.7 = 1.6, codex: 0.9 + 0.6 = 1.5, claude: 0.7
                expected_order = ["gpt-4", "codex", "claude"]
                assert recommendation["recommended_models"] == expected_order
                assert recommendation["confidence"] == 0.85
                assert "Based on 3 similar examples" in recommendation["reasoning"]

                # Verify similar examples structure
                assert len(recommendation["similar_examples"]) == 3
                for i, example in enumerate(recommendation["similar_examples"]):
                    assert "text" in example
                    assert "category" in example
                    assert "similarity" in example
                    assert "preferred_models" in example
                    assert example["similarity"] == mock_results[i].similarity

    def test_empty_results_handling_integration(self, mock_pinecone_environment):
        """Test integration when no similar examples are found."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                mock_store = MagicMock()
                mock_store.search.return_value = []  # No results
                mock_factory.return_value = mock_store
                
                mock_embedder = MagicMock()
                mock_embedder.embed.return_value = np.array([0.5] * 384)
                mock_embedding_service.return_value = mock_embedder

                service = VectorService()

                # Get routing recommendation with no similar examples
                recommendation = service.get_routing_recommendation("Unknown query")

                # Verify fallback behavior
                assert recommendation["recommended_models"] == []
                assert recommendation["confidence"] == 0.0
                assert "No similar examples found" in recommendation["reasoning"]
                assert recommendation["similar_examples"] == []

    def test_service_statistics_integration(self, mock_pinecone_environment):
        """Test that service statistics integrate with backend store."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                mock_store = MagicMock()
                mock_store.count.return_value = 42
                mock_store.get_dimension.return_value = 384
                mock_factory.return_value = mock_store
                
                mock_embedder = MagicMock()
                mock_embedder.use_cache = True
                mock_embedding_service.return_value = mock_embedder

                service = VectorService(
                    api_key="test-key",
                    environment="test-env",
                    index_name="test-index"
                )

                # Get statistics
                stats = service.get_stats()

                # Verify statistics structure and values
                assert stats["total_examples"] == 42
                assert stats["dimension"] == 384
                assert stats["index_name"] == "test-index"
                assert stats["environment"] == "test-env"
                assert stats["cache_enabled"] is True

                # Verify backend methods were called
                mock_store.count.assert_called_once()
                mock_store.get_dimension.assert_called_once()

    def test_factory_function_integration(self, mock_pinecone_environment):
        """Test that factory function creates properly integrated service."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                mock_store = MagicMock()
                mock_factory.return_value = mock_store
                mock_embedding_service.return_value = MagicMock()

                # Create service via factory
                service = create_vector_service(
                    api_key="factory-key",
                    environment="factory-env",
                    index_name="factory-index"
                )

                # Verify service was created with correct configuration
                assert isinstance(service, VectorService)
                assert service.api_key == "factory-key"
                assert service.environment == "factory-env"
                assert service.index_name == "factory-index"

                # Verify backend was initialized
                mock_factory.assert_called_once()
                call_kwargs = mock_factory.call_args[1]
                assert call_kwargs['api_key'] == "factory-key"
                assert call_kwargs['environment'] == "factory-env"
                assert call_kwargs['index_name'] == "factory-index"


class TestVectorServiceErrorHandlingIntegration:
    """Test VectorService error handling integration."""

    def test_missing_api_key_error_integration(self):
        """Test that missing API key is properly handled."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                VectorService()
            
            assert "API key required" in str(exc_info.value)
            assert "PINECONE_API_KEY" in str(exc_info.value)

    def test_backend_initialization_error_integration(self, mock_pinecone_environment):
        """Test error handling when backend initialization fails."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            mock_factory.side_effect = Exception("Pinecone connection failed")
            
            with pytest.raises(Exception) as exc_info:
                VectorService()
            
            assert "Pinecone connection failed" in str(exc_info.value)

    def test_embedding_service_error_integration(self, mock_pinecone_environment):
        """Test error handling when embedding service fails."""
        with patch('llm_router.vector_service.VectorStoreFactory.create_store') as mock_factory:
            with patch('llm_router.vector_service.EmbeddingService') as mock_embedding_service:
                
                mock_store = MagicMock()
                mock_factory.return_value = mock_store
                
                mock_embedder = MagicMock()
                mock_embedder.embed.side_effect = Exception("Embedding generation failed")
                mock_embedding_service.return_value = mock_embedder

                service = VectorService()
                example = ExamplePrompt(
                    text="Test prompt",
                    category=PromptCategory.CODE_GENERATION,
                    preferred_models=["gpt-4"]
                )

                # Should propagate embedding error
                with pytest.raises(Exception) as exc_info:
                    service.add_example(example)
                
                assert "Embedding generation failed" in str(exc_info.value)
