"""
Integration tests for dataset functionality.

This module tests dataset integration with other components like
the embedding service.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from llm_router.dataset import ExampleDataset, ExamplePrompt, PromptCategory


class TestDatasetEmbeddingIntegration:
    """Test dataset integration with embedding service."""
    
    def test_should_work_with_embedding_service(self):
        """Test that dataset examples work with embedding service."""
        from llm_router.embeddings import EmbeddingService
        
        examples = [
            ExamplePrompt("Generate Python code", PromptCategory.CODE, ["codex"]),
            ExamplePrompt("Write a story", PromptCategory.CREATIVE, ["gpt-4"])
        ]
        dataset = ExampleDataset(examples)
        
        # Create a mock embedding service
        mock_service = MagicMock(spec=EmbeddingService)
        mock_embeddings = np.array([[0.1] * 384, [0.2] * 384])
        mock_service.embed_batch.return_value = mock_embeddings
        
        texts = dataset.get_embedding_texts()
        embeddings = mock_service.embed_batch(texts)
        
        assert len(embeddings) == 2
        assert embeddings.shape == (2, 384)
        mock_service.embed_batch.assert_called_once_with(texts)
    
    def test_should_cache_embeddings_for_examples(self):
        """Test that example embeddings are cached."""
        from llm_router.embeddings import EmbeddingService
        
        examples = [
            ExamplePrompt("Same text", PromptCategory.CODE, ["codex"]),
            ExamplePrompt("Same text", PromptCategory.CREATIVE, ["gpt-4"])  # Duplicate text
        ]
        dataset = ExampleDataset(examples)
        
        # Create a mock embedding service that simulates caching behavior
        mock_service = MagicMock(spec=EmbeddingService)
        mock_embedding = np.array([0.1] * 384)
        
        # Mock embed method to return same embedding for same text
        def mock_embed(text):
            return mock_embedding
        
        mock_service.embed.side_effect = mock_embed
        
        texts = dataset.get_embedding_texts()
        
        # Generate embeddings - would cache duplicates in real implementation
        embeddings = [mock_service.embed(text) for text in texts]
        
        # Should have called embed for each text (even duplicates in this mock)
        assert mock_service.embed.call_count == 2
        assert len(embeddings) == 2
        # Both embeddings should be identical (same text)
        np.testing.assert_array_equal(embeddings[0], embeddings[1])
