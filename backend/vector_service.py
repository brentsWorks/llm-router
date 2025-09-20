"""
Vector service for LLM Router using Pinecone.

This module provides the main vector service interface for the LLM Router,
using Pinecone as the production vector store backend.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from .vector_stores import VectorStoreInterface, VectorStoreFactory, VectorStoreBackend, SearchResult
from .embeddings import EmbeddingService
from .dataset import ExampleDataset, ExamplePrompt

logger = logging.getLogger(__name__)


class VectorService:
    """Main vector service using Pinecone for production deployment."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 environment: Optional[str] = None,
                 index_name: str = "llm-router",
                 dimension: int = 384):
        """Initialize vector service with Pinecone.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
            index_name: Name of the Pinecone index
            dimension: Vector dimension
        """
        # Get credentials from environment if not provided
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        self.index_name = index_name
        self.dimension = dimension
        
        if not self.api_key:
            raise ValueError(
                "Pinecone API key required. Set PINECONE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize vector store
        self.vector_store = VectorStoreFactory.create_store(
            backend=VectorStoreBackend.PINECONE,
            api_key=self.api_key,
            environment=self.environment,
            index_name=self.index_name,
            dimension=self.dimension
        )
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(use_cache=True)
        
        logger.info(f"Vector service initialized with Pinecone index: {self.index_name}")
    
    def add_example(self, example: ExamplePrompt, example_id: Optional[str] = None) -> str:
        """Add an example prompt to the vector store.
        
        Args:
            example: Example prompt to add
            example_id: Optional custom ID (auto-generated if not provided)
            
        Returns:
            The ID of the added example
        """
        # Generate ID if not provided
        if example_id is None:
            example_id = f"example_{hash(example.text) % 1000000}"
        
        # Generate embedding
        embedding = self.embedding_service.embed(example.text)
        
        # Prepare metadata
        metadata = {
            "text": example.text,
            "category": example.category.value,
            "preferred_models": example.preferred_models,
            "description": example.description,
            "difficulty": example.difficulty,
            "expected_length": example.expected_length,
            "domain": example.domain,
            "tags": example.tags
        }
        
        # Add to vector store
        self.vector_store.add_vector(example_id, embedding, metadata)
        
        logger.debug(f"Added example to vector store: {example_id}")
        return example_id
    
    def add_dataset(self, dataset: ExampleDataset) -> List[str]:
        """Add all examples from a dataset to the vector store.
        
        Args:
            dataset: Dataset containing examples to add
            
        Returns:
            List of IDs of added examples
        """
        logger.info(f"Adding {len(dataset)} examples to vector store")
        
        example_ids = []
        for i, example in enumerate(dataset):
            example_id = f"dataset_example_{i}"
            added_id = self.add_example(example, example_id)
            example_ids.append(added_id)
        
        logger.info(f"Successfully added {len(example_ids)} examples to vector store")
        return example_ids
    
    def find_similar_examples(self, 
                            query_text: str, 
                            k: int = 5,
                            category_filter: Optional[str] = None,
                            model_filter: Optional[List[str]] = None) -> List[SearchResult]:
        """Find similar examples for a query text.
        
        Args:
            query_text: Text to find similar examples for
            k: Number of similar examples to return
            category_filter: Optional category to filter by
            model_filter: Optional list of preferred models to filter by
            
        Returns:
            List of similar examples with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query_text)
        
        # Build metadata filters
        filter_metadata = {}
        if category_filter:
            filter_metadata["category"] = category_filter
        if model_filter:
            filter_metadata["preferred_models"] = model_filter
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            k=k,
            filter_metadata=filter_metadata if filter_metadata else None
        )
        
        logger.debug(f"Found {len(results)} similar examples for query: {query_text[:50]}...")
        return results
    
    def get_routing_recommendation(self, 
                                 query_text: str, 
                                 k: int = 3) -> Dict[str, Any]:
        """Get routing recommendation based on similar examples.
        
        Args:
            query_text: Text to route
            k: Number of similar examples to consider
            
        Returns:
            Dictionary with routing recommendation and confidence
        """
        # Find similar examples
        similar_examples = self.find_similar_examples(query_text, k=k)
        
        if not similar_examples:
            return {
                "recommended_models": [],
                "confidence": 0.0,
                "reasoning": "No similar examples found",
                "similar_examples": []
            }
        
        # Calculate confidence
        confidence = self.vector_store.calculate_confidence(similar_examples)
        
        # Aggregate preferred models from similar examples
        model_votes = {}
        for result in similar_examples:
            preferred_models = result.metadata.get("preferred_models", [])
            weight = result.similarity  # Weight by similarity
            
            for model in preferred_models:
                model_votes[model] = model_votes.get(model, 0) + weight
        
        # Sort models by votes
        recommended_models = sorted(model_votes.items(), key=lambda x: x[1], reverse=True)
        recommended_models = [model for model, votes in recommended_models]
        
        return {
            "recommended_models": recommended_models,
            "confidence": confidence,
            "reasoning": f"Based on {len(similar_examples)} similar examples",
            "similar_examples": [
                {
                    "text": result.metadata.get("text", ""),
                    "category": result.metadata.get("category", ""),
                    "similarity": result.similarity,
                    "preferred_models": result.metadata.get("preferred_models", [])
                }
                for result in similar_examples
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "total_examples": self.vector_store.count(),
            "dimension": self.vector_store.get_dimension(),
            "index_name": self.index_name,
            "environment": self.environment,
            "cache_enabled": self.embedding_service.use_cache
        }


# Convenience function for easy initialization
def create_vector_service(api_key: Optional[str] = None,
                         environment: Optional[str] = None,
                         index_name: str = "llm-router") -> VectorService:
    """Create a vector service instance.
    
    Args:
        api_key: Pinecone API key (defaults to env var)
        environment: Pinecone environment (defaults to env var)
        index_name: Pinecone index name
        
    Returns:
        Configured VectorService instance
    """
    return VectorService(
        api_key=api_key,
        environment=environment,
        index_name=index_name
    )
