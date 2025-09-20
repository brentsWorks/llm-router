"""
Pinecone vector store implementation.

This module provides a Pinecone-backed implementation of the VectorStoreInterface
for production-scale vector similarity search.

Installation required: pip install pinecone
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .vector_store_interface import VectorStoreInterface, VectorRecord, SearchResult, VectorStoreError

logger = logging.getLogger(__name__)

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone client not installed. Run: pip install pinecone")


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone-backed vector storage and similarity search."""
    
    def __init__(self, 
                 api_key: str,
                 environment: str,
                 index_name: str,
                 dimension: int = 384,
                 metric: str = "cosine",
                 create_index: bool = True,
                 **kwargs):
        """Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-east1-gcp')
            index_name: Name of the Pinecone index
            dimension: Vector dimension
            metric: Similarity metric ('cosine', 'euclidean', 'dotproduct')
            create_index: Whether to create index if it doesn't exist
            **kwargs: Additional Pinecone configuration
        """
        if not PINECONE_AVAILABLE:
            raise VectorStoreError("Pinecone client not installed. Run: pip install pinecone")
        
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone with new API
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=api_key)
        
        # Create or connect to index
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if create_index and index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec={"serverless": {"cloud": "aws", "region": environment}},
                **kwargs
            )
        
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a vector to the Pinecone index."""
        if not isinstance(vector, np.ndarray):
            raise VectorStoreError("Vector must be a numpy array")
        
        if vector.shape[0] != self.dimension:
            raise VectorStoreError(
                f"Vector dimension {vector.shape[0]} does not match index dimension {self.dimension}"
            )
        
        # Convert numpy array to list for Pinecone
        vector_list = vector.tolist()
        
        # Upsert to Pinecone
        self.index.upsert(vectors=[(vector_id, vector_list, metadata or {})])
        logger.debug(f"Added vector to Pinecone with ID: {vector_id}")
    
    def add_vectors_batch(self, records: List[VectorRecord]) -> None:
        """Add multiple vectors in a batch operation."""
        # Prepare batch data for Pinecone
        vectors_to_upsert = []
        for record in records:
            if record.vector.shape[0] != self.dimension:
                raise VectorStoreError(
                    f"Vector dimension {record.vector.shape[0]} does not match index dimension {self.dimension}"
                )
            
            vectors_to_upsert.append((
                record.id,
                record.vector.tolist(),
                record.metadata or {}
            ))
        
        # Batch upsert to Pinecone (Pinecone handles batching internally)
        self.index.upsert(vectors=vectors_to_upsert)
        logger.info(f"Added {len(vectors_to_upsert)} vectors to Pinecone in batch")
    
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5, 
               filter_metadata: Optional[Dict[str, Any]] = None,
               exclude_ids: Optional[List[str]] = None) -> List[SearchResult]:
        """Search for k most similar vectors in Pinecone."""
        if not isinstance(query_vector, np.ndarray):
            raise VectorStoreError("Query vector must be a numpy array")
        
        if query_vector.shape[0] != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension {query_vector.shape[0]} does not match index dimension {self.dimension}"
            )
        
        # Convert numpy array to list for Pinecone
        query_list = query_vector.tolist()
        
        # Build Pinecone query
        query_params = {
            "vector": query_list,
            "top_k": k,
            "include_metadata": True,
            "include_values": True  # Include vector values in response
        }
        
        # Add metadata filter if provided
        if filter_metadata:
            query_params["filter"] = self._build_pinecone_filter(filter_metadata)
        
        # Execute query
        response = self.index.query(**query_params)
        
        # Convert Pinecone response to SearchResult objects
        results = []
        for i, match in enumerate(response.matches):
            # Filter out excluded IDs
            if exclude_ids and match.id in exclude_ids:
                continue
            
            # Convert vector back to numpy array
            vector_array = np.array(match.values) if match.values else query_vector
            
            result = SearchResult(
                vector=vector_array,
                similarity=match.score,
                index=i,
                metadata={"id": match.id, **(match.metadata or {})}
            )
            results.append(result)
        
        # Apply exclusion filter and re-limit to k
        if exclude_ids:
            results = results[:k]
        
        logger.debug(f"Found {len(results)} results from Pinecone for k={k} search")
        return results
    
    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Get a vector by ID from Pinecone."""
        try:
            response = self.index.fetch(ids=[vector_id])
            if vector_id not in response.vectors:
                return None
            
            vector_data = response.vectors[vector_id]
            return VectorRecord(
                id=vector_id,
                vector=np.array(vector_data.values),
                metadata=vector_data.metadata
            )
        except Exception as e:
            logger.error(f"Error fetching vector {vector_id} from Pinecone: {e}")
            return None
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID from Pinecone."""
        try:
            self.index.delete(ids=[vector_id])
            logger.debug(f"Deleted vector from Pinecone with ID: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector {vector_id} from Pinecone: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all vectors from the Pinecone index."""
        try:
            # Delete all vectors (Pinecone doesn't have a direct "clear all" method)
            self.index.delete(delete_all=True)
            logger.info("Cleared all vectors from Pinecone index")
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {e}")
            raise VectorStoreError(f"Failed to clear Pinecone index: {e}")
    
    def count(self) -> int:
        """Get the number of vectors in the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            logger.error(f"Error getting vector count from Pinecone: {e}")
            return 0
    
    def get_dimension(self) -> Optional[int]:
        """Get the vector dimension."""
        return self.dimension
    
    def _build_pinecone_filter(self, filter_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build Pinecone-compatible metadata filter."""
        # Pinecone uses a specific filter format
        # This is a simplified implementation - you may need to adjust based on your metadata structure
        pinecone_filter = {}
        
        for key, value in filter_metadata.items():
            if isinstance(value, list):
                # Use $in operator for list values
                pinecone_filter[key] = {"$in": value}
            else:
                # Direct equality for single values
                pinecone_filter[key] = {"$eq": value}
        
        return pinecone_filter


# Convenience function
def create_pinecone_store(api_key: str, 
                         environment: str,
                         index_name: str,
                         dimension: int = 384,
                         **kwargs) -> PineconeVectorStore:
    """Create a Pinecone vector store."""
    return PineconeVectorStore(
        api_key=api_key,
        environment=environment,
        index_name=index_name,
        dimension=dimension,
        **kwargs
    )
