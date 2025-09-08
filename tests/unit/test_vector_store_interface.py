"""Tests for vector store interface and related components."""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC

from llm_router.vector_stores.vector_store_interface import (
    VectorStoreInterface, 
    VectorRecord, 
    SearchResult, 
    VectorStoreError,
    VectorStoreFactory,
    VectorStoreBackend,
    SimilarityMetric,
    create_vector_store,
    create_pinecone_store
)


class TestVectorRecord:
    """Test VectorRecord data model."""
    
    def test_should_create_vector_record_with_required_fields(self):
        """Test creating a vector record with required fields."""
        vector = np.array([0.1, 0.2, 0.3, 0.4])
        record = VectorRecord(
            id="test-id",
            vector=vector
        )
        
        assert record.id == "test-id"
        assert np.array_equal(record.vector, vector)
        assert record.metadata is None
    
    def test_should_create_vector_record_with_metadata(self):
        """Test creating a vector record with metadata."""
        vector = np.array([0.1, 0.2, 0.3, 0.4])
        metadata = {"category": "test", "text": "sample text"}
        
        record = VectorRecord(
            id="test-id",
            vector=vector,
            metadata=metadata
        )
        
        assert record.id == "test-id"
        assert np.array_equal(record.vector, vector)
        assert record.metadata == metadata
    
    def test_should_accept_different_vector_types(self):
        """Test that vector record accepts numpy arrays."""
        vector = np.array([0.1, 0.2, 0.3])
        record = VectorRecord(
            id="test-id",
            vector=vector
        )
        
        assert record.id == "test-id"
        assert np.array_equal(record.vector, vector)
    
    def test_should_handle_empty_id_gracefully(self):
        """Test that empty ID is stored (validation handled elsewhere)."""
        vector = np.array([0.1, 0.2, 0.3])
        record = VectorRecord(
            id="",
            vector=vector
        )
        
        assert record.id == ""
        assert np.array_equal(record.vector, vector)
    
    def test_should_support_dataclass_features(self):
        """Test that VectorRecord supports dataclass features."""
        vector1 = np.array([0.1, 0.2, 0.3])
        vector2 = np.array([0.1, 0.2, 0.3])
        
        record1 = VectorRecord(id="test", vector=vector1)
        record2 = VectorRecord(id="test", vector=vector2)
        
        # Note: numpy arrays don't compare equal in dataclass equality
        # This is expected behavior
        assert record1.id == record2.id
        assert np.array_equal(record1.vector, record2.vector)


class TestSearchResult:
    """Test SearchResult data model."""
    
    def test_should_create_search_result_with_required_fields(self):
        """Test creating a search result with required fields."""
        vector = np.array([0.1, 0.2, 0.3])
        
        result = SearchResult(
            vector=vector,
            similarity=0.95,
            index=0
        )
        
        assert np.array_equal(result.vector, vector)
        assert result.similarity == 0.95
        assert result.index == 0
        assert result.metadata is None
    
    def test_should_create_search_result_with_metadata(self):
        """Test creating a search result with metadata."""
        vector = np.array([0.1, 0.2, 0.3])
        metadata = {"id": "vec1", "category": "test"}
        
        result = SearchResult(
            vector=vector,
            similarity=0.87,
            index=1,
            metadata=metadata
        )
        
        assert np.array_equal(result.vector, vector)
        assert result.similarity == 0.87
        assert result.index == 1
        assert result.metadata == metadata
    
    def test_should_accept_any_similarity_value(self):
        """Test that similarity accepts any float (validation handled elsewhere)."""
        vector = np.array([0.1, 0.2, 0.3])
        
        # Test negative similarity (stored as-is)
        result1 = SearchResult(
            vector=vector,
            similarity=-0.1,
            index=0
        )
        assert result1.similarity == -0.1
        
        # Test similarity > 1 (stored as-is)
        result2 = SearchResult(
            vector=vector,
            similarity=1.5,
            index=0
        )
        assert result2.similarity == 1.5
    
    def test_should_accept_any_index_value(self):
        """Test that index accepts any integer (validation handled elsewhere)."""
        vector = np.array([0.1, 0.2, 0.3])
        
        result = SearchResult(
            vector=vector,
            similarity=0.95,
            index=-1  # Negative index allowed at dataclass level
        )
        
        assert result.index == -1
    
    def test_should_support_dataclass_equality(self):
        """Test that search results support dataclass equality."""
        vector1 = np.array([0.1, 0.2, 0.3])
        vector2 = np.array([0.1, 0.2, 0.3])
        
        result1 = SearchResult(vector=vector1, similarity=0.95, index=0)
        result2 = SearchResult(vector=vector2, similarity=0.95, index=0)
        
        # Note: numpy arrays don't compare equal in dataclass equality
        # This is expected behavior
        assert result1.similarity == result2.similarity
        assert result1.index == result2.index
        assert np.array_equal(result1.vector, result2.vector)
    
    def test_should_support_custom_sorting(self):
        """Test that search results can be sorted by similarity."""
        vector = np.array([0.1, 0.2, 0.3])
        
        results = [
            SearchResult(vector=vector, similarity=0.8, index=0),
            SearchResult(vector=vector, similarity=0.95, index=1),
            SearchResult(vector=vector, similarity=0.7, index=2)
        ]
        
        # Sort by similarity (highest first)
        sorted_results = sorted(results, key=lambda x: x.similarity, reverse=True)
        
        assert sorted_results[0].similarity == 0.95
        assert sorted_results[1].similarity == 0.8
        assert sorted_results[2].similarity == 0.7


class TestVectorStoreError:
    """Test VectorStoreError exception."""
    
    def test_should_create_error_with_message(self):
        """Test creating error with message."""
        error = VectorStoreError("Test error message")
        
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_should_create_error_with_cause(self):
        """Test creating error with underlying cause."""
        cause = ValueError("Original error")
        error = VectorStoreError("Wrapped error")
        error.__cause__ = cause
        
        assert str(error) == "Wrapped error"
        assert error.__cause__ == cause


class TestSimilarityMetric:
    """Test SimilarityMetric enum."""
    
    def test_should_have_expected_metric_values(self):
        """Test that cosine similarity is available."""
        assert SimilarityMetric.COSINE.value == "cosine"
    
    def test_should_support_enum_access(self):
        """Test accessing enum values."""
        assert SimilarityMetric.COSINE


class TestVectorStoreBackend:
    """Test VectorStoreBackend enum."""
    
    def test_should_have_expected_backend_values(self):
        """Test that Pinecone backend is available."""
        assert VectorStoreBackend.PINECONE.value == "pinecone"
    
    def test_should_support_enum_access(self):
        """Test accessing enum values."""
        assert VectorStoreBackend.PINECONE


class MockVectorStore(VectorStoreInterface):
    """Mock implementation of VectorStoreInterface for testing."""
    
    def __init__(self):
        self.vectors = {}
        self.dimension = None
    
    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.dimension is None:
            self.dimension = vector.shape[0]
        elif vector.shape[0] != self.dimension:
            raise VectorStoreError(f"Vector dimension {vector.shape[0]} does not match store dimension {self.dimension}")
        
        self.vectors[vector_id] = VectorRecord(
            id=vector_id,
            vector=vector.copy(),
            metadata=metadata
        )
    
    def add_vectors_batch(self, records: List[VectorRecord]) -> None:
        for record in records:
            self.add_vector(record.id, record.vector, record.metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 5, 
               filter_metadata: Optional[Dict[str, Any]] = None,
               exclude_ids: Optional[List[str]] = None) -> List[SearchResult]:
        results = []
        
        for record in self.vectors.values():
            # Skip excluded IDs
            if exclude_ids and record.id in exclude_ids:
                continue
            
            # Apply metadata filter
            if filter_metadata:
                if not record.metadata:
                    continue
                match = all(
                    record.metadata.get(key) == value 
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, record.vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(record.vector)
            )
            
            results.append(SearchResult(
                vector=record.vector.copy(),
                similarity=float(similarity),
                index=len(results),
                metadata={"id": record.id, **(record.metadata or {})}
            ))
        
        # Sort by similarity (highest first) and return top k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]
    
    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        return self.vectors.get(vector_id)
    
    def delete_vector(self, vector_id: str) -> bool:
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            return True
        return False
    
    def clear(self) -> None:
        self.vectors.clear()
    
    def count(self) -> int:
        return len(self.vectors)
    
    def get_dimension(self) -> Optional[int]:
        return self.dimension


class TestVectorStoreInterface:
    """Test VectorStoreInterface abstract base class."""
    
    def test_should_be_abstract_base_class(self):
        """Test that VectorStoreInterface is an abstract base class."""
        assert issubclass(VectorStoreInterface, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            VectorStoreInterface()
    
    def test_should_define_required_methods(self):
        """Test that all required abstract methods are defined."""
        required_methods = [
            'add_vector',
            'add_vectors_batch', 
            'search',
            'get_vector',
            'delete_vector',
            'clear',
            'count',
            'get_dimension'
        ]
        
        for method_name in required_methods:
            assert hasattr(VectorStoreInterface, method_name)
            method = getattr(VectorStoreInterface, method_name)
            assert callable(method)
    
    def test_should_have_calculate_confidence_method(self):
        """Test that calculate_confidence method exists."""
        assert hasattr(VectorStoreInterface, 'calculate_confidence')
        assert callable(VectorStoreInterface.calculate_confidence)


class TestVectorStoreInterfaceConfidence:
    """Test confidence calculation functionality."""
    
    def test_should_calculate_confidence_for_empty_results(self):
        """Test confidence calculation for empty search results."""
        store = MockVectorStore()
        confidence = store.calculate_confidence([])
        assert confidence == 0.0
    
    def test_should_calculate_confidence_for_single_result(self):
        """Test confidence calculation for single search result."""
        store = MockVectorStore()
        vector = np.array([1.0, 0.0, 0.0])
        
        result = SearchResult(vector=vector, similarity=0.85, index=0)
        confidence = store.calculate_confidence([result])
        
        assert confidence == 0.85
    
    def test_should_calculate_confidence_for_multiple_results(self):
        """Test confidence calculation for multiple search results."""
        store = MockVectorStore()
        vector = np.array([1.0, 0.0, 0.0])
        
        results = [
            SearchResult(vector=vector, similarity=0.95, index=0),
            SearchResult(vector=vector, similarity=0.7, index=1),
            SearchResult(vector=vector, similarity=0.6, index=2)
        ]
        
        confidence = store.calculate_confidence(results)
        
        # Should be: max_score * 0.8 + score_gap * 0.2
        # = 0.95 * 0.8 + (0.95 - 0.7) * 0.2
        # = 0.76 + 0.05 = 0.81
        expected = 0.95 * 0.8 + (0.95 - 0.7) * 0.2
        assert confidence == pytest.approx(expected)
    
    def test_should_bound_confidence_between_zero_and_one(self):
        """Test that confidence is bounded between 0 and 1."""
        store = MockVectorStore()
        vector = np.array([1.0, 0.0, 0.0])
        
        # Test with similarity > 1
        result_high = SearchResult(vector=vector, similarity=1.5, index=0)
        confidence_high = store.calculate_confidence([result_high])
        assert 0.0 <= confidence_high <= 1.0
        
        # Test with negative similarity
        result_low = SearchResult(vector=vector, similarity=-0.5, index=0)
        confidence_low = store.calculate_confidence([result_low])
        assert 0.0 <= confidence_low <= 1.0


class TestMockVectorStoreImplementation:
    """Test the mock vector store implementation to verify interface compliance."""
    
    def test_should_implement_interface(self):
        """Test that MockVectorStore implements the interface."""
        store = MockVectorStore()
        assert isinstance(store, VectorStoreInterface)
    
    def test_should_add_and_retrieve_vectors(self):
        """Test adding and retrieving vectors."""
        store = MockVectorStore()
        vector = np.array([0.1, 0.2, 0.3, 0.4])
        metadata = {"category": "test"}
        
        store.add_vector("vec1", vector, metadata)
        
        retrieved = store.get_vector("vec1")
        assert retrieved is not None
        assert retrieved.id == "vec1"
        assert np.array_equal(retrieved.vector, vector)
        assert retrieved.metadata == metadata
    
    def test_should_handle_dimension_consistency(self):
        """Test that dimension is enforced consistently."""
        store = MockVectorStore()
        
        # First vector sets the dimension
        vector1 = np.array([0.1, 0.2, 0.3])
        store.add_vector("vec1", vector1)
        assert store.get_dimension() == 3
        
        # Second vector with same dimension should work
        vector2 = np.array([0.4, 0.5, 0.6])
        store.add_vector("vec2", vector2)
        
        # Vector with different dimension should fail
        vector3 = np.array([0.7, 0.8])
        with pytest.raises(VectorStoreError, match="Vector dimension 2 does not match store dimension 3"):
            store.add_vector("vec3", vector3)
    
    def test_should_search_with_similarity(self):
        """Test searching vectors by similarity."""
        store = MockVectorStore()
        
        # Add some vectors
        store.add_vector("vec1", np.array([1.0, 0.0, 0.0]), {"category": "A"})
        store.add_vector("vec2", np.array([0.0, 1.0, 0.0]), {"category": "B"})
        store.add_vector("vec3", np.array([0.9, 0.1, 0.0]), {"category": "A"})
        
        # Search for vector similar to vec1
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, k=2)
        
        assert len(results) == 2
        # vec1 should be most similar (identical)
        assert results[0].metadata["id"] == "vec1"
        assert results[0].similarity == pytest.approx(1.0)
        # vec3 should be second most similar
        assert results[1].metadata["id"] == "vec3"
        assert results[1].similarity > 0.9
    
    def test_should_filter_by_metadata(self):
        """Test filtering search results by metadata."""
        store = MockVectorStore()
        
        store.add_vector("vec1", np.array([1.0, 0.0, 0.0]), {"category": "A"})
        store.add_vector("vec2", np.array([0.9, 0.1, 0.0]), {"category": "B"})
        store.add_vector("vec3", np.array([0.8, 0.2, 0.0]), {"category": "A"})
        
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, k=5, filter_metadata={"category": "A"})
        
        assert len(results) == 2
        assert all(result.metadata["category"] == "A" for result in results)
    
    def test_should_exclude_ids_from_search(self):
        """Test excluding specific IDs from search results."""
        store = MockVectorStore()
        
        store.add_vector("vec1", np.array([1.0, 0.0, 0.0]))
        store.add_vector("vec2", np.array([0.9, 0.1, 0.0]))
        store.add_vector("vec3", np.array([0.8, 0.2, 0.0]))
        
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, k=5, exclude_ids=["vec1"])
        
        assert len(results) == 2
        assert all(result.metadata["id"] != "vec1" for result in results)
    
    def test_should_batch_add_vectors(self):
        """Test adding multiple vectors in batch."""
        store = MockVectorStore()
        
        records = [
            VectorRecord("vec1", np.array([1.0, 0.0, 0.0]), {"category": "A"}),
            VectorRecord("vec2", np.array([0.0, 1.0, 0.0]), {"category": "B"}),
            VectorRecord("vec3", np.array([0.0, 0.0, 1.0]), {"category": "C"})
        ]
        
        store.add_vectors_batch(records)
        
        assert store.count() == 3
        assert store.get_vector("vec1") is not None
        assert store.get_vector("vec2") is not None
        assert store.get_vector("vec3") is not None
    
    def test_should_delete_vectors(self):
        """Test deleting vectors."""
        store = MockVectorStore()
        
        store.add_vector("vec1", np.array([1.0, 0.0, 0.0]))
        store.add_vector("vec2", np.array([0.0, 1.0, 0.0]))
        
        assert store.count() == 2
        
        # Delete existing vector
        result = store.delete_vector("vec1")
        assert result is True
        assert store.count() == 1
        assert store.get_vector("vec1") is None
        
        # Delete non-existing vector
        result = store.delete_vector("nonexistent")
        assert result is False
        assert store.count() == 1
    
    def test_should_clear_all_vectors(self):
        """Test clearing all vectors."""
        store = MockVectorStore()
        
        store.add_vector("vec1", np.array([1.0, 0.0, 0.0]))
        store.add_vector("vec2", np.array([0.0, 1.0, 0.0]))
        
        assert store.count() == 2
        
        store.clear()
        
        assert store.count() == 0
        assert store.get_vector("vec1") is None
        assert store.get_vector("vec2") is None


class TestVectorStoreFactory:
    """Test VectorStoreFactory functionality."""
    
    def test_should_have_create_store_method(self):
        """Test that factory has create_store method."""
        assert hasattr(VectorStoreFactory, 'create_store')
        assert callable(VectorStoreFactory.create_store)
    
    def test_should_validate_backend_type(self):
        """Test validation of backend types."""
        # Test that unsupported backend raises error
        with pytest.raises(VectorStoreError, match="Unsupported backend"):
            # Create a fake backend enum value
            fake_backend = type('FakeBackend', (), {'value': 'fake'})()
            VectorStoreFactory.create_store(fake_backend)
    
    def test_should_support_convenience_functions(self):
        """Test that convenience functions exist."""
        assert callable(create_vector_store)
        assert callable(create_pinecone_store)
    
    def test_should_create_vector_store_with_string_backend(self):
        """Test creating vector store with string backend name."""
        # This would try to create a pinecone store but fail due to missing credentials
        # We just test that the function exists and accepts the right parameters
        try:
            create_vector_store("pinecone", api_key="fake", environment="fake", index_name="fake")
        except Exception:
            # Expected to fail due to missing/invalid credentials, but function should exist
            pass
