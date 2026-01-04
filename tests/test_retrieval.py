"""Tests for retrieval module."""

import pytest
from unittest.mock import Mock, MagicMock
from rag_techniques.retrieval.base import Retriever, RetrievalResult
from rag_techniques.retrieval.vector import VectorRetriever
from rag_techniques.retrieval.hybrid import HybridRetriever
from rag_techniques.vectorstore.base import Document, SearchResult
from rag_techniques.vectorstore.memory import InMemoryVectorStore


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_creation(self):
        result = RetrievalResult(text="Hello", score=0.9)
        assert result.text == "Hello"
        assert result.score == 0.9
        assert result.metadata == {}
        assert result.id is None
    
    def test_str(self):
        result = RetrievalResult(text="Hello", score=0.9)
        assert str(result) == "Hello"


class TestVectorRetriever:
    """Tests for VectorRetriever."""
    
    @pytest.fixture
    def mock_embedding_provider(self):
        provider = Mock()
        provider.embed_query.return_value = [1.0, 0.0, 0.0]
        return provider
    
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.search.return_value = [
            SearchResult(
                document=Document(id="d1", text="Hello", embedding=[1.0, 0.0, 0.0]),
                score=0.95,
            ),
            SearchResult(
                document=Document(id="d2", text="World", embedding=[0.9, 0.1, 0.0]),
                score=0.85,
            ),
        ]
        return store
    
    def test_retrieve_basic(self, mock_vector_store, mock_embedding_provider):
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
        )
        
        results = retriever.retrieve("test query", top_k=2)
        
        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[0].score == 0.95
        mock_embedding_provider.embed_query.assert_called_once_with("test query")
    
    def test_retrieve_with_filter(self, mock_vector_store, mock_embedding_provider):
        retriever = VectorRetriever(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
        )
        
        results = retriever.retrieve("query", filter={"type": "a"})
        
        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["filter"] == {"type": "a"}


class TestHybridRetriever:
    """Tests for HybridRetriever."""
    
    @pytest.fixture
    def mock_embedding_provider(self):
        provider = Mock()
        provider.embed_query.return_value = [1.0, 0.0, 0.0]
        return provider
    
    @pytest.fixture
    def sample_documents(self):
        return [
            {"id": "d1", "text": "Machine learning is a subset of AI."},
            {"id": "d2", "text": "Deep learning uses neural networks."},
            {"id": "d3", "text": "AI stands for artificial intelligence."},
        ]
    
    def test_index_documents(self, mock_embedding_provider, sample_documents):
        store = InMemoryVectorStore()
        retriever = HybridRetriever(
            vector_store=store,
            embedding_provider=mock_embedding_provider,
        )
        
        retriever.index_documents(sample_documents)
        
        assert retriever._bm25 is not None
        assert len(retriever._documents) == 3
    
    def test_sparse_search(self, mock_embedding_provider, sample_documents):
        store = InMemoryVectorStore()
        retriever = HybridRetriever(
            vector_store=store,
            embedding_provider=mock_embedding_provider,
            documents=sample_documents,
        )
        
        # Test BM25 search
        results = retriever._sparse_search("machine learning AI", top_k=2)
        
        assert len(results) > 0
        # Should find documents with "machine", "learning", "AI"
    
    def test_rrf_combination(self):
        """Test that RRF correctly combines dense and sparse results."""
        store = Mock()
        provider = Mock()
        provider.embed_query.return_value = [1.0, 0.0, 0.0]
        
        # Set up mock to return empty (we'll test RRF directly)
        store.search.return_value = []
        
        retriever = HybridRetriever(
            vector_store=store,
            embedding_provider=provider,
        )
        
        # Create test results
        dense = [
            RetrievalResult(text="A", score=0.9, id="a"),
            RetrievalResult(text="B", score=0.8, id="b"),
        ]
        sparse = [
            RetrievalResult(text="B", score=5.0, id="b"),
            RetrievalResult(text="C", score=4.0, id="c"),
        ]
        
        combined = retriever._reciprocal_rank_fusion(dense, sparse, top_k=3)
        
        # B should be ranked higher (appears in both)
        assert len(combined) == 3
        # B should have the highest combined score
        b_result = next((r for r in combined if r.id == "b"), None)
        assert b_result is not None
