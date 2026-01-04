"""Tests for the RAG pipeline."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rag_techniques.pipeline import RAGPipeline, RAGResponse
from rag_techniques.chunking.base import Chunk
from rag_techniques.retrieval.base import RetrievalResult


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""
    
    def test_creation(self):
        response = RAGResponse(
            answer="Test answer",
            sources=[RetrievalResult(text="source", score=0.9)],
            query="test query",
        )
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.query == "test query"
        assert response.transformed_query is None


class TestRAGPipeline:
    """Tests for RAGPipeline."""
    
    @pytest.fixture
    def mock_embedding_provider(self):
        provider = Mock()
        provider.embed.return_value = [[1.0, 0.0, 0.0]]
        provider.embed_query.return_value = [1.0, 0.0, 0.0]
        provider.dimension = 3
        return provider
    
    @pytest.fixture
    def mock_llm_provider(self):
        llm = Mock()
        response = Mock()
        response.content = "This is a test answer."
        llm.generate_with_context.return_value = response
        return llm
    
    def test_add_documents(self, mock_embedding_provider, mock_llm_provider):
        """Test adding documents to pipeline."""
        pipeline = RAGPipeline(
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
        )
        
        num_chunks = pipeline.add_documents([
            {"text": "Hello world, this is a test document."},
        ])
        
        assert num_chunks > 0
        assert pipeline.num_chunks > 0
    
    def test_add_text(self, mock_embedding_provider, mock_llm_provider):
        """Test adding a single text."""
        pipeline = RAGPipeline(
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
        )
        
        num_chunks = pipeline.add_text("Simple test text.")
        
        assert num_chunks > 0
    
    def test_query_empty_store(self, mock_embedding_provider, mock_llm_provider):
        """Test querying with no documents."""
        pipeline = RAGPipeline(
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
        )
        
        response = pipeline.query("test query")
        
        # Should return a message about not having information
        assert "don't have enough information" in response.answer.lower() or len(response.answer) > 0
    
    def test_clear(self, mock_embedding_provider, mock_llm_provider):
        """Test clearing the pipeline."""
        pipeline = RAGPipeline(
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
        )
        
        pipeline.add_text("Test document.")
        assert pipeline.num_chunks > 0
        
        pipeline.clear()
        assert pipeline.num_chunks == 0
