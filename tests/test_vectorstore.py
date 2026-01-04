"""Tests for vector store module."""

import pytest
import numpy as np
from rag_techniques.vectorstore.base import Document, SearchResult, VectorStore
from rag_techniques.vectorstore.memory import InMemoryVectorStore


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_document_creation(self):
        doc = Document(
            id="doc1",
            text="Hello world",
            embedding=[0.1, 0.2, 0.3],
        )
        assert doc.id == "doc1"
        assert doc.text == "Hello world"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {}
    
    def test_document_with_metadata(self):
        doc = Document(
            id="doc1",
            text="Test",
            embedding=[0.1],
            metadata={"source": "test.txt"},
        )
        assert doc.metadata["source"] == "test.txt"


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""
    
    @pytest.fixture
    def store(self):
        return InMemoryVectorStore()
    
    @pytest.fixture
    def sample_docs(self):
        return [
            Document(id="doc1", text="Hello world", embedding=[1.0, 0.0, 0.0]),
            Document(id="doc2", text="Goodbye world", embedding=[0.0, 1.0, 0.0]),
            Document(id="doc3", text="Hello again", embedding=[0.9, 0.1, 0.0]),
        ]
    
    def test_add_documents(self, store, sample_docs):
        ids = store.add(sample_docs)
        assert len(ids) == 3
        assert store.count == 3
    
    def test_search_basic(self, store, sample_docs):
        store.add(sample_docs)
        
        # Query similar to doc1 and doc3
        query = [1.0, 0.0, 0.0]
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be doc1 (exact match)
        assert results[0].document.id == "doc1"
    
    def test_search_top_k(self, store, sample_docs):
        store.add(sample_docs)
        
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
    
    def test_search_empty_store(self, store):
        results = store.search([1.0, 0.0, 0.0], top_k=5)
        assert results == []
    
    def test_delete(self, store, sample_docs):
        store.add(sample_docs)
        assert store.count == 3
        
        store.delete(["doc1"])
        assert store.count == 2
        
        results = store.get(["doc1"])
        assert results == []
    
    def test_get(self, store, sample_docs):
        store.add(sample_docs)
        
        docs = store.get(["doc1", "doc2"])
        assert len(docs) == 2
        assert {d.id for d in docs} == {"doc1", "doc2"}
    
    def test_clear(self, store, sample_docs):
        store.add(sample_docs)
        assert store.count == 3
        
        store.clear()
        assert store.count == 0
    
    def test_add_texts_helper(self, store):
        texts = ["Hello", "World"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        
        ids = store.add_texts(texts, embeddings)
        assert len(ids) == 2
        assert store.count == 2
    
    def test_metadata_filter(self, store):
        docs = [
            Document(id="d1", text="A", embedding=[1.0, 0.0], metadata={"type": "a"}),
            Document(id="d2", text="B", embedding=[0.9, 0.1], metadata={"type": "b"}),
            Document(id="d3", text="C", embedding=[0.8, 0.2], metadata={"type": "a"}),
        ]
        store.add(docs)
        
        # Filter to only type "a"
        results = store.search([1.0, 0.0], top_k=3, filter={"type": "a"})
        
        assert len(results) == 2
        assert all(r.document.metadata["type"] == "a" for r in results)
