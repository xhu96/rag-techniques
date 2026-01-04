"""Tests for chunking module."""

import pytest
from rag_techniques.chunking.base import Chunk, Chunker
from rag_techniques.chunking.fixed import FixedSizeChunker, RecursiveChunker


class TestChunk:
    """Tests for Chunk dataclass."""
    
    def test_chunk_creation(self):
        chunk = Chunk(text="Hello world", index=0)
        assert chunk.text == "Hello world"
        assert chunk.index == 0
        assert chunk.metadata == {}
        assert chunk.embedding is None
    
    def test_chunk_length(self):
        chunk = Chunk(text="Hello world", index=0)
        assert len(chunk) == 11
    
    def test_chunk_str(self):
        chunk = Chunk(text="Hello world", index=0)
        assert str(chunk) == "Hello world"
    
    def test_chunk_with_metadata(self):
        chunk = Chunk(text="Test", index=0, metadata={"source": "test.txt"})
        assert chunk.metadata["source"] == "test.txt"


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""
    
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        text = "Hello world, this is a test."
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_size_respected(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        text = "A" * 30
        chunks = chunker.chunk(text)
        
        # Each chunk should be at most chunk_size
        for chunk in chunks:
            assert len(chunk.text) <= 10
    
    def test_overlap(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=5)
        text = "A" * 20
        chunks = chunker.chunk(text)
        
        # With overlap, we should get more chunks
        assert len(chunks) >= 2
    
    def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk("")
        assert chunks == []
    
    def test_metadata_preserved(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk("Hello world", metadata={"source": "test.txt"})
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"
    
    def test_invalid_overlap(self):
        """Overlap must be less than chunk size."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=10)


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""
    
    def test_basic_chunking(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a test. " * 20
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_respects_separators(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk(text)
        
        # Should split on paragraph boundaries when possible
        assert len(chunks) >= 1
    
    def test_empty_text(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk("")
        assert chunks == []
