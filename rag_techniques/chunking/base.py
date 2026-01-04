"""Base classes for text chunking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Chunk:
    """
    A chunk of text with metadata.
    
    Attributes:
        text: The chunk text content
        index: Position of chunk in the source document
        metadata: Additional metadata (source, page, etc.)
        embedding: Optional pre-computed embedding
    """
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] | None = None
    
    def __len__(self) -> int:
        """Return the length of the chunk text."""
        return len(self.text)
    
    def __str__(self) -> str:
        """Return the chunk text."""
        return self.text


class Chunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] | None = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dicts
            text_key: Key containing the text in each document
            
        Returns:
            List of Chunk objects from all documents
        """
        all_chunks = []
        chunk_offset = 0
        
        for doc in documents:
            text = doc.get(text_key, "")
            metadata = {k: v for k, v in doc.items() if k != text_key}
            chunks = self.chunk(text, metadata)
            
            # Update indices to be globally unique
            for chunk in chunks:
                chunk.index += chunk_offset
            
            all_chunks.extend(chunks)
            chunk_offset += len(chunks)
        
        return all_chunks
