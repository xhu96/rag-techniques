"""Base classes for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Document:
    """
    A document stored in a vector store.
    
    Attributes:
        id: Unique identifier
        text: Document text content
        embedding: Vector embedding
        metadata: Additional metadata
    """
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    A search result from a vector store.
    
    Attributes:
        document: The matched document
        score: Similarity score (higher is more similar)
    """
    document: Document
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete
        """
        pass
    
    @abstractmethod
    def get(self, ids: List[str]) -> List[Document]:
        """
        Get documents by ID.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        pass
    
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] | None = None,
        ids: List[str] | None = None,
    ) -> List[str]:
        """
        Convenience method to add texts with embeddings.
        
        Args:
            texts: List of text strings
            embeddings: Corresponding embeddings
            metadatas: Optional list of metadata dicts
            ids: Optional list of IDs (auto-generated if not provided)
            
        Returns:
            List of document IDs
        """
        import uuid
        
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        
        documents = [
            Document(
                id=doc_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
            )
            for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
        ]
        
        return self.add(documents)
    
    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        pass
