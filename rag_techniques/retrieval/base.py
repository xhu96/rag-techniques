"""Base classes for retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RetrievalResult:
    """
    A result from retrieval.
    
    Attributes:
        text: The retrieved text content
        score: Relevance score (higher is more relevant)
        metadata: Additional metadata
        id: Optional document ID
    """
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    
    def __str__(self) -> str:
        return self.text


class Retriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of RetrievalResult objects
        """
        pass
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents with minimum score threshold.
        
        Args:
            query: User query
            top_k: Number of results to return
            min_score: Minimum score threshold
            
        Returns:
            Filtered list of RetrievalResult objects
        """
        results = self.retrieve(query, top_k=top_k)
        return [r for r in results if r.score >= min_score]
