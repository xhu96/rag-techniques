"""Base classes for reranking."""

from abc import ABC, abstractmethod
from typing import List

from rag_techniques.retrieval.base import RetrievalResult


class Reranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results.
        
        Args:
            query: User query
            results: Initial retrieval results
            top_k: Number of results to return (all if None)
            
        Returns:
            Reranked list of RetrievalResult objects
        """
        pass
