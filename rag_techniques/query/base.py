"""Base classes for query transformation."""

from abc import ABC, abstractmethod
from typing import List


class QueryTransformer(ABC):
    """Abstract base class for query transformers."""
    
    @abstractmethod
    def transform(self, query: str) -> str | List[str]:
        """
        Transform a query.
        
        Args:
            query: Original user query
            
        Returns:
            Transformed query string or list of query strings
        """
        pass
