"""
Retrieval algorithms and strategies.

Components:
- VectorRetriever: Standard dense vector similarity search
- HybridRetriever: Ensemble of BM25 (sparse) and Vector (dense) with RRF
- ContextEnrichedRetriever: Retrieves window of surrounding context
""" for searching and ranking documents."""

from rag_techniques.retrieval.base import Retriever, RetrievalResult
from rag_techniques.retrieval.vector import VectorRetriever
from rag_techniques.retrieval.hybrid import HybridRetriever
from rag_techniques.retrieval.context_enriched import ContextEnrichedRetriever

__all__ = [
    "Retriever",
    "RetrievalResult",
    "VectorRetriever",
    "HybridRetriever",
    "ContextEnrichedRetriever",
]
