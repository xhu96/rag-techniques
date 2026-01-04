"""Hybrid retrieval combining sparse and dense search."""

from typing import List, Dict, Any
from collections import defaultdict

from rank_bm25 import BM25Okapi

from rag_techniques.retrieval.base import Retriever, RetrievalResult
from rag_techniques.vectorstore.base import VectorStore
from rag_techniques.core.embeddings import EmbeddingProvider
from rag_techniques.config import get_settings


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining BM25 (sparse) and vector (dense) search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
    
    Reference: "Blended RAG" (2024) - https://arxiv.org/abs/2404.07220
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        documents: List[Dict[str, Any]] | None = None,
        alpha: float | None = None,
        top_k: int | None = None,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for dense retrieval
            embedding_provider: Provider for embedding queries
            documents: Optional list of documents with 'text' and 'id' keys for BM25
            alpha: Weight for dense retrieval (1-alpha for sparse). Default from settings.
            top_k: Default number of results
            rrf_k: RRF parameter (default 60)
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        settings = get_settings()
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.default_top_k = top_k or settings.top_k
        self.rrf_k = rrf_k
        
        # Initialize BM25 index
        self._documents: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        
        if documents:
            self.index_documents(documents)
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents with 'text' and optionally 'id' keys
        """
        self._documents = documents
        
        # Tokenize documents for BM25
        tokenized_docs = [
            self._tokenize(doc.get("text", ""))
            for doc in documents
        ]
        
        self._bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: User query
            top_k: Number of results to return
            filter: Optional metadata filter (applied to dense results only)
            
        Returns:
            List of RetrievalResult objects sorted by combined score
        """
        k = top_k or self.default_top_k
        
        # Get dense results
        dense_results = self._dense_search(query, k * 2, filter)
        
        # Get sparse results
        sparse_results = self._sparse_search(query, k * 2)
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k,
        )
        
        return combined
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        filter: Dict[str, Any] | None,
    ) -> List[RetrievalResult]:
        """Perform dense (vector) search."""
        query_embedding = self.embedding_provider.embed_query(query)
        
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter,
        )
        
        return [
            RetrievalResult(
                text=result.document.text,
                score=result.score,
                metadata=result.document.metadata,
                id=result.document.id,
            )
            for result in search_results
        ]
    
    def _sparse_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform sparse (BM25) search."""
        if self._bm25 is None or not self._documents:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._documents[idx]
                results.append(RetrievalResult(
                    text=doc.get("text", ""),
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    id=doc.get("id", str(idx)),
                ))
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) across all result lists
        """
        # Build score map by document ID/text
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, RetrievalResult] = {}
        
        # Add dense scores with alpha weight
        for rank, result in enumerate(dense_results):
            key = result.id or result.text[:100]
            rrf_score = self.alpha * (1.0 / (self.rrf_k + rank + 1))
            scores[key] += rrf_score
            doc_map[key] = result
        
        # Add sparse scores with (1-alpha) weight
        for rank, result in enumerate(sparse_results):
            key = result.id or result.text[:100]
            rrf_score = (1 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))
            scores[key] += rrf_score
            if key not in doc_map:
                doc_map[key] = result
        
        # Sort by combined score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        # Build final results
        results = []
        for key in sorted_keys[:top_k]:
            result = doc_map[key]
            results.append(RetrievalResult(
                text=result.text,
                score=scores[key],
                metadata=result.metadata,
                id=result.id,
            ))
        
        return results
