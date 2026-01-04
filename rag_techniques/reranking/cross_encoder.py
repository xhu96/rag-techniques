"""
Cross-Encoder reranking.

Uses BERT-based Cross-Encoder models (via sentence-transformers) to
compute precise relevance scores between query and document pairs.
"""

from typing import List

from rag_techniques.reranking.base import Reranker
from rag_techniques.retrieval.base import RetrievalResult
from rag_techniques.config import get_settings


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using sentence-transformers.
    
    Cross-encoders process query and document together, providing
    more accurate relevance scores than bi-encoder similarity.
    
    Reference: https://www.sbert.net/examples/applications/cross-encoder/README.html
    """
    
    def __init__(self, model_name: str | None = None):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name (default from settings)
        """
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self._model = None  # Lazy load
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: User query
            results: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Reranked list of RetrievalResult objects
        """
        if not results:
            return []
        
        model = self._load_model()
        
        # Create query-document pairs
        pairs = [[query, result.text] for result in results]
        
        # Get cross-encoder scores
        scores = model.predict(pairs)
        
        # Create scored results
        scored_results = [
            RetrievalResult(
                text=result.text,
                score=float(score),
                metadata={**result.metadata, "original_score": result.score},
                id=result.id,
            )
            for result, score in zip(results, scores)
        ]
        
        # Sort by new scores
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k
        if top_k:
            return scored_results[:top_k]
        return scored_results
