"""
Context-enriched retrieval.

Technique to retrieve the most relevant chunk but return it along with
its surrounding neighbors (pre/post context) for better LLM grounding.
"""

from typing import List, Dict, Any

from rag_techniques.retrieval.base import Retriever, RetrievalResult
from rag_techniques.retrieval.vector import VectorRetriever
from rag_techniques.vectorstore.base import VectorStore
from rag_techniques.core.embeddings import EmbeddingProvider


class ContextEnrichedRetriever(Retriever):
    """
    Context-enriched retriever that includes neighboring chunks.
    
    When a chunk is retrieved, this retriever also returns the chunks
    immediately before and after it, providing additional context.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        chunks: List[Dict[str, Any]],
        context_size: int = 1,
        top_k: int = 3,
    ):
        """
        Initialize context-enriched retriever.
        
        Args:
            vector_store: Vector store containing document chunks
            embedding_provider: Provider for embedding queries
            chunks: Ordered list of all chunks with 'text' and 'index' keys
            context_size: Number of neighboring chunks to include on each side
            top_k: Default number of results
        """
        self.vector_retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            top_k=top_k,
        )
        self.chunks = chunks
        self.context_size = context_size
        self.default_top_k = top_k
        
        # Build index for fast lookup
        self._chunk_by_index: Dict[int, Dict[str, Any]] = {
            chunk.get("index", i): chunk
            for i, chunk in enumerate(chunks)
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents with surrounding context.
        
        Args:
            query: User query
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of RetrievalResult objects with enriched context
        """
        k = top_k or self.default_top_k
        
        # Get initial results
        initial_results = self.vector_retriever.retrieve(query, top_k=k, filter=filter)
        
        # Enrich each result with context
        enriched_results = []
        seen_indices = set()
        
        for result in initial_results:
            # Get the chunk index from metadata
            chunk_index = result.metadata.get("index")
            if chunk_index is None:
                # Try to find by matching text
                chunk_index = self._find_chunk_index(result.text)
            
            if chunk_index is None:
                # Can't find context, use original
                enriched_results.append(result)
                continue
            
            # Skip if we've already processed this context window
            if chunk_index in seen_indices:
                continue
            
            # Get context window
            context_text, context_indices = self._get_context_window(chunk_index)
            seen_indices.update(context_indices)
            
            enriched_results.append(RetrievalResult(
                text=context_text,
                score=result.score,
                metadata={
                    **result.metadata,
                    "center_index": chunk_index,
                    "context_indices": list(context_indices),
                    "context_size": self.context_size,
                },
                id=result.id,
            ))
        
        return enriched_results
    
    def _find_chunk_index(self, text: str) -> int | None:
        """Find chunk index by matching text."""
        for chunk in self.chunks:
            if chunk.get("text", "") == text:
                return chunk.get("index")
        return None
    
    def _get_context_window(self, center_index: int) -> tuple[str, set[int]]:
        """
        Get text from a context window around the center chunk.
        
        Args:
            center_index: Index of the center chunk
            
        Returns:
            Tuple of (combined text, set of indices included)
        """
        min_index = max(0, center_index - self.context_size)
        max_index = min(len(self.chunks) - 1, center_index + self.context_size)
        
        context_parts = []
        indices = set()
        
        for idx in range(min_index, max_index + 1):
            if idx in self._chunk_by_index:
                chunk = self._chunk_by_index[idx]
                context_parts.append(chunk.get("text", ""))
                indices.add(idx)
        
        return " ".join(context_parts), indices
