"""In-memory vector store implementation."""

from typing import List, Dict, Any
import numpy as np

from rag_techniques.vectorstore.base import VectorStore, Document, SearchResult


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using numpy for similarity search.
    
    Best for small datasets and testing. Not persistent across restarts.
    """
    
    def __init__(self):
        """Initialize empty in-memory store."""
        self._documents: Dict[str, Document] = {}
        self._embeddings: np.ndarray | None = None
        self._ids: List[str] = []
    
    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to the store."""
        added_ids = []
        
        for doc in documents:
            self._documents[doc.id] = doc
            added_ids.append(doc.id)
        
        # Rebuild embedding matrix
        self._rebuild_index()
        
        return added_ids
    
    def _rebuild_index(self) -> None:
        """Rebuild the embedding matrix for efficient search."""
        if not self._documents:
            self._embeddings = None
            self._ids = []
            return
        
        self._ids = list(self._documents.keys())
        embeddings = [self._documents[id].embedding for id in self._ids]
        self._embeddings = np.array(embeddings)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings = self._embeddings / norms
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
    ) -> List[SearchResult]:
        """Search for similar documents using cosine similarity."""
        if self._embeddings is None or len(self._ids) == 0:
            return []
        
        # Normalize query
        query = np.array(query_embedding)
        query = query / np.linalg.norm(query)
        
        # Compute similarities
        similarities = np.dot(self._embeddings, query)
        
        # Apply metadata filter if provided
        if filter:
            mask = self._apply_filter(filter)
            similarities = similarities * mask
        
        # Get top-k indices
        k = min(top_k, len(self._ids))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            doc_id = self._ids[idx]
            score = float(similarities[idx])
            
            if score > 0:  # Filter out masked results
                results.append(SearchResult(
                    document=self._documents[doc_id],
                    score=score,
                ))
        
        return results
    
    def _apply_filter(self, filter: Dict[str, Any]) -> np.ndarray:
        """Create a mask for documents matching the filter."""
        mask = np.ones(len(self._ids), dtype=float)
        
        for i, doc_id in enumerate(self._ids):
            doc = self._documents[doc_id]
            for key, value in filter.items():
                if doc.metadata.get(key) != value:
                    mask[i] = 0
                    break
        
        return mask
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        for doc_id in ids:
            self._documents.pop(doc_id, None)
        
        self._rebuild_index()
    
    def get(self, ids: List[str]) -> List[Document]:
        """Get documents by ID."""
        return [self._documents[id] for id in ids if id in self._documents]
    
    @property
    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self._documents.clear()
        self._embeddings = None
        self._ids = []
