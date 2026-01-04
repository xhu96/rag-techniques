"""Vector-based semantic retrieval."""

from typing import List, Dict, Any

from rag_techniques.retrieval.base import Retriever, RetrievalResult
from rag_techniques.vectorstore.base import VectorStore
from rag_techniques.core.embeddings import EmbeddingProvider
from rag_techniques.config import get_settings


class VectorRetriever(Retriever):
    """
    Simple vector-based semantic retriever.
    
    Uses embedding similarity to find relevant documents.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        top_k: int | None = None,
    ):
        """
        Initialize vector retriever.
        
        Args:
            vector_store: Vector store containing documents
            embedding_provider: Provider for embedding queries
            top_k: Default number of results (from settings if not specified)
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.default_top_k = top_k or get_settings().top_k
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using vector similarity search.
        
        Args:
            query: User query
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        k = top_k or self.default_top_k
        
        # Embed the query
        query_embedding = self.embedding_provider.embed_query(query)
        
        # Search the vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filter=filter,
        )
        
        # Convert to RetrievalResult
        return [
            RetrievalResult(
                text=result.document.text,
                score=result.score,
                metadata=result.document.metadata,
                id=result.document.id,
            )
            for result in search_results
        ]
