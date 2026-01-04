"""
ChromaDB vector store backend.

Provides persistent storage, metadata filtering, and optimized HNSW search.
"""

from typing import List, Dict, Any
import uuid

from rag_techniques.vectorstore.base import VectorStore, Document, SearchResult
from rag_techniques.config import get_settings


class ChromaVectorStore(VectorStore):
    """
    Persistent vector store using ChromaDB.
    
    Provides durable storage with automatic embedding management.
    Supports metadata filtering and efficient similarity search.
    
    Example:
        ```python
        store = ChromaVectorStore(collection_name="my_docs")
        store.add(documents)
        results = store.search(query_embedding, top_k=5)
        ```
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str | None = None,
        distance_fn: str = "cosine",
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence (default from settings)
            distance_fn: Distance function ("cosine", "l2", "ip")
        """
        # Lazy import to avoid loading if not used
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        settings = get_settings()
        persist_dir = persist_directory or settings.chroma_persist_directory
        
        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_fn},
        )
        
        self._collection_name = collection_name
    
    def add(self, documents: List[Document]) -> List[str]:
        """Add documents to the collection."""
        if not documents:
            return []
        
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        where = filter if filter else None
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to SearchResult objects
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                # For cosine distance, similarity = 1 - distance
                similarity = 1.0 - distance
                
                doc = Document(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    embedding=[],  # ChromaDB doesn't return embeddings by default
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                
                search_results.append(SearchResult(document=doc, score=similarity))
        
        return search_results
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self._collection.delete(ids=ids)
    
    def get(self, ids: List[str]) -> List[Document]:
        """Get documents by ID."""
        results = self._collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"],
        )
        
        documents = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                doc = Document(
                    id=doc_id,
                    text=results["documents"][i] if results["documents"] else "",
                    embedding=results["embeddings"][i] if results["embeddings"] else [],
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
                documents.append(doc)
        
        return documents
    
    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def persist(self) -> None:
        """
        Persist changes to disk.
        
        Note: With PersistentClient, changes are automatically persisted.
        This method is kept for API compatibility.
        """
        pass  # PersistentClient auto-persists
