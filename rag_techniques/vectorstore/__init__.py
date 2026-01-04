"""Vector store module for storing and retrieving embeddings."""

from rag_techniques.vectorstore.base import VectorStore, Document
from rag_techniques.vectorstore.memory import InMemoryVectorStore

# Optional: ChromaDB (requires chromadb package)
try:
    from rag_techniques.vectorstore.chroma import ChromaVectorStore
except ImportError:
    ChromaVectorStore = None  # type: ignore

__all__ = [
    "VectorStore",
    "Document",
    "InMemoryVectorStore",
    "ChromaVectorStore",
]
