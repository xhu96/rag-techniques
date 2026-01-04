"""
Vector storage backends.

Provides a common interface for vector databases:
- InMemoryVectorStore: Simple, transient storage using NumPy
- ChromaVectorStore: Persistent, production-ready storage using ChromaDB
"""

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
