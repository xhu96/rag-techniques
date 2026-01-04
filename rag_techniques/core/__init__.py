"""
Core components for the RAG library.

Defines the abstract base classes and concrete implementations for:
- EmbeddingProvider: Wrapper for embedding models (OpenAI, SentenceTransformers)
- LLMProvider: Wrapper for Language Models (OpenAI, etc.)
"""

from rag_techniques.core.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from rag_techniques.core.llm import LLMProvider, OpenAILLM

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "LLMProvider",
    "OpenAILLM",
]
