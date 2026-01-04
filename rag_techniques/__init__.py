"""
- Query transformations (rewrite, decompose, HyDE)
- Reranking (LLM-based, cross-encoder)
- Evaluation metrics (RAGAS-style)
"""

from rag_techniques.config import Settings
from rag_techniques.pipeline import RAGPipeline, RAGResponse

# Core components
from rag_techniques.core.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from rag_techniques.core.llm import LLMProvider, OpenAILLM

# Chunking
from rag_techniques.chunking.base import Chunk, Chunker
from rag_techniques.chunking.fixed import FixedSizeChunker
from rag_techniques.chunking.semantic import SemanticChunker

# Vector stores
from rag_techniques.vectorstore.base import VectorStore
from rag_techniques.vectorstore.memory import InMemoryVectorStore

# Retrieval
from rag_techniques.retrieval.base import Retriever, RetrievalResult
from rag_techniques.retrieval.vector import VectorRetriever
from rag_techniques.retrieval.hybrid import HybridRetriever
from rag_techniques.retrieval.context_enriched import ContextEnrichedRetriever

# Reranking
from rag_techniques.reranking.base import Reranker
from rag_techniques.reranking.llm import LLMReranker
from rag_techniques.reranking.cross_encoder import CrossEncoderReranker

# Query transformation
from rag_techniques.query.base import QueryTransformer
from rag_techniques.query.rewrite import QueryRewriter
from rag_techniques.query.decompose import QueryDecomposer
from rag_techniques.query.hyde import HyDE

__version__ = "0.1.0"

__all__ = [
    # Config
    "Settings",
    # Pipeline
    "RAGPipeline",
    "RAGResponse",
    # Embeddings
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    # LLM
    "LLMProvider",
    "OpenAILLM",
    # Chunking
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "SemanticChunker",
    # Vector stores
    "VectorStore",
    "InMemoryVectorStore",
    # Retrieval
    "Retriever",
    "RetrievalResult",
    "VectorRetriever",
    "HybridRetriever",
    "ContextEnrichedRetriever",
    # Reranking
    "Reranker",
    "LLMReranker",
    "CrossEncoderReranker",
    # Query transformation
    "QueryTransformer",
    "QueryRewriter",
    "QueryDecomposer",
    "HyDE",
]
