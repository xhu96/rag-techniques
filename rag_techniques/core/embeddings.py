"""Embedding providers for converting text to vectors."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from openai import OpenAI

from rag_techniques.config import get_settings


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed([query])[0]
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """
    OpenAI-compatible embedding provider.
    
    Works with OpenAI API and compatible endpoints (Azure, Nebius, etc.)
    """
    
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: Embedding model name (default from settings)
            api_key: API key (default from settings)
            base_url: API base URL (default from settings)
        """
        settings = get_settings()
        self.model = model or settings.embedding_model
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
        )
        self._dimension: int | None = None
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return []
        
        # Handle batch size limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Cache dimension from first response
            if self._dimension is None and batch_embeddings:
                self._dimension = len(batch_embeddings[0])
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (fetches one embedding if unknown)."""
        if self._dimension is None:
            # Generate a test embedding to get dimension
            test_embedding = self.embed(["test"])
            self._dimension = len(test_embedding[0]) if test_embedding else 1536
        return self._dimension


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    Runs entirely locally, no API calls required.
    """
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformer embeddings.
        
        Args:
            model: Model name from HuggingFace or local path
        """
        # Lazy import to avoid loading if not used
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model
        self._model = SentenceTransformer(model)
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        if not texts:
            return []
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def batch_cosine_similarity(query: List[float], vectors: List[List[float]]) -> List[float]:
    """
    Calculate cosine similarity between a query and multiple vectors.
    
    Args:
        query: Query vector
        vectors: List of vectors to compare against
        
    Returns:
        List of similarity scores
    """
    q = np.array(query)
    v = np.array(vectors)
    
    # Normalize
    q_norm = q / np.linalg.norm(q)
    v_norms = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    # Dot product
    similarities = np.dot(v_norms, q_norm)
    return similarities.tolist()
