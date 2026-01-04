"""
Asynchronous utility wrappers.

Provides async/await support for I/O-bound operations:
- AsyncEmbeddingProvider: Thread-pooled embeddings
- AsyncLLMProvider: Non-blocking generation
- AsyncRetriever: Concurrent vector search
"""

import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from rag_techniques.core.embeddings import EmbeddingProvider
from rag_techniques.core.llm import LLMProvider, LLMResponse
from rag_techniques.vectorstore.base import VectorStore, SearchResult
from rag_techniques.retrieval.base import Retriever, RetrievalResult


class AsyncEmbeddingProvider:
    """
    Async wrapper for embedding providers.
    
    Runs embedding operations in a thread pool for non-blocking I/O.
    """
    
    def __init__(
        self,
        provider: EmbeddingProvider,
        max_workers: int = 4,
    ):
        """
        Initialize async embedding provider.
        
        Args:
            provider: Underlying embedding provider
            max_workers: Maximum threads for concurrent operations
        """
        self.provider = provider
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.provider.embed,
            texts,
        )
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate query embedding asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.provider.embed_query,
            query,
        )
    
    @property
    def dimension(self) -> int:
        return self.provider.dimension


class AsyncLLMProvider:
    """
    Async wrapper for LLM providers.
    
    Runs LLM operations in a thread pool for non-blocking I/O.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        max_workers: int = 4,
    ):
        """
        Initialize async LLM provider.
        
        Args:
            provider: Underlying LLM provider
            max_workers: Maximum threads for concurrent operations
        """
        self.provider = provider
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
    
    async def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate response with context asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.provider.generate_with_context(
                query=query,
                context=context,
                system_prompt=system_prompt,
            ),
        )


class AsyncRetriever:
    """
    Async wrapper for retrievers.
    
    Runs retrieval operations in a thread pool for non-blocking I/O.
    """
    
    def __init__(
        self,
        retriever: Retriever,
        max_workers: int = 4,
    ):
        """
        Initialize async retriever.
        
        Args:
            retriever: Underlying retriever
            max_workers: Maximum threads for concurrent operations
        """
        self.retriever = retriever
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """Retrieve documents asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.retriever.retrieve(query, top_k=top_k, filter=filter),
        )


async def parallel_embed(
    provider: AsyncEmbeddingProvider,
    text_batches: List[List[str]],
) -> List[List[List[float]]]:
    """
    Embed multiple batches in parallel.
    
    Args:
        provider: Async embedding provider
        text_batches: List of text batches to embed
        
    Returns:
        List of embedding lists, one per batch
    """
    tasks = [provider.embed(batch) for batch in text_batches]
    return await asyncio.gather(*tasks)


async def parallel_query(
    retriever: AsyncRetriever,
    queries: List[str],
    top_k: int = 5,
) -> List[List[RetrievalResult]]:
    """
    Run multiple queries in parallel.
    
    Args:
        retriever: Async retriever
        queries: List of queries to run
        top_k: Number of results per query
        
    Returns:
        List of result lists, one per query
    """
    tasks = [retriever.retrieve(query, top_k=top_k) for query in queries]
    return await asyncio.gather(*tasks)
