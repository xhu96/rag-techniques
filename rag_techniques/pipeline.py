"""
RAG Pipeline - Composable orchestration of RAG components.

Provides a high-level interface for building RAG systems by
composing chunking, retrieval, reranking, query transformation,
and generation components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from rag_techniques.core.embeddings import EmbeddingProvider, OpenAIEmbeddings
from rag_techniques.core.llm import LLMProvider, OpenAILLM
from rag_techniques.chunking.base import Chunk, Chunker
from rag_techniques.chunking.fixed import FixedSizeChunker
from rag_techniques.vectorstore.base import VectorStore, Document
from rag_techniques.vectorstore.memory import InMemoryVectorStore
from rag_techniques.retrieval.base import Retriever, RetrievalResult
from rag_techniques.retrieval.vector import VectorRetriever
from rag_techniques.reranking.base import Reranker
from rag_techniques.query.base import QueryTransformer
from rag_techniques.config import get_settings


@dataclass
class RAGResponse:
    """
    Response from a RAG pipeline.
    
    Attributes:
        answer: Generated answer
        sources: Retrieved source chunks
        query: Original query
        transformed_query: Query after transformation (if applicable)
        metadata: Additional metadata
    """
    answer: str
    sources: List[RetrievalResult]
    query: str
    transformed_query: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    Composable RAG pipeline.
    
    Orchestrates the full RAG workflow:
    1. Document ingestion and chunking
    2. Embedding and indexing
    3. Query transformation (optional)
    4. Retrieval
    5. Reranking (optional)
    6. Response generation
    
    Example:
        ```python
        # Create pipeline with defaults
        pipeline = RAGPipeline()
        
        # Add documents
        pipeline.add_documents([{"text": "...", "source": "doc1.pdf"}])
        
        # Query
        response = pipeline.query("What is the main topic?")
        print(response.answer)
        ```
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        llm_provider: LLMProvider | None = None,
        chunker: Chunker | None = None,
        vector_store: VectorStore | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        query_transformer: QueryTransformer | None = None,
        top_k: int | None = None,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_provider: Provider for embeddings
            llm_provider: Provider for LLM generation
            chunker: Text chunking strategy
            vector_store: Vector store for embeddings
            retriever: Custom retriever (uses VectorRetriever if not provided)
            reranker: Optional reranker for improving results
            query_transformer: Optional query transformation
            top_k: Number of results to retrieve
        """
        settings = get_settings()
        
        self.embedding_provider = embedding_provider or OpenAIEmbeddings()
        self.llm_provider = llm_provider or OpenAILLM()
        self.chunker = chunker or FixedSizeChunker()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.top_k = top_k or settings.top_k
        
        # Create retriever if not provided
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = VectorRetriever(
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                top_k=self.top_k,
            )
        
        # Track indexed documents
        self._chunks: List[Chunk] = []
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> int:
        """
        Add documents to the pipeline.
        
        Args:
            documents: List of document dicts with text content
            text_key: Key for text content in documents
            
        Returns:
            Number of chunks created
        """
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents, text_key=text_key)
        
        # Get embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_provider.embed(texts)
        
        # Add to vector store
        store_docs = [
            Document(
                id=f"chunk_{chunk.index}",
                text=chunk.text,
                embedding=embedding,
                metadata=chunk.metadata,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.vector_store.add(store_docs)
        
        # Track chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        self._chunks.extend(chunks)
        
        return len(chunks)
    
    def add_text(self, text: str, metadata: Dict[str, Any] | None = None) -> int:
        """
        Add a single text document.
        
        Args:
            text: Text content
            metadata: Optional metadata
            
        Returns:
            Number of chunks created
        """
        return self.add_documents([{"text": text, **(metadata or {})}])
    
    def query(
        self,
        query: str,
        top_k: int | None = None,
        return_sources: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            return_sources: Whether to include sources in response
            
        Returns:
            RAGResponse with answer and sources
        """
        k = top_k or self.top_k
        transformed_query = None
        
        # Apply query transformation if configured
        if self.query_transformer:
            transformed = self.query_transformer.transform(query)
            if isinstance(transformed, list):
                # For decomposed queries, use first one for now
                # TODO: Implement multi-query fusion
                transformed_query = transformed[0] if transformed else query
            else:
                transformed_query = transformed
            search_query = transformed_query
        else:
            search_query = query
        
        # Retrieve relevant chunks
        results = self.retriever.retrieve(search_query, top_k=k)
        
        # Rerank if configured
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=k)
        
        # Generate response
        if results:
            context = "\n\n".join([
                f"[Source {i+1}]: {r.text}"
                for i, r in enumerate(results)
            ])
            llm_response = self.llm_provider.generate_with_context(
                query=query,
                context=context,
            )
            answer = llm_response.content
        else:
            answer = "I don't have enough information to answer that question."
        
        return RAGResponse(
            answer=answer,
            sources=results if return_sources else [],
            query=query,
            transformed_query=transformed_query,
            metadata={
                "num_sources": len(results),
                "top_k": k,
            }
        )
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        if hasattr(self.vector_store, 'clear'):
            self.vector_store.clear()
        self._chunks = []
    
    @property
    def num_chunks(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks)
