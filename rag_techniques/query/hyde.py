"""HyDE - Hypothetical Document Embeddings."""

from typing import List

from rag_techniques.query.base import QueryTransformer
from rag_techniques.core.llm import LLMProvider, OpenAILLM
from rag_techniques.core.embeddings import EmbeddingProvider


class HyDE(QueryTransformer):
    """
    Hypothetical Document Embeddings (HyDE) for improved retrieval.
    
    Instead of embedding the query directly, HyDE:
    1. Generates a hypothetical answer to the query
    2. Embeds the hypothetical answer
    3. Uses that embedding for similarity search
    
    This can improve retrieval because the hypothetical answer is more
    similar to actual documents than the question itself.
    
    Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
               https://arxiv.org/abs/2212.10496
    """
    
    HYDE_PROMPT = """You are a helpful assistant. Answer the following question as if you were writing a passage that would appear in a document.

Question: {query}

Write a detailed, informative passage that would answer this question. The passage should be factual and comprehensive, like content from an encyclopedia or textbook. Do not include phrases like "I think" or "In my opinion".

Passage:"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        num_hypotheticals: int = 1,
    ):
        """
        Initialize HyDE.
        
        Args:
            llm_provider: LLM provider for generating hypothetical documents
            embedding_provider: Embedding provider (optional, for getting embeddings)
            num_hypotheticals: Number of hypothetical documents to generate
        """
        self.llm = llm_provider or OpenAILLM()
        self.embedding_provider = embedding_provider
        self.num_hypotheticals = num_hypotheticals
    
    def transform(self, query: str) -> str:
        """
        Generate a hypothetical document for the query.
        
        Args:
            query: Original user query
            
        Returns:
            Hypothetical document text
        """
        prompt = self.HYDE_PROMPT.format(query=query)
        response = self.llm.generate(prompt)
        return response.content.strip()
    
    def transform_multiple(self, query: str) -> List[str]:
        """
        Generate multiple hypothetical documents.
        
        Args:
            query: Original user query
            
        Returns:
            List of hypothetical document texts
        """
        hypotheticals = []
        for _ in range(self.num_hypotheticals):
            hypotheticals.append(self.transform(query))
        return hypotheticals
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for query using HyDE.
        
        Generates a hypothetical document and embeds that instead of
        the original query.
        
        Args:
            query: Original user query
            
        Returns:
            Embedding vector for the hypothetical document
        """
        if self.embedding_provider is None:
            raise ValueError("embedding_provider required for get_query_embedding")
        
        hypothetical = self.transform(query)
        return self.embedding_provider.embed_query(hypothetical)
