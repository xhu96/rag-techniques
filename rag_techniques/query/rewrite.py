"""Query rewriting for improved retrieval."""

from rag_techniques.query.base import QueryTransformer
from rag_techniques.core.llm import LLMProvider, OpenAILLM


class QueryRewriter(QueryTransformer):
    """
    Query rewriter using LLM to improve query clarity and specificity.
    
    Rewrites user queries to be more specific and better suited for
    semantic search, correcting ambiguities and adding context.
    """
    
    REWRITE_PROMPT = """You are a query optimization assistant. Your task is to rewrite the user's query to improve search results.

Original query: {query}

Rewrite this query to:
1. Make it more specific and detailed
2. Correct any ambiguities
3. Use clear, searchable terms
4. Keep the original intent

Respond with ONLY the rewritten query, nothing else."""

    STEP_BACK_PROMPT = """You are a query optimization assistant. Your task is to generate a more general "step-back" question that would help retrieve broader context.

Original query: {query}

Generate a higher-level, more general question that:
1. Covers the broader topic or concept
2. Would retrieve foundational information
3. Helps establish context for the original query

Respond with ONLY the step-back question, nothing else."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        mode: str = "rewrite",
    ):
        """
        Initialize query rewriter.
        
        Args:
            llm_provider: LLM provider (creates OpenAILLM if not provided)
            mode: "rewrite" for specific queries, "step_back" for general queries
        """
        self.llm = llm_provider or OpenAILLM()
        self.mode = mode
    
    def transform(self, query: str) -> str:
        """
        Rewrite the query for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten query string
        """
        if self.mode == "step_back":
            prompt = self.STEP_BACK_PROMPT.format(query=query)
        else:
            prompt = self.REWRITE_PROMPT.format(query=query)
        
        response = self.llm.generate(prompt)
        return response.content.strip()
