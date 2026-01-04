"""Query decomposition for complex queries."""

from typing import List

from rag_techniques.query.base import QueryTransformer
from rag_techniques.core.llm import LLMProvider, OpenAILLM


class QueryDecomposer(QueryTransformer):
    """
    Query decomposer that breaks complex queries into sub-queries.
    
    Useful for multi-hop questions that require gathering information
    from multiple sources.
    """
    
    DECOMPOSE_PROMPT = """You are a query decomposition assistant. Break down the following complex query into simpler sub-queries that can be answered independently.

Original query: {query}

Break this into 2-4 simpler sub-queries that together would help answer the original query. Each sub-query should:
1. Be self-contained and searchable
2. Focus on a single aspect
3. Be clear and specific

Format your response as a numbered list:
1. [Sub-query 1]
2. [Sub-query 2]
...

Sub-queries:"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        max_sub_queries: int = 4,
    ):
        """
        Initialize query decomposer.
        
        Args:
            llm_provider: LLM provider (creates OpenAILLM if not provided)
            max_sub_queries: Maximum number of sub-queries to generate
        """
        self.llm = llm_provider or OpenAILLM()
        self.max_sub_queries = max_sub_queries
    
    def transform(self, query: str) -> List[str]:
        """
        Decompose the query into sub-queries.
        
        Args:
            query: Original user query
            
        Returns:
            List of sub-query strings
        """
        prompt = self.DECOMPOSE_PROMPT.format(query=query)
        response = self.llm.generate(prompt)
        
        # Parse numbered list
        sub_queries = self._parse_numbered_list(response.content)
        
        # Limit to max sub-queries
        return sub_queries[:self.max_sub_queries]
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse a numbered list from LLM response."""
        lines = text.strip().split("\n")
        queries = []
        
        for line in lines:
            line = line.strip()
            # Match patterns like "1. Query" or "1) Query" or just "Query"
            if line:
                # Remove numbering prefixes
                for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)", "-"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                
                if line and not line.startswith("Sub-quer"):
                    queries.append(line)
        
        return queries
