"""
LLM-based reranking.

Uses a generative LLM to assess document relevance by asking it to
rate the usefulness of the context for answering the query.
"""

from typing import List
import re

from rag_techniques.reranking.base import Reranker
from rag_techniques.retrieval.base import RetrievalResult
from rag_techniques.core.llm import LLMProvider, OpenAILLM


class LLMReranker(Reranker):
    """
    LLM-based reranker that uses a language model to score relevance.
    
    This approach provides high-quality reranking but is slower and more
    expensive than embedding-based methods.
    """
    
    RERANK_PROMPT = """You are a relevance scoring system. Score how relevant each document is to the query.

Query: {query}

Documents to score:
{documents}

For each document, provide a relevance score from 0.0 to 1.0 where:
- 1.0 = Perfectly relevant, directly answers the query
- 0.7-0.9 = Highly relevant, contains key information
- 0.4-0.6 = Somewhat relevant, contains related information
- 0.1-0.3 = Slightly relevant, tangentially related
- 0.0 = Not relevant at all

Respond with ONLY the scores in this format, one per line:
DOC1: 0.85
DOC2: 0.42
...

Scores:"""

    def __init__(self, llm_provider: LLMProvider | None = None):
        """
        Initialize LLM reranker.
        
        Args:
            llm_provider: LLM provider for scoring (creates OpenAILLM if not provided)
        """
        self.llm = llm_provider or OpenAILLM()
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """
        Rerank results using LLM scoring.
        
        Args:
            query: User query
            results: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Reranked list of RetrievalResult objects
        """
        if not results:
            return []
        
        # Format documents for the prompt
        doc_texts = []
        for i, result in enumerate(results):
            # Truncate long documents
            text = result.text[:500] + "..." if len(result.text) > 500 else result.text
            doc_texts.append(f"DOC{i+1}: {text}")
        
        documents_str = "\n\n".join(doc_texts)
        
        # Get LLM scores
        prompt = self.RERANK_PROMPT.format(query=query, documents=documents_str)
        response = self.llm.generate(prompt)
        
        # Parse scores
        scores = self._parse_scores(response.content, len(results))
        
        # Create scored results
        scored_results = [
            RetrievalResult(
                text=result.text,
                score=scores.get(i, result.score),
                metadata={**result.metadata, "original_score": result.score},
                id=result.id,
            )
            for i, result in enumerate(results)
        ]
        
        # Sort by new scores
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k
        if top_k:
            return scored_results[:top_k]
        return scored_results
    
    def _parse_scores(self, response: str, num_docs: int) -> dict[int, float]:
        """Parse scores from LLM response."""
        scores = {}
        
        # Match patterns like "DOC1: 0.85" or "1: 0.85"
        pattern = r"(?:DOC)?(\d+)\s*:\s*([\d.]+)"
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        for doc_num, score_str in matches:
            try:
                idx = int(doc_num) - 1  # Convert to 0-indexed
                score = float(score_str)
                if 0 <= idx < num_docs and 0 <= score <= 1:
                    scores[idx] = score
            except ValueError:
                continue
        
        return scores
