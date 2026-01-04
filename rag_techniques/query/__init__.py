"""
Query transformation techniques.

Techniques to optimize user queries before retrieval:
- QueryRewriter: Standard rewriting and Step-Back Prompting
- QueryDecomposer: Breaking complex queries into sub-questions
- HyDE: Hypothetical Document Embeddings
"""

from rag_techniques.query.base import QueryTransformer
from rag_techniques.query.rewrite import QueryRewriter
from rag_techniques.query.decompose import QueryDecomposer
from rag_techniques.query.hyde import HyDE

__all__ = [
    "QueryTransformer",
    "QueryRewriter",
    "QueryDecomposer",
    "HyDE",
]
