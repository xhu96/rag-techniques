"""
Text chunking strategies.

Includes implementations for:
- FixedSizeChunker: Simple structure-agnostic chunking
- RecursiveChunker: Structure-aware hierarchical splitting
- SemanticChunker: Embedding-similarity based segmentation
"""

from rag_techniques.chunking.base import Chunk, Chunker
from rag_techniques.chunking.fixed import FixedSizeChunker
from rag_techniques.chunking.semantic import SemanticChunker

__all__ = [
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "SemanticChunker",
]
