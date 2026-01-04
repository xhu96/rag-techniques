"""
Semantic text segmentation.

Uses embedding similarity to identify topic shifts and natural breakpoints
in the text stream.
"""

from typing import List, Dict, Any, Literal
import re
import numpy as np

from rag_techniques.chunking.base import Chunk, Chunker
from rag_techniques.core.embeddings import EmbeddingProvider, cosine_similarity


class SemanticChunker(Chunker):
    """
    Semantic chunking using embedding similarity to find natural breakpoints.
    
    This chunker:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Finds breakpoints where semantic similarity drops significantly
    4. Groups sentences between breakpoints into chunks
    
    Reference: Based on techniques from LangChain's SemanticChunker
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        breakpoint_method: Literal["percentile", "standard_deviation", "interquartile"] = "percentile",
        breakpoint_threshold: float = 90.0,
        min_chunk_size: int = 100,
        max_chunk_size: int = 3000,
    ):
        """
        Initialize semantic chunker.
        
        Args:
            embedding_provider: Provider for generating sentence embeddings
            breakpoint_method: Method for detecting breakpoints:
                - "percentile": Break at similarity below X percentile
                - "standard_deviation": Break at similarity below mean - X*std
                - "interquartile": Break using IQR outlier detection
            breakpoint_threshold: Threshold value for the method:
                - percentile: 0-100 (higher = fewer breaks)
                - standard_deviation: number of std devs (typically 1-3)
                - interquartile: IQR multiplier (typically 1.5)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
        """
        self.embedding_provider = embedding_provider
        self.breakpoint_method = breakpoint_method
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Dict[str, Any] | None = None) -> List[Chunk]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text, index=0, metadata=metadata or {})]
        
        # Get embeddings for all sentences
        embeddings = self.embedding_provider.embed(sentences)
        
        # Compute similarities between consecutive sentences
        similarities = [
            cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        
        # Find breakpoints
        breakpoints = self._compute_breakpoints(similarities)
        
        # Split into chunks at breakpoints
        chunks = self._create_chunks_from_breakpoints(sentences, breakpoints, metadata)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Split on sentence-ending punctuation followed by space or newline
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]
    
    def _compute_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        Compute breakpoint indices based on similarity drops.
        
        Args:
            similarities: List of similarity scores between consecutive sentences
            
        Returns:
            List of indices where breaks should occur
        """
        if not similarities:
            return []
        
        sim_array = np.array(similarities)
        
        if self.breakpoint_method == "percentile":
            # Break where similarity is below the X percentile
            threshold = np.percentile(sim_array, 100 - self.breakpoint_threshold)
        
        elif self.breakpoint_method == "standard_deviation":
            # Break where similarity is below mean - X*std
            mean = np.mean(sim_array)
            std = np.std(sim_array)
            threshold = mean - (self.breakpoint_threshold * std)
        
        elif self.breakpoint_method == "interquartile":
            # Use IQR method for outlier detection
            q1, q3 = np.percentile(sim_array, [25, 75])
            iqr = q3 - q1
            threshold = q1 - (self.breakpoint_threshold * iqr)
        
        else:
            raise ValueError(f"Unknown breakpoint method: {self.breakpoint_method}")
        
        # Find indices where similarity drops below threshold
        breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]
        
        return breakpoints
    
    def _create_chunks_from_breakpoints(
        self,
        sentences: List[str],
        breakpoints: List[int],
        metadata: Dict[str, Any] | None,
    ) -> List[Chunk]:
        """Create chunks by grouping sentences between breakpoints."""
        chunks = []
        base_metadata = metadata or {}
        
        start_idx = 0
        chunk_index = 0
        
        # Add end of document as final breakpoint
        all_breakpoints = breakpoints + [len(sentences) - 1]
        
        for bp in all_breakpoints:
            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:bp + 1]
            chunk_text = " ".join(chunk_sentences)
            
            # Handle size constraints
            if len(chunk_text) < self.min_chunk_size and chunks:
                # Merge with previous chunk if too small
                prev_chunk = chunks[-1]
                prev_chunk.text = prev_chunk.text + " " + chunk_text
                prev_chunk.metadata["end_sentence"] = bp
            elif len(chunk_text) > self.max_chunk_size:
                # Split oversized chunks
                sub_chunks = self._split_oversized_chunk(
                    chunk_text, chunk_index, base_metadata
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                chunk_metadata = {
                    **base_metadata,
                    "start_sentence": start_idx,
                    "end_sentence": bp,
                    "num_sentences": len(chunk_sentences),
                }
                chunks.append(Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    metadata=chunk_metadata,
                ))
                chunk_index += 1
            
            start_idx = bp + 1
        
        return chunks
    
    def _split_oversized_chunk(
        self,
        text: str,
        start_index: int,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Split an oversized chunk into smaller pieces."""
        chunks = []
        for i in range(0, len(text), self.max_chunk_size):
            chunk_text = text[i:i + self.max_chunk_size]
            chunks.append(Chunk(
                text=chunk_text,
                index=start_index + len(chunks),
                metadata={**metadata, "split_from_oversized": True},
            ))
        return chunks
