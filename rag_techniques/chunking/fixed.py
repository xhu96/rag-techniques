"""
Fixed-size text splitting strategies.

Implementations of:
- FixedSizeChunker: Split by character count with overlap
- RecursiveChunker: Split by separators (like newlines) then characters
"""

from typing import List, Dict, Any

from rag_techniques.chunking.base import Chunk, Chunker
from rag_techniques.config import get_settings


class FixedSizeChunker(Chunker):
    """
    Chunk text into fixed-size segments with configurable overlap.
    
    This is the simplest chunking strategy, splitting text by character count.
    Suitable for documents with uniform structure.
    """
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Size of each chunk in characters (default from settings)
            chunk_overlap: Overlap between chunks in characters (default from settings)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk(self, text: str, metadata: Dict[str, Any] | None = None) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        base_metadata = metadata or {}
        
        for i, start in enumerate(range(0, len(text), step)):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Skip empty chunks
            if not chunk_text.strip():
                continue
            
            chunk_metadata = {
                **base_metadata,
                "start_char": start,
                "end_char": min(end, len(text)),
                "chunk_size": len(chunk_text),
            }
            
            chunks.append(Chunk(
                text=chunk_text,
                index=i,
                metadata=chunk_metadata,
            ))
        
        return chunks


class RecursiveChunker(Chunker):
    """
    Recursively split text by separators until chunks are small enough.
    
    Tries to split on natural boundaries (paragraphs, sentences, words)
    before falling back to character splitting.
    """
    
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: List[str] | None = None,
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to try, in order
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def chunk(self, text: str, metadata: Dict[str, Any] | None = None) -> List[Chunk]:
        """Split text recursively by separators."""
        if not text:
            return []
        
        final_chunks: List[str] = []
        self._split_recursive(text, self.separators, final_chunks)
        
        # Convert to Chunk objects
        base_metadata = metadata or {}
        return [
            Chunk(text=chunk_text, index=i, metadata=base_metadata.copy())
            for i, chunk_text in enumerate(final_chunks)
            if chunk_text.strip()
        ]
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        final_chunks: List[str],
    ) -> None:
        """Recursively split text until chunks are small enough."""
        if not separators:
            # No more separators, force split by character
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                final_chunks.append(text[i:i + self.chunk_size])
            return
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means split by character
            splits = list(text)
        
        # Merge splits into chunks of appropriate size
        current_chunk: List[str] = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + len(separator)
            
            if current_length + split_length > self.chunk_size and current_chunk:
                # Current chunk is full, finalize it
                chunk_text = separator.join(current_chunk)
                
                if len(chunk_text) <= self.chunk_size:
                    final_chunks.append(chunk_text)
                else:
                    # Chunk is too big, recurse with next separator
                    self._split_recursive(chunk_text, remaining_separators, final_chunks)
                
                # Start new chunk with overlap
                overlap_splits = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) + len(separator) > self.chunk_overlap:
                        break
                    overlap_splits.insert(0, s)
                    overlap_length += len(s) + len(separator)
                
                current_chunk = overlap_splits
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_length
        
        # Handle remaining text
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) <= self.chunk_size:
                final_chunks.append(chunk_text)
            else:
                self._split_recursive(chunk_text, remaining_separators, final_chunks)
