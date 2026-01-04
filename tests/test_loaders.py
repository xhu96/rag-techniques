"""Tests for document loaders."""

import pytest
from pathlib import Path
import tempfile

from rag_techniques.loaders import (
    LoadedDocument,
    TextLoader,
    MarkdownLoader,
)


class TestLoadedDocument:
    """Tests for LoadedDocument dataclass."""
    
    def test_creation(self):
        doc = LoadedDocument(
            text="Hello world",
            metadata={"source": "test.txt"},
        )
        assert doc.text == "Hello world"
        assert doc.metadata["source"] == "test.txt"


class TestTextLoader:
    """Tests for TextLoader."""
    
    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world, this is a test.")
            temp_path = f.name
        
        loader = TextLoader()
        docs = loader.load(temp_path)
        
        assert len(docs) == 1
        assert "Hello world" in docs[0].text
        assert "source" in docs[0].metadata
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_file_not_found(self):
        loader = TextLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.txt")


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""
    
    def test_load_markdown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Heading\n\nThis is a test.")
            temp_path = f.name
        
        loader = MarkdownLoader()
        docs = loader.load(temp_path)
        
        assert len(docs) == 1
        assert "# Heading" in docs[0].text
        assert docs[0].metadata["format"] == "markdown"
        
        Path(temp_path).unlink()
    
    def test_remove_code_blocks(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Text\n\n```python\ncode here\n```\n\nMore text")
            temp_path = f.name
        
        loader = MarkdownLoader(remove_code_blocks=True)
        docs = loader.load(temp_path)
        
        assert "code here" not in docs[0].text
        assert "Text" in docs[0].text
        
        Path(temp_path).unlink()
