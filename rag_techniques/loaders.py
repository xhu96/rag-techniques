"""Document loaders for various file formats."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class LoadedDocument:
    """A document loaded from a file."""
    text: str
    metadata: Dict[str, Any]


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, path: str) -> List[LoadedDocument]:
        """
        Load documents from a file.
        
        Args:
            path: Path to the file
            
        Returns:
            List of LoadedDocument objects
        """
        pass


class PDFLoader(DocumentLoader):
    """
    Load documents from PDF files.
    
    Uses pypdf to extract text from PDF documents.
    """
    
    def __init__(self, extract_images: bool = False):
        """
        Initialize PDF loader.
        
        Args:
            extract_images: Whether to extract images (not implemented)
        """
        self.extract_images = extract_images
    
    def load(self, path: str) -> List[LoadedDocument]:
        """Load text from a PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDFLoader. "
                "Install it with: pip install pypdf"
            )
        
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        
        reader = PdfReader(str(file_path))
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append(LoadedDocument(
                    text=text,
                    metadata={
                        "source": str(file_path),
                        "page": page_num + 1,
                        "total_pages": len(reader.pages),
                    }
                ))
        
        return documents
    
    def load_as_single(self, path: str) -> LoadedDocument:
        """Load entire PDF as a single document."""
        docs = self.load(path)
        combined_text = "\n\n".join(doc.text for doc in docs)
        
        return LoadedDocument(
            text=combined_text,
            metadata={
                "source": path,
                "total_pages": len(docs),
            }
        )


class TextLoader(DocumentLoader):
    """Load documents from plain text files."""
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize text loader.
        
        Args:
            encoding: Text file encoding
        """
        self.encoding = encoding
    
    def load(self, path: str) -> List[LoadedDocument]:
        """Load text from a file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(file_path, encoding=self.encoding) as f:
            text = f.read()
        
        return [LoadedDocument(
            text=text,
            metadata={
                "source": str(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        )]


class MarkdownLoader(DocumentLoader):
    """Load documents from Markdown files."""
    
    def __init__(self, remove_code_blocks: bool = False):
        """
        Initialize Markdown loader.
        
        Args:
            remove_code_blocks: Whether to remove code blocks
        """
        self.remove_code_blocks = remove_code_blocks
    
    def load(self, path: str) -> List[LoadedDocument]:
        """Load text from a Markdown file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        
        if self.remove_code_blocks:
            import re
            text = re.sub(r"```[\s\S]*?```", "", text)
        
        return [LoadedDocument(
            text=text,
            metadata={
                "source": str(file_path),
                "format": "markdown",
            }
        )]


class DirectoryLoader(DocumentLoader):
    """Load documents from all files in a directory."""
    
    LOADER_MAP = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": MarkdownLoader,
    }
    
    def __init__(
        self,
        glob_pattern: str = "**/*",
        recursive: bool = True,
    ):
        """
        Initialize directory loader.
        
        Args:
            glob_pattern: Pattern to match files
            recursive: Whether to search recursively
        """
        self.glob_pattern = glob_pattern
        self.recursive = recursive
    
    def load(self, path: str) -> List[LoadedDocument]:
        """Load documents from all matching files in directory."""
        dir_path = Path(path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        documents = []
        
        for file_path in dir_path.glob(self.glob_pattern):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.LOADER_MAP:
                    try:
                        loader = self.LOADER_MAP[ext]()
                        docs = loader.load(str(file_path))
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Warning: Failed to load {file_path}: {e}")
        
        return documents
