"""Configuration management using Pydantic settings."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Global settings for the RAG Techniques library.
    
    Settings are loaded from environment variables with the RAG_ prefix.
    Example: RAG_OPENAI_API_KEY, RAG_EMBEDDING_MODEL
    """
    
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL"
    )
    
    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Default embedding model"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Default LLM model for generation"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Default cross-encoder model for reranking"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Default chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Default chunk overlap")
    
    # Retrieval Configuration
    top_k: int = Field(default=5, ge=1, le=100, description="Default number of results")
    similarity_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum similarity threshold for retrieval"
    )
    
    # Hybrid Search Configuration
    hybrid_alpha: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Weight for dense retrieval in hybrid search (1-alpha for sparse)"
    )
    
    # Generation Configuration
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=1024, ge=1, description="Maximum tokens for LLM response")
    
    # Vector Store Configuration
    vectorstore_type: Literal["memory", "chroma"] = Field(
        default="memory",
        description="Default vector store type"
    )
    chroma_persist_directory: str = Field(
        default="./chroma_db",
        description="ChromaDB persistence directory"
    )


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure(**kwargs) -> Settings:
    """
    Configure global settings with custom values.
    
    Args:
        **kwargs: Settings to override
        
    Returns:
        Updated Settings instance
    """
    global _settings
    _settings = Settings(**kwargs)
    return _settings
