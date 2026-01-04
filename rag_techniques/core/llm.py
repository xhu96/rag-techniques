"""LLM p"""
LLM provider implementations.

Contains:
- OpenAILLM: Wrapper for OpenAI chat completion API (and compatible endpoints)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

from openai import OpenAI

from rag_techniques.config import get_settings


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: Dict[str, int] | None = None
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User query
            context: Retrieved context to ground the response
            system_prompt: Optional system prompt
            
        Returns:
            LLMResponse with generated content
        """
        pass


class OpenAILLM(LLMProvider):
    """
    OpenAI-compatible LLM provider.
    
    Works with OpenAI API and compatible endpoints.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context. 
If the answer cannot be found in the context, say so clearly. 
Do not make up information that is not supported by the context."""
    
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model: Model name (default from settings)
            api_key: API key (default from settings)
            base_url: API base URL (default from settings)
            temperature: Default temperature (default from settings)
            max_tokens: Default max tokens (default from settings)
        """
        settings = get_settings()
        self.model = model or settings.llm_model
        self.default_temperature = temperature if temperature is not None else settings.temperature
        self.default_max_tokens = max_tokens or settings.max_tokens
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        messages: List[Dict[str, str]] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw_response=response,
        )
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using retrieved context."""
        sys_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a helpful answer based on the context above."""
        
        return self.generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
        )
