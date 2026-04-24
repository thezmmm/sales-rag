"""
LLM provider package.

Public API
----------
    from llm import make_llm, LLMProvider, ChatResponse, ToolCall

    llm = make_llm("ollama", model="llama3.2:3b")
    llm = make_llm("openai", model="gpt-4o-mini", api_key="sk-...")
    llm = make_llm("openai", model="llama3-8b-8192",
                   api_key="gsk_...", base_url="https://api.groq.com/openai/v1")
"""

from llm.base import ChatResponse, LLMProvider, ToolCall
from llm.ollama import OllamaProvider
from llm.openai import OpenAIProvider

__all__ = [
    "LLMProvider",
    "ChatResponse",
    "ToolCall",
    "OllamaProvider",
    "OpenAIProvider",
    "make_llm",
]

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


def make_llm(provider: str = "ollama", **kwargs) -> LLMProvider:
    """
    Instantiate an LLM provider by name.

    Parameters
    ----------
    provider : 'ollama' or 'openai'
    **kwargs : forwarded to the provider constructor
               ollama  → model, host
               openai  → model, api_key, base_url

    Raises
    ------
    ValueError if provider is not recognised.
    """
    if provider not in _PROVIDERS:
        supported = ", ".join(f"'{p}'" for p in _PROVIDERS)
        raise ValueError(f"Unknown LLM provider {provider!r}. Choose from: {supported}.")
    return _PROVIDERS[provider](**kwargs)