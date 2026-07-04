from __future__ import annotations

from typing import Optional

from .base import LLMProvider


def get_provider(name: str, model: Optional[str] = None) -> LLMProvider:
    key = (name or "gemini").lower().strip()
    if key == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)
    if key == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    if key == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    raise ValueError(f"Unknown provider: {name}")
