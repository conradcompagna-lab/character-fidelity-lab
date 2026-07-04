from __future__ import annotations

import os
import time
from typing import Dict

from .base import LLMProvider
from ..models import ProviderResponse


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str | None = None):
        super().__init__(model or "gemini-2.5-pro")

    def generate(self, prompt: Dict[str, str]) -> ProviderResponse:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install google-genai") from exc

        start = time.time()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=f"SYSTEM:\n{prompt['system']}\n\nUSER:\n{prompt['user']}",
        )
        answer = getattr(response, "text", None) or str(response)
        latency = int((time.time() - start) * 1000)
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            answer=answer,
            latency_ms=latency,
            raw={"sdk_response_type": type(response).__name__},
        )
