from __future__ import annotations

import time
from typing import Dict

from .base import LLMProvider
from ..models import ProviderResponse


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None):
        super().__init__(model or "gpt-5.5")

    def generate(self, prompt: Dict[str, str]) -> ProviderResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install openai") from exc

        start = time.time()
        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
        )
        answer = getattr(response, "output_text", None)
        if not answer:
            answer = str(response)
        latency = int((time.time() - start) * 1000)
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            answer=answer,
            latency_ms=latency,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw={"id": getattr(response, "id", None)},
        )
