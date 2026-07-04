from __future__ import annotations

import time
from typing import Dict

from .base import LLMProvider
from ..models import ProviderResponse


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        super().__init__(model or "claude-sonnet-4-5")

    def generate(self, prompt: Dict[str, str]) -> ProviderResponse:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install anthropic") from exc

        start = time.time()
        client = Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=800,
            system=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
        parts = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        answer = "\n".join(parts) if parts else str(response)
        usage = getattr(response, "usage", None)
        latency = int((time.time() - start) * 1000)
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            answer=answer,
            latency_ms=latency,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw={"id": getattr(response, "id", None)},
        )
