from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..models import ProviderResponse


class LLMProvider(ABC):
    name = "base"

    def __init__(self, model: Optional[str] = None):
        self.model = model or "default"

    @abstractmethod
    def generate(self, prompt: Dict[str, str]) -> ProviderResponse:
        raise NotImplementedError
