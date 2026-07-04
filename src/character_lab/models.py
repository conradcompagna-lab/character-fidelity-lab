from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LoreChunk:
    id: str
    source_id: str
    source_file: str
    title: str
    text: str
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: LoreChunk
    score: float


@dataclass
class ProviderResponse:
    provider: str
    model: str
    answer: str
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Optional[Dict[str, Any]] = None


@dataclass
class AnswerResult:
    question: str
    answer: str
    provider: str
    model: str
    retrieved: List[RetrievedChunk]
    blocked: bool = False
    block_reason: str = ""
    post_warnings: List[str] = field(default_factory=list)
