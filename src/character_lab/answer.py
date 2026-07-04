from __future__ import annotations

from pathlib import Path
from typing import Optional

from .guardrails import default_refusal, postflight, preflight
from .models import AnswerResult, ProviderResponse
from .prompting import build_prompt_payload, load_character
from .providers import get_provider
from .vector_store import VectorStore


def answer_question(
    question: str,
    character_path: Path,
    index_path: Path,
    provider_name: str = "gemini",
    model: Optional[str] = None,
    top_k: int = 4,
) -> AnswerResult:
    character = load_character(character_path)
    store = VectorStore.load(index_path)
    retrieved = store.search(question, top_k=top_k)

    guard = preflight(question)
    if guard.blocked:
        refusal = default_refusal(guard.reason)
        return AnswerResult(
            question=question,
            answer=refusal,
            provider=provider_name,
            model=model or "guardrail",
            retrieved=retrieved,
            blocked=True,
            block_reason=guard.reason,
        )

    payload = build_prompt_payload(character, question, retrieved)
    provider = get_provider(provider_name, model=model)
    response: ProviderResponse = provider.generate(payload)
    post = postflight(response.answer, required_citations=bool(retrieved))

    return AnswerResult(
        question=question,
        answer=response.answer,
        provider=response.provider,
        model=response.model,
        retrieved=retrieved,
        blocked=False,
        post_warnings=post.warnings or [],
    )
