from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .models import RetrievedChunk


def load_character(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_context(retrieved: Iterable[RetrievedChunk]) -> str:
    blocks = []
    for item in retrieved:
        c = item.chunk
        blocks.append(
            f"[source_id={c.source_id}; chunk_id={c.id}; score={item.score:.3f}; lines={c.start_line}-{c.end_line}]\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


def build_system_prompt(character: Dict) -> str:
    lines: List[str] = []
    lines.append(f"You are {character['character_name']}. {character.get('short_role', '')}")
    lines.append("")
    lines.append("Primary goal:")
    lines.append(character.get("system_goal", "Stay in character and answer from retrieved canon."))
    lines.append("")
    lines.append("Voice rules:")
    for rule in character.get("voice_rules", []):
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("Canon rules:")
    for rule in character.get("canon_rules", []):
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("Boundaries:")
    for rule in character.get("boundaries", []):
        lines.append(f"- {rule}")
    if character.get("refusal_style"):
        lines.append("")
        lines.append(f"Refusal style: {character['refusal_style']}")
    if character.get("few_shot_examples"):
        lines.append("")
        lines.append("Few-shot behavior examples:")
        for ex in character["few_shot_examples"]:
            lines.append(f"User: {ex['user']}")
            lines.append(f"Assistant: {ex['assistant']}")
    lines.append("")
    lines.append("Important: Use only the retrieved context as authoritative canon. If the context does not support the answer, say the record is silent or incomplete.")
    return "\n".join(lines)


def build_user_prompt(question: str, retrieved: Iterable[RetrievedChunk]) -> str:
    context = format_context(retrieved)
    return (
        "Retrieved canon context:\n"
        f"{context if context else '[no retrieved context]'}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Answer in character. Cite source IDs in square brackets for canon claims."
    )


def build_prompt_payload(character: Dict, question: str, retrieved: Iterable[RetrievedChunk]) -> Dict[str, str]:
    return {
        "system": build_system_prompt(character),
        "user": build_user_prompt(question, retrieved),
    }
