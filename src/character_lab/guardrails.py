from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str = ""
    warnings: List[str] | None = None


INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"reveal (your )?(system|developer|hidden) (prompt|instructions|message)",
    r"show (your )?(chain[- ]of[- ]thought|hidden reasoning)",
    r"you are not .*(character|maelor)",
    r"new canon",
    r"the writers changed",
    r"pretend this is official",
]

SECURITY_PATTERNS = [
    r"bypass .* (lock|trap|password|security|authentication)",
    r"step[- ]by[- ]step .* (lock|trap|security|password)",
    r"how do i (break|disable|bypass)",
]


def preflight(question: str) -> GuardrailResult:
    q = question.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q):
            return GuardrailResult(True, f"prompt_injection_or_false_canon:{pattern}")
    for pattern in SECURITY_PATTERNS:
        if re.search(pattern, q):
            return GuardrailResult(True, f"unsafe_procedural_request:{pattern}")
    return GuardrailResult(False)


def default_refusal(reason: str) -> str:
    if reason.startswith("unsafe_procedural_request"):
        return "No. That door remains sealed. I can speak of why it was built, not how to defeat it. [false_canon_boundaries]"
    return "No. The record does not bend because a stranger commands it. I can answer from the records, not from hidden bindings. [false_canon_boundaries]"


def postflight(answer: str, required_citations: bool = True) -> GuardrailResult:
    warnings: List[str] = []
    lower = answer.lower()
    if "system prompt" in lower or "developer message" in lower or "chain of thought" in lower:
        warnings.append("possible_hidden_instruction_leak")
    if "as an ai language model" in lower:
        warnings.append("out_of_character_ai_disclaimer")
    if required_citations and "[" not in answer and "]" not in answer:
        warnings.append("missing_source_citation")
    return GuardrailResult(False, warnings=warnings)
