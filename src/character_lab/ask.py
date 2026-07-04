from __future__ import annotations

import argparse
from pathlib import Path

from .answer import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a lore-grounded character a question.")
    parser.add_argument("--character", type=Path, default=Path("data/characters/maelor_v2.json"))
    parser.add_argument("--index", type=Path, default=Path("data/index/vector_store_gemini.json"))
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    result = answer_question(
        question=args.question,
        character_path=args.character,
        index_path=args.index,
        provider_name=args.provider,
        model=args.model,
        top_k=args.top_k,
    )
    print("ANSWER")
    print(result.answer)
    print("\nRETRIEVED")
    for item in result.retrieved:
        print(f"- {item.chunk.source_id} / {item.chunk.id}: {item.score:.3f}")
    if result.blocked:
        print(f"\nBLOCKED: {result.block_reason}")
    if result.post_warnings:
        print("\nWARNINGS")
        for warning in result.post_warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
