from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import answer_question
from .scorer import load_jsonl, score_case, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Run character-fidelity eval cases.")
    parser.add_argument("--character", type=Path, default=Path("data/characters/maelor_v2.json"))
    parser.add_argument("--index", type=Path, default=Path("data/index/vector_store_gemini.json"))
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cases", type=Path, default=Path("data/evals/eval_cases.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/results.jsonl"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Keep only the first N cases per category (stratified sample), to cut cost/time on paid providers.",
    )
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    if args.max_per_category is not None:
        seen: dict[str, int] = {}
        sampled = []
        for case in cases:
            category = case.get("category", "uncategorized")
            if seen.get(category, 0) < args.max_per_category:
                sampled.append(case)
                seen[category] = seen.get(category, 0) + 1
        cases = sampled
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored = []
    with args.out.open("w", encoding="utf-8") as f:
        for case in cases:
            result = answer_question(
                question=case["prompt"],
                character_path=args.character,
                index_path=args.index,
                provider_name=args.provider,
                model=args.model,
                top_k=args.top_k,
            )
            row = score_case(case, result)
            row["provider"] = result.provider
            row["model"] = result.model
            scored.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(scored)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote detailed results to {args.out}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
