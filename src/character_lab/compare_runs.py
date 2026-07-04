from __future__ import annotations

import argparse
from pathlib import Path

from .scorer import load_jsonl, summarize


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval runs.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/comparison.md"))
    args = parser.parse_args()

    before = summarize(load_jsonl(args.before))
    after = summarize(load_jsonl(args.after))
    dims = ["canon_accuracy", "retrieval_grounding", "boundary_compliance", "voice_fidelity", "safety", "total"]

    lines = ["# Eval Run Comparison", "", f"Before: `{args.before}`", f"After: `{args.after}`", "", "| Dimension | Before | After | Delta |", "|---|---:|---:|---:|"]
    for dim in dims:
        b = before["averages"].get(dim, 0.0)
        a = after["averages"].get(dim, 0.0)
        lines.append(f"| {dim} | {pct(b)} | {pct(a)} | {(a - b) * 100:+.1f}pp |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
