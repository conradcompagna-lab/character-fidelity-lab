from __future__ import annotations

import argparse
from pathlib import Path

from .vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector index from lore chunks.")
    parser.add_argument("--chunks", type=Path, default=Path("data/index/chunks.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/index/vector_store.json"))
    parser.add_argument("--embedder", choices=["hashing", "gemini"], default="hashing")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--dim", type=int, default=None, help="Hashing vector width (default 512), or Gemini output_dimensionality (default: model default, ~3072)")
    args = parser.parse_args()

    dim = args.dim if args.dim is not None else (512 if args.embedder == "hashing" else None)
    store = VectorStore.from_chunks_path(
        args.chunks,
        embedder_type=args.embedder,
        embedder_model=args.embedding_model,
        dim=dim,
    )
    store.save(args.out)
    print(f"Indexed {len(store.chunks)} chunks to {args.out} using {store.embedder_type} embeddings (dim={store.dim})")


if __name__ == "__main__":
    main()
