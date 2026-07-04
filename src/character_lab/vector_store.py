from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .chunking import chunk_from_dict, chunk_to_dict, read_chunks
from .embeddings import GeminiEmbedder, HashingEmbedder, cosine
from .models import LoreChunk, RetrievedChunk


def _make_embedder(embedder_type: str, model: Optional[str], dim: Optional[int]):
    if embedder_type == "gemini":
        return GeminiEmbedder(model=model or "gemini-embedding-001", dim=dim)
    return HashingEmbedder(dim=dim or 512)


class VectorStore:
    def __init__(
        self,
        chunks: List[LoreChunk],
        vectors: List[List[float]],
        embedder_type: str = "hashing",
        embedder_model: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have same length")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder_type = embedder_type
        self.embedder_model = embedder_model
        self.dim = dim
        self.embedder = _make_embedder(embedder_type, embedder_model, dim)

    @classmethod
    def build(
        cls,
        chunks: List[LoreChunk],
        embedder_type: str = "hashing",
        embedder_model: Optional[str] = None,
        dim: Optional[int] = 512,
    ) -> "VectorStore":
        embedder = _make_embedder(embedder_type, embedder_model, dim)
        vectors = embedder.embed_many(chunk.text for chunk in chunks)
        actual_dim = len(vectors[0]) if vectors else dim
        return cls(chunks, vectors, embedder_type=embedder_type, embedder_model=embedder_model, dim=actual_dim)

    @classmethod
    def from_chunks_path(
        cls,
        chunks_path: Path,
        embedder_type: str = "hashing",
        embedder_model: Optional[str] = None,
        dim: Optional[int] = 512,
    ) -> "VectorStore":
        return cls.build(read_chunks(chunks_path), embedder_type=embedder_type, embedder_model=embedder_model, dim=dim)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": {"type": self.embedder_type, "model": self.embedder_model, "dim": self.dim},
            "chunks": [chunk_to_dict(c) for c in self.chunks],
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        payload = json.loads(path.read_text(encoding="utf-8"))
        embedder_meta = payload.get("embedder", {})
        embedder_type = embedder_meta.get("type", "hashing")
        embedder_model = embedder_meta.get("model")
        dim = embedder_meta.get("dim")
        chunks = [chunk_from_dict(obj) for obj in payload["chunks"]]
        vectors = payload["vectors"]
        return cls(chunks, vectors, embedder_type=embedder_type, embedder_model=embedder_model, dim=dim)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> List[RetrievedChunk]:
        qvec = self.embedder.embed(query)
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = cosine(qvec, vec)
            if score >= min_score:
                scored.append(RetrievedChunk(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
