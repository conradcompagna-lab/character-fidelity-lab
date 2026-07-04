from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Iterable, List

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.IGNORECASE)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "for", "on", "by", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "its", "from",
    "at", "into", "about", "can", "could", "should", "would", "will", "not", "no", "do", "does"
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, text: str) -> List[float]:
        counts = Counter(tokenize(text))
        vec = [0.0] * self.dim
        if not counts:
            return vec
        for token, count in counts.items():
            vec[self._index(token)] += 1.0 + math.log(count)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector lengths differ")
    return sum(x * y for x, y in zip(a, b))


class GeminiEmbedder:
    def __init__(self, model: str = "gemini-embedding-001", dim: int | None = None):
        self.model = model
        self.dim = dim
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Install optional dependency: pip install google-genai") from exc
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts:
            return []
        from google.genai import types

        client = self._get_client()
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dim,
        )
        response = client.models.embed_content(model=self.model, contents=texts, config=config)
        return [list(embedding.values) for embedding in response.embeddings]

    def embed(self, text: str) -> List[float]:
        return self._embed_batch([text], task_type="RETRIEVAL_QUERY")[0]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return self._embed_batch(list(texts), task_type="RETRIEVAL_DOCUMENT")
