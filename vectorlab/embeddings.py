"""Real embedding adapters; model downloads happen only when explicitly selected."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    model: str
    def encode(self, texts: list[str]) -> NDArray: ...


class SentenceEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = model
        self.encoder = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> NDArray:
        return np.asarray(self.encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                                             show_progress_bar=False), dtype=np.float64)



def create_embedder(backend: str = "sentence-transformers", model: str | None = None) -> Embedder:
    if backend != "sentence-transformers":
        raise ValueError("only sentence-transformers embeddings are supported")
    return SentenceEmbedder(model or "sentence-transformers/all-MiniLM-L6-v2")
