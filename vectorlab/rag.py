"""Document retrieval with durable metadata and explicit no-evidence responses."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np

from .embeddings import Embedder, generate
from .indexes import HNSW


def chunks(text: str, size: int = 250, overlap: int = 30) -> list[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("require 0 <= overlap < size")
    words = text.split()
    result = []
    for start in range(0, len(words), size - overlap):
        result.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return result


class DocumentStore:
    """A snapshot contains vectors, HNSW graph, embedding identity, and text together."""

    def __init__(self, path: Path, embedder: Embedder, max_distance: float = 0.4):
        self.path, self.embedder, self.max_distance = path, embedder, max_distance
        self.index: HNSW | None = None
        self.documents: dict[int, dict] = {}
        self.next_id = 1
        self.lock = RLock()
        if path.exists():
            self.index = HNSW.load(path)
            with np.load(path, allow_pickle=False) as archive:
                state = json.loads(str(archive["metadata"]))
            if state["embedding_model"] != embedder.model:
                raise ValueError("snapshot embedding model differs from configured model")
            self.documents = {int(id): doc for id, doc in state["documents"].items()}
            self.next_id = state["next_id"]

    def save(self) -> None:
        if self.index is None:
            return
        self.index.save(self.path, {"documents": self.documents, "next_id": self.next_id,
                                   "embedding_model": self.embedder.model})

    def insert(self, title: str, text: str) -> list[int]:
        pieces = chunks(text)
        if not title.strip() or not pieces:
            raise ValueError("title and nonempty text required")
        vectors = np.asarray(self.embedder.encode(pieces), dtype=np.float64)
        if vectors.ndim != 2 or len(vectors) != len(pieces) or not np.isfinite(vectors).all() or vectors.shape[1] == 0:
            raise ValueError("invalid embedding output")
        with self.lock:
            if self.index is None:
                self.index = HNSW(vectors.shape[1])
            if vectors.shape[1] != self.index.dim:
                raise ValueError("embedding dimension changed")
            ids = []
            for position, (piece, vector) in enumerate(zip(pieces, vectors)):
                id = self.next_id
                self.next_id += 1
                self.index.insert(id, vector)
                self.documents[id] = {"id": id, "title": title, "text": piece,
                                      "chunk": position, "chunks": len(pieces)}
                ids.append(id)
            self.save()
            return ids

    def delete(self, id: int) -> bool:
        with self.lock:
            if id not in self.documents:
                return False
            self.index.delete(id)
            del self.documents[id]
            self.save()
            return True

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        if not question.strip() or k < 1:
            raise ValueError("question and positive k required")
        with self.lock:
            if not self.documents:
                return []
        query = self.embedder.encode([question])[0]
        with self.lock:
            hits = self.index.search(query, k, ef=max(50, k))
            return [{**self.documents[id], "distance": distance} for id, distance in hits
                    if distance <= self.max_distance]

    def ask(self, question: str, k: int = 3, generator: Callable = generate) -> dict:
        contexts = self.retrieve(question, k)
        if not contexts:
            return {"answer": "I don't know: no relevant document was found.",
                    "contexts": [], "abstained": True}
        return {"answer": generator(question, contexts), "contexts": contexts, "abstained": False}
