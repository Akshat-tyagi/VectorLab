"""FastAPI app factory. Run one worker: the store is protected by a process-local lock."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .embeddings import Embedder, create_embedder
from .rag import DocumentStore


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=200_000)


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=3, ge=1, le=100, strict=True)


def create_app(path: Path | None = None, embedder: Embedder | None = None) -> FastAPI:
    encoder = embedder or create_embedder(os.getenv("VECTORLAB_BACKEND", "sentence-transformers"),
                                         os.getenv("VECTORLAB_MODEL"))
    threshold = float(os.getenv("VECTORLAB_MAX_DISTANCE", "0.4"))
    if not 0 <= threshold <= 2:
        raise ValueError("VECTORLAB_MAX_DISTANCE must lie in [0,2]")
    store = DocumentStore(path or Path(os.getenv("VECTORLAB_SNAPSHOT", "data/documents.npz")),
                          encoder, threshold)
    app = FastAPI(title="VectorLab NumPy", version="0.1.0",
                  description="Local document retrieval with a hand-built HNSW index")
    app.state.store = store

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/status")
    def status() -> dict:
        with store.lock:
            return {"documents": len(store.documents), "model": encoder.model,
                    "dims": store.index.dim if store.index else None,
                    "max_distance": threshold}

    @app.post("/documents", status_code=201)
    def insert(body: DocumentInput) -> dict:
        return {"ids": store.insert(body.title, body.text)}

    @app.get("/documents")
    def documents() -> list[dict]:
        with store.lock:
            return list(store.documents.values())

    @app.delete("/documents/{id}")
    def delete(id: int) -> dict:
        if not store.delete(id):
            raise HTTPException(404, "document chunk not found")
        return {"deleted": id}

    @app.post("/search")
    def search(body: QueryInput) -> dict:
        return {"contexts": store.retrieve(body.question, body.k)}

    @app.post("/ask")
    def ask(body: QueryInput) -> dict:
        try:
            return store.ask(body.question, body.k)
        except (OSError, KeyError) as exc:
            raise HTTPException(503, "embedding or generation service unavailable") from exc

    return app
