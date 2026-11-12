"""FastAPI app factory. Run one worker: the store is protected by a process-local lock."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from .generation import GenerationError, GroqGenerator

from .embeddings import Embedder, create_embedder
from .rag import DocumentStore, SearchAlgorithm


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=200_000)


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=3, ge=1, le=100, strict=True)


class SearchInput(QueryInput):
    algorithm: SearchAlgorithm = "hnsw"


def create_app(path: Path | None = None, embedder: Embedder | None = None) -> FastAPI:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    generator = GroqGenerator.from_env()
    encoder = embedder or create_embedder(model=os.getenv("VECTORLAB_MODEL"))
    threshold = float(os.getenv("VECTORLAB_MAX_DISTANCE", "0.4"))
    if not 0 <= threshold <= 2:
        raise ValueError("VECTORLAB_MAX_DISTANCE must lie in [0,2]")
    store = DocumentStore(path or Path(os.getenv("VECTORLAB_SNAPSHOT", "data/documents.npz")),
                          encoder, threshold)
    app = FastAPI(title="VectorLab NumPy", version="0.1.0",
                  description="Local document retrieval with a hand-built HNSW index")
    app.state.store = store
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(static / "index.html")

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/status")
    def status() -> dict:
        with store.lock:
            return {"documents": len(store.documents), "model": encoder.model,
                    "dims": store.index.dim if store.index else None,
                    "max_distance": threshold, "generation_provider": "groq",
                    "generation_model": generator.model, "generation_configured": generator.configured}

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
    def search(body: SearchInput) -> dict:
        return {"contexts": store.retrieve(body.question, body.k, body.algorithm),
                "algorithm": body.algorithm}

    @app.post("/ask")
    def ask(body: QueryInput) -> dict:
        try:
            result = store.ask(body.question, body.k, generator=generator)
            return {**result, "provider": "groq", "model": generator.model}
        except GenerationError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (OSError, KeyError) as exc:
            raise HTTPException(503, "embedding or generation service unavailable") from exc

    return app
