import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorlab.api import create_app
from vectorlab.rag import DocumentStore, chunks


class FakeEmbedder:
    """Deterministic test fixture only; never used in dataset or benchmark results."""
    model = "test-only"

    def encode(self, texts):
        return np.array([[1, 0] if "tree" in text else [0, 1] for text in texts], dtype=float)


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(500))
    result = chunks(text)
    assert len(result) == 3
    assert result[0].split()[-30:] == result[1].split()[:30]
    assert result[-1].endswith("499")
    assert chunks("  ") == []
    with pytest.raises(ValueError):
        chunks(text, 10, 10)


def test_rag_restart_and_abstention(tmp_path):
    path = tmp_path / "docs.npz"
    store = DocumentStore(path, FakeEmbedder())
    assert store.ask("tree")["abstained"]
    ids = store.insert("Trees", "A tree is a connected graph without cycles.")
    assert store.retrieve("tree")[0]["id"] == ids[0]
    answer = store.ask("tree", generator=lambda q, contexts: "Known answer [1]")
    assert not answer["abstained"]
    def forbidden(*args):
        pytest.fail("generation must not run without evidence")
    assert store.ask("unrelated", generator=forbidden)["abstained"]
    restored = DocumentStore(path, FakeEmbedder())
    assert restored.retrieve("tree") == store.retrieve("tree")
    assert restored.delete(ids[0])
    assert not restored.delete(ids[0])
    assert DocumentStore(path, FakeEmbedder()).retrieve("tree") == []
    restored.insert("Again", "tree")
    assert restored.retrieve("tree")[0]["id"] > ids[0]


def test_api_validation_and_restart(tmp_path):
    path = tmp_path / "api.npz"
    client = TestClient(create_app(path, FakeEmbedder()))
    assert client.get("/status").status_code == 200
    assert client.post("/documents", json={"title": " ", "text": "tree"}).status_code == 422
    assert client.post("/documents", json={"title": "T", "text": "tree", "extra": 1}).status_code == 422
    for k in [0, -1, 101, "3", True]:
        assert client.post("/search", json={"question": "tree", "k": k}).status_code == 422
    response = client.post("/documents", json={"title": "T", "text": "tree"})
    assert response.status_code == 201
    id = response.json()["ids"][0]
    restarted = TestClient(create_app(path, FakeEmbedder()))
    assert restarted.post("/search", json={"question": "tree"}).json()["contexts"][0]["id"] == id
    assert restarted.delete(f"/documents/{id}").status_code == 200
    assert restarted.delete(f"/documents/{id}").status_code == 404
    assert restarted.post("/ask", json={"question": "tree"}).json()["abstained"]
