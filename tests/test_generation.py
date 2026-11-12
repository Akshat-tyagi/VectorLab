from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError, APIStatusError

from vectorlab import generation
from vectorlab.api import create_app
from test_rag_api import FakeEmbedder


def mock_client(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(generation, "Groq", factory)
    return factory, client


def test_generation_prompt_and_model(monkeypatch):
    factory, client = mock_client(monkeypatch)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" Answer [1] "))])
    generator = generation.GroqGenerator("test-secret")
    assert generator("What is a tree?", [{"text": "A tree has no cycles."}]) == "Answer [1]"
    payload = client.chat.completions.create.call_args.kwargs
    assert payload["model"] == "openai/gpt-oss-20b"
    assert "[1] A tree has no cycles." in payload["messages"][1]["content"]
    assert "What is a tree?" in payload["messages"][1]["content"]
    assert "untrusted" in payload["messages"][0]["content"]
    assert factory.call_args.kwargs["timeout"] == 60
    assert "test-secret" not in repr(generator)


@pytest.mark.parametrize("status,expected", [(401, "credentials"), (403, "credentials"),
    (429, "rate limit"), (500, "could not generate"), (400, "GROQ_MODEL")])
def test_provider_errors_are_safe(monkeypatch, status, expected):
    _, client = mock_client(monkeypatch)
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))
    client.chat.completions.create.side_effect = APIStatusError("test-secret", response=response, body={})
    with pytest.raises(generation.GenerationError, match=expected) as error:
        generation.GroqGenerator("test-secret")("question", [{"text": "evidence"}])
    assert "test-secret" not in str(error.value)


def test_connection_and_empty_answer(monkeypatch):
    _, client = mock_client(monkeypatch)
    client.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com"))
    with pytest.raises(generation.GenerationError, match="Cannot reach"):
        generation.GroqGenerator("test")("q", [])
    client.chat.completions.create.side_effect = None
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    with pytest.raises(generation.GenerationError, match="empty answer"):
        generation.GroqGenerator("test")("q", [])


def test_api_missing_key_and_abstention(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")
    client = TestClient(create_app(tmp_path / "docs.npz", FakeEmbedder()))
    assert client.post("/ask", json={"question": "tree"}).json()["abstained"]
    client.post("/documents", json={"title": "Trees", "text": "tree"})
    assert client.post("/search", json={"question": "tree"}).status_code == 200
    response = client.post("/ask", json={"question": "tree"})
    assert response.status_code == 503 and "GROQ_API_KEY" in response.json()["detail"]
    assert client.get("/status").json()["generation_configured"] is False


def test_api_cloud_answer_and_secret_not_exposed(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setenv("GROQ_MODEL", "configured-model")
    _, provider = mock_client(monkeypatch)
    provider.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [1]"))])
    client = TestClient(create_app(tmp_path / "docs.npz", FakeEmbedder()))
    client.post("/documents", json={"title": "Trees", "text": "tree"})
    response = client.post("/ask", json={"question": "tree"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Answer [1]"
    assert response.json()["model"] == "configured-model"
    assert response.json()["contexts"][0]["title"] == "Trees"
    assert "test-secret" not in client.get("/status").text
    provider.chat.completions.create.reset_mock()
    assert client.post("/ask", json={"question": "unrelated"}).json()["abstained"]
    provider.chat.completions.create.assert_not_called()
