import os

from fastapi.testclient import TestClient


os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["TRAIN_ENDPOINT_TOKEN"] = "secret-token"

from app import app


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_train_endpoint_forbidden_without_token():
    response = client.get("/train")
    assert response.status_code == 403


def test_summarize_langchain_returns_payload(monkeypatch):
    class FakeSummarizer:
        def summarize(self, **kwargs):
            return {
                "summary": "Résumé court.",
                "metadata": {
                    "strategy": "single_pass",
                    "chunk_count": 1,
                    "style": kwargs["style"],
                    "audience": kwargs["audience"],
                    "language": kwargs["language"],
                    "max_words": kwargs["max_words"],
                },
                "quality": None,
            }

    monkeypatch.setattr("app.get_langchain_summarizer", lambda: FakeSummarizer())

    response = client.post(
        "/summarize/langchain",
        json={
            "text": "Texte source de test.",
            "style": "executive",
            "audience": "recruiter",
            "language": "FR",
            "max_words": 100,
            "redact_pii": False,
            "run_quality_check": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]
    assert payload["metadata"]["language"] == "fr"
