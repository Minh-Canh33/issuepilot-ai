
from fastapi.testclient import TestClient

from api import main


client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_rejects_short_question():
    response = client.post(
        "/ask",
        json={"question": "hi"},
    )

    assert response.status_code == 422


def test_ask_rejects_missing_question():
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_returns_answer(monkeypatch):
    # Replace the real model/database and Gemini with fake functions.
    monkeypatch.setattr(
        main,
        "get_rag_resources",
        lambda: (None, None, {}),
    )

    def fake_answer(question, model, collection, issue_by_id):
        assert question == "How do I fix SSL errors?"
        return "Check the certificate configuration."

    monkeypatch.setattr(
        main,
        "answer_question",
        fake_answer,
    )

    response = client.post(
        "/ask",
        json={"question": "How do I fix SSL errors?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Check the certificate configuration."
    }


def test_ask_handles_backend_error(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_rag_resources",
        lambda: (None, None, {}),
    )

    def fake_failure(*args, **kwargs):
        raise RuntimeError("Simulated backend failure")

    monkeypatch.setattr(
        main,
        "answer_question",
        fake_failure,
    )

    response = client.post(
        "/ask",
        json={"question": "Investigate SSL verification failure"},
    )

    assert response.status_code == 500
    assert "could not process" in response.json()["detail"]
