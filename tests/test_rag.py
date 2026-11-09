
from scripts import rag_assistant as rag


SAMPLE_ISSUES = [
    {
        "issue_id": 7547,
        "title": "SSL verification documentation",
        "url": "https://github.com/psf/requests/issues/7547",
    }
]

ISSUE_BY_ID = {
    "7547": {
        "body": "Users need guidance on SSL verification errors."
    }
}


def test_build_context_includes_issue_body():
    context = rag.build_context(
        SAMPLE_ISSUES,
        ISSUE_BY_ID,
    )

    assert "Issue #7547" in context
    assert "SSL verification documentation" in context
    assert "Users need guidance" in context
    assert "https://github.com/psf/requests/issues/7547" in context


def test_build_prompt_includes_question_and_evidence():
    prompt = rag.build_prompt(
        "How do I investigate SSL errors?",
        "Issue #7547: SSL verification documentation",
    )

    assert "How do I investigate SSL errors?" in prompt
    assert "Issue #7547" in prompt
    assert "insufficient" in prompt.lower()
    assert "cite" in prompt.lower()


def test_call_gemini_parses_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Answer from test Gemini"}
                            ]
                        }
                    }
                ]
            }

    def fake_post(url, **kwargs):
        assert "gemini-3.5-flash-lite" in url
        assert kwargs["headers"]["x-goog-api-key"] == "test-key"
        return FakeResponse()

    monkeypatch.setattr(rag.httpx, "post", fake_post)

    result = rag.call_gemini("Test prompt")

    assert result == "Answer from test Gemini"


def test_answer_question_uses_retrieved_issues(monkeypatch):
    monkeypatch.setattr(
        rag,
        "dense_search",
        lambda *args, **kwargs: SAMPLE_ISSUES,
    )

    def fake_gemini(prompt):
        assert "Users need guidance" in prompt
        return "Grounded test answer"

    monkeypatch.setattr(rag, "call_gemini", fake_gemini)

    result = rag.answer_question(
        "SSL error",
        model=None,
        collection=None,
        issue_by_id=ISSUE_BY_ID,
    )

    assert result == "Grounded test answer"


def test_no_results_does_not_call_gemini(monkeypatch):
    monkeypatch.setattr(
        rag,
        "dense_search",
        lambda *args, **kwargs: [],
    )

    def should_not_be_called(prompt):
        raise AssertionError("Gemini should not be called")

    monkeypatch.setattr(
        rag,
        "call_gemini",
        should_not_be_called,
    )

    result = rag.answer_question(
        "Unknown problem",
        model=None,
        collection=None,
        issue_by_id=ISSUE_BY_ID,
    )

    assert "No relevant GitHub issues" in result
