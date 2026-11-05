
from scripts.tfidf_search import build_index, search


def test_search_returns_relevant_issue():
    issues = [
        {
            "issue_id": 1,
            "title": "Connection timeout",
            "text": "HTTP connection timeout error",
            "url": "https://github.com/example/issues/1",
        },
        {
            "issue_id": 2,
            "title": "SSL verification",
            "text": "SSL certificate verification failed",
            "url": "https://github.com/example/issues/2",
        },
        {
            "issue_id": 3,
            "title": "Documentation update",
            "text": "Update project documentation",
            "url": "https://github.com/example/issues/3",
        },
    ]

    vectorizer, vectors = build_index(issues)

    results = search(
        query="HTTP connection timeout",
        issues=issues,
        vectorizer=vectorizer,
        document_vectors=vectors,
        top_k=2,
    )

    assert len(results) <= 2
    assert results[0]["issue_id"] == 1
    assert results[0]["score"] > 0


def test_search_with_unknown_words():
    issues = [
        {
            "issue_id": 1,
            "title": "Connection timeout",
            "text": "HTTP connection timeout",
            "url": "https://github.com/example/issues/1",
        },
        {
            "issue_id": 2,
            "title": "SSL certificate",
            "text": "SSL certificate failed",
            "url": "https://github.com/example/issues/2",
        },
    ]

    vectorizer, vectors = build_index(issues)

    results = search(
        query="quantum banana spaceship",
        issues=issues,
        vectorizer=vectorizer,
        document_vectors=vectors,
    )

    assert results == []
