
from scripts.semantic_search import search
import numpy as np


class FakeModel:
    """Simulate a Sentence Transformer without loading a real model."""

    def encode(self, texts, normalize_embeddings=True):
        assert texts == ["SSL certificate error"]
        assert normalize_embeddings is True
        return np.array([[0.1, 0.2, 0.3]])


class FakeCollection:
    """Simulate a ChromaDB collection."""

    def count(self):
        return 2

    def query(self, query_embeddings, n_results, include):
        assert query_embeddings == [[0.1, 0.2, 0.3]]
        assert n_results == 2
        assert include == ["metadatas", "distances"]

        return {
            "ids": [["101", "102"]],
            "metadatas": [[
                {
                    "title": "SSL certificate verification failed",
                    "url": "https://github.com/example/issues/101",
                },
                {
                    "title": "Documentation update",
                    "url": "https://github.com/example/issues/102",
                },
            ]],
            "distances": [[0.1, 0.8]],
        }


def test_semantic_search_returns_ranked_results():
    model = FakeModel()
    collection = FakeCollection()

    results = search(
        query="SSL certificate error",
        model=model,
        collection=collection,
        top_k=5,
    )

    assert len(results) == 2
    assert results[0]["issue_id"] == "101"
    assert results[0]["score"] == 0.9
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["url"].endswith("/101")


def test_semantic_search_limits_results_to_collection_size():
    model = FakeModel()
    collection = FakeCollection()

    results = search(
        query="SSL certificate error",
        model=model,
        collection=collection,
        top_k=10,
    )

    assert len(results) == 2
