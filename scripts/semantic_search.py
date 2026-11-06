
import chromadb
from sentence_transformers import SentenceTransformer

from scripts.build_vector_db import (
    COLLECTION_NAME,
    DB_PATH,
    MODEL_NAME,
)


def search(query, model, collection, top_k=5):
    """Search GitHub issues using semantic embeddings."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    matches = []

    for issue_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "issue_id": issue_id,
            "title": metadata["title"],
            "url": metadata["url"],
            "score": 1.0 - float(distance),
        })

    return matches


def main():
    client = chromadb.PersistentClient(path=str(DB_PATH))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        print("Vector index not found. Run build_vector_db first.")
        return

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Loaded vector index:", collection.count(), "issues")

    while True:
        query = input(
            "\nEnter issue description (or 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(query, model, collection, top_k=5)

        print("\n===== TOP 5 SEMANTIC RESULTS =====")

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Issue: #{result['issue_id']}")
            print(f"Title: {result['title']}")
            print(f"Score: {result['score']:.4f}")
            print(f"URL: {result['url']}")


if __name__ == "__main__":
    main()
