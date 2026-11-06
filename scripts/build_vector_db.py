
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from scripts.tfidf_search import DATA_PATH, load_issues


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = Path("data/chroma")
COLLECTION_NAME = "github_issues_dense_v1"


def issue_to_text(issue):
    """Prepare the same text representation for every issue."""
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()

    # Prevent very long issue bodies from dominating the input.
    return f"Title: {title}\nDescription: {body[:1500]}"


def main():
    issues = load_issues(DATA_PATH)
    print(f"Loaded {len(issues)} GitHub issues")

    # Run Transformer locally on CPU.
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    documents = [issue_to_text(issue) for issue in issues]

    print("Generating issue embeddings...")

    embeddings = model.encode(
        documents,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Embedding matrix shape:", embeddings.shape)

    # Create a persistent local vector database.
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Rebuild only our dedicated collection.
    # Previous embeddings in this collection will be replaced.
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[str(issue["issue_id"]) for issue in issues],
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=[
            {
                "title": issue["title"],
                "url": issue["url"],
            }
            for issue in issues
        ],
    )

    print("Stored issues:", collection.count())
    print("Vector database:", DB_PATH)


if __name__ == "__main__":
    main()
