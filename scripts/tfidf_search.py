
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/processed/issues.jsonl")


def load_issues(path):
    issues = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                issues.append(json.loads(line))

    return issues


def build_index(issues):
    documents = [issue["text"] for issue in issues]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        sublinear_tf=True
    )

    document_vectors = vectorizer.fit_transform(documents)

    return vectorizer, document_vectors


def search(query, issues, vectorizer, document_vectors, top_k=5):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []

    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue

        issue = issues[idx]

        results.append({
            "issue_id": issue["issue_id"],
            "title": issue["title"],
            "score": float(scores[idx]),
            "url": issue["url"]
        })

        if len(results) >= top_k:
            break

    return results


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    issues = load_issues(DATA_PATH)
    print(f"Loaded {len(issues)} issues")

    vectorizer, document_vectors = build_index(issues)

    print("TF-IDF index created:", document_vectors.shape)

    while True:
        query = input("\nEnter issue description (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(
            query,
            issues,
            vectorizer,
            document_vectors,
            top_k=5
        )

        print("\n===== TOP 5 RELATED ISSUES =====")

        if not results:
            print("No matching issues found.")

        for rank, result in enumerate(results, 1):
            print(f"\nRank: {rank}")
            print(f"Issue: #{result['issue_id']}")
            print(f"Title: {result['title']}")
            print(f"Score: {result['score']:.4f}")
            print(f"URL: {result['url']}")


if __name__ == "__main__":
    main()
