
import csv
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from scripts.tfidf_search import (
    DATA_PATH,
    build_index,
    load_issues,
    search as tfidf_search,
)

from scripts.build_vector_db import (
    COLLECTION_NAME,
    DB_PATH,
    MODEL_NAME,
)

from scripts.semantic_search import search as dense_search


QUERY_PATH = Path("data/evaluation/queries.jsonl")
OUTPUT_PATH = Path("data/evaluation/labeling_pool.csv")

TOP_K = 10


def load_queries():
    with QUERY_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    issues = load_issues(DATA_PATH)
    queries = load_queries()

    issue_lookup = {
        str(issue["issue_id"]): issue
        for issue in issues
    }

    tfidf_vectorizer, tfidf_vectors = build_index(issues)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    rows = []

    for query_record in queries:
        query_id = query_record["query_id"]
        source_id = str(query_record["source_issue_id"])
        query = query_record["query"]

        # Request one extra result to account for self-retrieval.
        lexical_results = tfidf_search(
            query,
            issues,
            tfidf_vectorizer,
            tfidf_vectors,
            top_k=TOP_K + 1,
        )

        semantic_results = dense_search(
            query,
            model,
            collection,
            top_k=TOP_K + 1,
        )

        candidates = {}

        for method, results in [
            ("tfidf", lexical_results),
            ("dense", semantic_results),
        ]:
            filtered = [
                result
                for result in results
                if str(result["issue_id"]) != source_id
            ][:TOP_K]

            for rank, result in enumerate(filtered, start=1):
                candidate_id = str(result["issue_id"])

                if candidate_id not in candidates:
                    candidates[candidate_id] = {
                        "query_id": query_id,
                        "source_issue_id": source_id,
                        "query": query,
                        "candidate_issue_id": candidate_id,
                        "candidate_title": result["title"],
                        "candidate_url": result["url"],
                        "tfidf_rank": "",
                        "dense_rank": "",
                        "label": "",
                    }

                candidates[candidate_id][f"{method}_rank"] = rank

        rows.extend(candidates.values())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id",
        "source_issue_id",
        "query",
        "candidate_issue_id",
        "candidate_title",
        "candidate_url",
        "tfidf_rank",
        "dense_rank",
        "label",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Queries:", len(queries))
    print("Candidate pairs:", len(rows))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
