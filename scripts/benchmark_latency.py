
import json
import statistics
import time
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
WARMUP_RUNS = 3
MEASURED_RUNS = 10
TOP_K = 5


def percentile(values, percent):
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def benchmark(search_fn, queries):
    # Warm-up để giảm ảnh hưởng lần gọi đầu.
    for i in range(WARMUP_RUNS):
        search_fn(queries[i % len(queries)])

    durations_ms = []

    for _ in range(MEASURED_RUNS):
        for query in queries:
            start = time.perf_counter()
            search_fn(query)
            duration_ms = (time.perf_counter() - start) * 1000
            durations_ms.append(duration_ms)

    return {
        "calls": len(durations_ms),
        "mean_ms": statistics.mean(durations_ms),
        "p50_ms": percentile(durations_ms, 50),
        "p95_ms": percentile(durations_ms, 95),
    }


def main():
    with QUERY_PATH.open(encoding="utf-8") as file:
        queries = [
            json.loads(line)["query"]
            for line in file if line.strip()
        ]

    if not queries:
        raise ValueError("No evaluation queries found.")

    # Setup không tính vào retrieval latency.
    issues = load_issues(DATA_PATH)
    vectorizer, vectors = build_index(issues)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    def run_tfidf(query):
        return tfidf_search(
            query,
            issues,
            vectorizer,
            vectors,
            top_k=TOP_K,
        )

    def run_dense(query):
        return dense_search(
            query,
            model,
            collection,
            top_k=TOP_K,
        )

    print("Benchmarking TF-IDF...")
    tfidf_result = benchmark(run_tfidf, queries)

    print("Benchmarking Dense...")
    dense_result = benchmark(run_dense, queries)

    print("\n========== LATENCY BENCHMARK ==========")
    print(f"Queries: {len(queries)}")
    print(f"Calls per method: {tfidf_result['calls']}")

    for name, result in [
        ("TF-IDF", tfidf_result),
        ("Dense", dense_result),
    ]:
        print(f"\n{name}")
        print(f"Mean: {result['mean_ms']:.2f} ms")
        print(f"P50:  {result['p50_ms']:.2f} ms")
        print(f"P95:  {result['p95_ms']:.2f} ms")

    output = Path("data/evaluation/latency_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "queries": len(queries),
                "warmup_runs": WARMUP_RUNS,
                "measured_runs": MEASURED_RUNS,
                "top_k": TOP_K,
                "hardware": "CPU",
                "tfidf": tfidf_result,
                "dense": dense_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
