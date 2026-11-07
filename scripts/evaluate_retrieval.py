
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/evaluation/labeling_pool.csv")
OUTPUT_PATH = Path("data/evaluation/benchmark_results.csv")


def evaluate_method(rows, rank_column, relevant_ids):
    """Compute pooled Recall@5 and MRR@10 for one query."""

    ranked = rows.dropna(subset=[rank_column]).copy()
    ranked = ranked.sort_values(rank_column)

    top5_ids = set(
        ranked.loc[
            ranked[rank_column] <= 5,
            "candidate_issue_id"
        ].astype(str)
    )

    recall5 = len(top5_ids & relevant_ids) / len(relevant_ids)

    mrr10 = 0.0

    for _, row in ranked.iterrows():
        rank = int(row[rank_column])

        if rank > 10:
            break

        if str(row["candidate_issue_id"]) in relevant_ids:
            mrr10 = 1.0 / rank
            break

    return recall5, mrr10


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "query_id": str,
            "candidate_issue_id": str,
            "source_issue_id": str,
        },
    )

    required = {
        "query_id",
        "source_issue_id",
        "candidate_issue_id",
        "tfidf_rank",
        "dense_rank",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["label"] = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for column in ["tfidf_rank", "dense_rank"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df.duplicated(
        subset=["query_id", "candidate_issue_id"]
    ).any():
        raise ValueError("Duplicate query-candidate pairs found.")

    if (
        df["source_issue_id"]
        == df["candidate_issue_id"]
    ).any():
        raise ValueError("Self-retrieval found in evaluation pool.")

    results = []
    incomplete_queries = []
    no_positive_queries = []

    for query_id, group in df.groupby("query_id"):
        # Only fully reviewed queries are used.
        if not group["label"].isin(["y", "n"]).all():
            incomplete_queries.append(query_id)
            continue

        relevant_ids = set(
            group.loc[
                group["label"] == "y",
                "candidate_issue_id"
            ].astype(str)
        )

        # Recall is undefined when there are no
        # confirmed relevant issues in the pool.
        if not relevant_ids:
            no_positive_queries.append(query_id)
            continue

        tfidf_recall, tfidf_mrr = evaluate_method(
            group, "tfidf_rank", relevant_ids
        )

        dense_recall, dense_mrr = evaluate_method(
            group, "dense_rank", relevant_ids
        )

        results.append({
            "query_id": query_id,
            "relevant_count": len(relevant_ids),
            "tfidf_recall5": tfidf_recall,
            "tfidf_mrr10": tfidf_mrr,
            "dense_recall5": dense_recall,
            "dense_mrr10": dense_mrr,
        })

    print("Total queries:", df["query_id"].nunique())
    print("Evaluated queries:", len(results))
    print("Incomplete queries:", len(incomplete_queries))
    print("Queries without positive labels:", len(no_positive_queries))

    if not results:
        print(
            "\nNo evaluable queries yet. "
            "Review labels before benchmarking."
        )
        return

    report = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(OUTPUT_PATH, index=False)

    print("\n========== RETRIEVAL BENCHMARK ==========")

    for method in ["tfidf", "dense"]:
        recall = report[f"{method}_recall5"].mean()
        mrr = report[f"{method}_mrr10"].mean()

        print(f"\n{method.upper()}")
        print(f"Recall@5: {recall:.4f} ({recall:.2%})")
        print(f"MRR@10:   {mrr:.4f}")

    diff = (
        report["dense_recall5"].mean()
        - report["tfidf_recall5"].mean()
    )

    print(
        f"\nDense - TF-IDF Recall@5: "
        f"{diff * 100:+.2f} percentage points"
    )

    print("\nSaved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
