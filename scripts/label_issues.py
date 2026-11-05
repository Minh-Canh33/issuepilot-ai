
import json
from pathlib import Path

from scripts.tfidf_search import (
    DATA_PATH,
    build_index,
    load_issues,
    search,
)


QUERY_PATH = Path("data/evaluation/queries.jsonl")


def load_queries(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_queries(queries, path):
    """Save labels safely using a temporary file."""
    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        for record in queries:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    temp_path.replace(path)


def show_issue(issue):
    print(f"\nIssue #{issue['issue_id']}")
    print("Title:", issue["title"])
    print("URL:", issue["url"])

    body = issue.get("body") or ""
    preview = body[:1200].replace("\r", "")

    print("\nDescription:")
    print(preview if preview.strip() else "(Empty)")
    if len(body) > 1200:
        print("... (truncated)")


def main():
    if not DATA_PATH.exists() or not QUERY_PATH.exists():
        print("Missing dataset or evaluation queries.")
        print("Run prepare_eval.py first.")
        return

    issues = load_issues(DATA_PATH)
    queries = load_queries(QUERY_PATH)

    issue_lookup = {
        str(issue["issue_id"]): issue
        for issue in issues
    }

    vectorizer, vectors = build_index(issues)

    print(f"Loaded {len(issues)} issues")
    print(f"Loaded {len(queries)} evaluation queries")

    for record in queries:
        if record.get("status") != "unlabeled":
            continue

        source_id = str(record["source_issue_id"])
        source = issue_lookup.get(source_id)

        if source is None:
            print(f"Source issue {source_id} not found.")
            continue

        print("\n" + "=" * 65)
        print("QUERY:", record["query"])
        print("=" * 65)

        print("\nORIGINAL ISSUE")
        show_issue(source)

        # Retrieve one extra candidate because the query
        # may retrieve its own source issue.
        results = search(
            query=record["query"],
            issues=issues,
            vectorizer=vectorizer,
            document_vectors=vectors,
            top_k=11,
        )

        candidates = [
            result
            for result in results
            if str(result["issue_id"]) != source_id
        ][:10]

        if not candidates:
            print("No TF-IDF candidates found.")
            record["status"] = "needs_review"
            save_queries(queries, QUERY_PATH)
            continue

        relevant_ids = {
            str(i)
            for i in record.get("relevant_issue_ids", [])
        }

        judged_ids = {
            str(i)
            for i in record.get("judged_issue_ids", [])
        }

        quit_requested = False

        for rank, result in enumerate(candidates, start=1):
            candidate_id = str(result["issue_id"])

            if candidate_id in judged_ids:
                continue

            candidate = issue_lookup[candidate_id]

            print(f"\n{'-' * 55}")
            print(f"TF-IDF Rank: {rank}")
            print(f"Similarity: {result['score']:.4f}")

            show_issue(candidate)

            while True:
                answer = input(
                    "\nRelated? [y=yes / n=no / s=skip / q=quit]: "
                ).strip().lower()

                if answer in {"y", "n", "s", "q"}:
                    break

                print("Enter y, n, s, or q.")

            if answer == "q":
                quit_requested = True
                break

            if answer == "s":
                continue

            judged_ids.add(candidate_id)

            if answer == "y":
                relevant_ids.add(candidate_id)
            else:
                relevant_ids.discard(candidate_id)

            # Preserve the type used by the source dataset.
            id_by_string = {
                str(issue["issue_id"]): issue["issue_id"]
                for issue in issues
            }

            record["relevant_issue_ids"] = [
                id_by_string[i]
                for i in sorted(relevant_ids)
            ]
            record["judged_issue_ids"] = [
                id_by_string[i]
                for i in sorted(judged_ids)
            ]

            # Labels are provisional: only TF-IDF candidates
            # have been reviewed.
            record["status"] = "needs_review"
            save_queries(queries, QUERY_PATH)

        if quit_requested:
            print("\nProgress saved. Exiting.")
            return

        record["status"] = "needs_review"
        save_queries(queries, QUERY_PATH)

    print("\nFinished current TF-IDF labeling pass.")
    print("Labels still need review using Dense Retrieval.")


if __name__ == "__main__":
    main()
