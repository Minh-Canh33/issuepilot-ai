
import json
from pathlib import Path

from scripts.tfidf_search import load_issues, DATA_PATH


OUTPUT_PATH = Path("data/evaluation/queries.jsonl")


def main():
    issues = load_issues(DATA_PATH)

    # Select a small, reproducible sample of issues
    candidates = sorted(
        issues,
        key=lambda issue: str(issue["issue_id"])
    )[:15]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for issue in candidates:
            record = {
                "query_id": f"q_{issue['issue_id']}",
                "source_issue_id": issue["issue_id"],
                "query": issue["title"],
                "relevant_issue_ids": [],
                "status": "unlabeled"
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Prepared {len(candidates)} queries")
    print(f"Saved to {OUTPUT_PATH}")
    print("All queries are UNLABELED.")


if __name__ == "__main__":
    main()
