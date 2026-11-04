"""Inspect collected GitHub issues: python -m scripts.inspect_dataset"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "issues.jsonl"


def inspect(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}. Run collection first.")
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Dataset is empty.")
    df = pd.DataFrame(records)
    required = {"issue_id", "title", "body", "text", "state"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print("\n===== DATASET OVERVIEW =====")
    print("Total issues:", len(df))
    print("Columns:", list(df.columns))
    print("\n===== ISSUE STATES =====")
    print(df["state"].value_counts().to_string())
    print("\n===== DATA QUALITY =====")
    print("Blank titles:", df["title"].fillna("").str.strip().eq("").sum())
    print("Blank bodies:", df["body"].fillna("").str.strip().eq("").sum())
    print("Duplicate IDs:", df["issue_id"].duplicated().sum())
    print("\n===== TEXT LENGTH =====")
    print(df["text"].str.len().describe().to_string())
    print("\n===== SAMPLE =====")
    print(df[["issue_id", "title", "state"]].head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    inspect()
