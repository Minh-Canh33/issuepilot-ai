
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from scripts.tfidf_search import (
    DATA_PATH,
    build_index,
    load_issues,
    search,
)


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

QUERY_PATH = Path("data/evaluation/queries.jsonl")
OUTPUT_PATH = Path("data/evaluation/ai_labels.jsonl")

MAX_NEW_PAIRS = 5

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 5
REQUEST_DELAY_SECONDS = 15

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# =========================================================
# LOAD JSONL
# =========================================================

def read_jsonl(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


# =========================================================
# GEMINI LABELING
# =========================================================

def ask_llm(client, source, candidate):
    """
    Ask Gemini whether two GitHub issues are related.

    Labels:
        y = related
        n = unrelated
        s = uncertain
    """

    prompt = f"""
You are an expert software engineer evaluating
related GitHub bug reports.

Your task is to determine whether the SOURCE ISSUE
and CANDIDATE ISSUE describe substantially the same
technical problem, likely root cause, or directly
applicable troubleshooting/fix path.

LABELING RULES:

- Return "y" if the issues are clearly related.
- Return "n" if the issues are clearly unrelated.
- Return "s" if the evidence is insufficient.

Important:
- The same exception name alone is NOT enough.
- The same keyword alone is NOT enough.
- The same library alone is NOT enough.
- Focus on actual technical problems and likely causes.
- Do not invent missing information.

Return a JSON object containing:
- "label": "y", "n", or "s"
- "reason": a short technical explanation

SOURCE ISSUE:
Title: {source["title"]}
Body: {(source.get("body") or "")[:4500]}

CANDIDATE ISSUE:
Title: {candidate["title"]}
Body: {(candidate.get("body") or "")[:4500]}
"""

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }

    response = None

    # Retry on temporary server errors and network failures.
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(
                url,
                headers={
                    "x-goog-api-key": API_KEY,
                },
                json=payload,
            )

        except httpx.RequestError as exc:
            if attempt == MAX_RETRIES - 1:
                raise

            wait_seconds = RETRY_BASE_SECONDS * (2 ** attempt)

            print(f"Network error: {exc}")
            print(f"Retrying in {wait_seconds}s...")

            time.sleep(wait_seconds)
            continue

        if response.status_code not in RETRYABLE_STATUS_CODES:
            break

        if attempt == MAX_RETRIES - 1:
            break

        # Respect Retry-After when it contains a valid number.
        retry_after = response.headers.get("Retry-After")

        try:
            wait_seconds = (
                max(0.0, float(retry_after))
                if retry_after is not None
                else RETRY_BASE_SECONDS * (2 ** attempt)
            )
        except ValueError:
            wait_seconds = RETRY_BASE_SECONDS * (2 ** attempt)

        print(
            f"Gemini API returned {response.status_code}. "
            f"Retrying in {wait_seconds:g}s..."
        )

        time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError("No response received from Gemini API.")

    # Show useful API error details without exposing the key.
    if response.is_error:
        try:
            error = response.json().get("error", {})
            print("HTTP status:", response.status_code)
            print("API message:", error.get("message", ""))

        except (ValueError, AttributeError):
            print("API returned a non-JSON error response.")

    response.raise_for_status()

    # Parse the structured response.
    data = response.json()
    answer = data["candidates"][0]["content"]["parts"][0]["text"]

    result = json.loads(answer)

    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object.")

    if result.get("label") not in {"y", "n", "s"}:
        raise ValueError(
            f"Invalid label returned by Gemini: {result.get('label')}"
        )

    if not isinstance(result.get("reason"), str):
        raise ValueError("Gemini response is missing a valid reason.")

    return result


# =========================================================
# MAIN LABELING PIPELINE
# =========================================================

def main():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Check your .env file."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Issue dataset not found: {DATA_PATH}"
        )

    if not QUERY_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation queries not found: {QUERY_PATH}"
        )

    issues = load_issues(DATA_PATH)
    queries = read_jsonl(QUERY_PATH)
    existing_labels = read_jsonl(OUTPUT_PATH)

    print(f"Loaded {len(issues)} GitHub issues")
    print(f"Loaded {len(queries)} evaluation queries")
    print(f"Existing AI labels: {len(existing_labels)}")
    print(f"Gemini model: {MODEL}")

    # Find issues by their IDs.
    issue_lookup = {
        str(issue["issue_id"]): issue
        for issue in issues
    }

    # Avoid labeling the same pair twice.
    completed_pairs = {
        (
            str(record["source_issue_id"]),
            str(record["candidate_issue_id"]),
        )
        for record in existing_labels
    }

    # Build TF-IDF retrieval index.
    vectorizer, document_vectors = build_index(issues)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0

    with httpx.Client(timeout=90) as client:
        with OUTPUT_PATH.open("a", encoding="utf-8") as output_file:

            for query_record in queries:
                source_id = str(query_record["source_issue_id"])
                source_issue = issue_lookup.get(source_id)

                if source_issue is None:
                    print(f"Skipping missing source issue #{source_id}")
                    continue

                results = search(
                    query=query_record["query"],
                    issues=issues,
                    vectorizer=vectorizer,
                    document_vectors=document_vectors,
                    top_k=11,
                )

                # Remove the query issue itself.
                candidates = [
                    result
                    for result in results
                    if str(result["issue_id"]) != source_id
                ][:10]

                for candidate_result in candidates:
                    candidate_id = str(candidate_result["issue_id"])

                    pair = (source_id, candidate_id)

                    if pair in completed_pairs:
                        continue

                    candidate_issue = issue_lookup.get(candidate_id)

                    if candidate_issue is None:
                        continue

                    try:
                        prediction = ask_llm(
                            client,
                            source_issue,
                            candidate_issue,
                        )

                    except (
                        httpx.HTTPError,
                        ValueError,
                        KeyError,
                        IndexError,
                        RuntimeError,
                    ) as exc:
                        print("API/response error:", exc)
                        print("Stopping. Existing labels are saved.")
                        return

                    output_record = {
                        "query_id": query_record["query_id"],
                        "source_issue_id": source_issue["issue_id"],
                        "candidate_issue_id": candidate_issue["issue_id"],
                        "candidate_title": candidate_issue["title"],
                        "candidate_url": candidate_issue["url"],
                        "label": prediction["label"],
                        "reason": prediction["reason"],
                        "label_source": "llm",
                        "reviewed_by_human": False,
                        "model": MODEL,
                    }

                    output_file.write(
                        json.dumps(
                            output_record,
                            ensure_ascii=False,
                        ) + "\n"
                    )

                    # Save each result immediately.
                    output_file.flush()

                    completed_pairs.add(pair)
                    processed += 1

                    print(
                        f"#{source_id} vs #{candidate_id}: "
                        f"{prediction['label']} | "
                        f"{prediction['reason']}"
                    )

                    if processed >= MAX_NEW_PAIRS:
                        print(
                            f"\nSaved {processed} new AI labels "
                            f"to {OUTPUT_PATH}"
                        )
                        return

                    # Avoid sending requests too quickly.
                    time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nFinished. Saved {processed} new AI labels.")


if __name__ == "__main__":
    main()
