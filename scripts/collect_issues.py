"""Collect public GitHub issues into a local JSONL dataset.

Run from repository root: python -m scripts.collect_issues
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "issues.jsonl"
API_BASE = "https://api.github.com"


def normalize_issue(issue: dict[str, Any], repo: str) -> dict[str, Any] | None:
    """Skip pull requests and entries without usable title/body."""
    if "pull_request" in issue:
        return None
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    if not title or not body:
        return None
    return {
        "issue_id": issue["number"],
        "repository": repo,
        "title": title,
        "body": body,
        "text": f"{title}\n\n{body}",
        "state": issue.get("state", "unknown"),
        "labels": [
            label["name"] if isinstance(label, dict) else str(label)
            for label in issue.get("labels", [])
        ],
        "url": issue.get("html_url", ""),
        "created_at": issue.get("created_at", ""),
    }


def fetch_issues(
    owner: str,
    repo: str,
    token: str | None = None,
    max_issues: int = 300,
    max_pages: int = 15,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Collect up to max_issues records, respecting pagination and timeouts.

    Passing a client enables offline mock testing.
    """
    if not owner or not repo or max_issues < 1 or max_pages < 1:
        raise ValueError("owner/repo must be non-empty; max_issues/max_pages >= 1")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "IssuePilot-Day1-Student-Project",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    collected: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()
    own_client = client is None
    if client is None:
        client = httpx.Client(headers=headers, timeout=30.0, follow_redirects=True)
    try:
        for page in range(1, max_pages + 1):
            response = client.get(
                f"{API_BASE}/repos/{owner}/{repo}/issues",
                params={"state": "all", "per_page": 100, "page": page},
                headers=headers,
            )
            response.raise_for_status()
            records = response.json()
            if not isinstance(records, list):
                raise ValueError("GitHub returned an unexpected JSON response")
            if not records:
                break

            for raw in records:
                issue = normalize_issue(raw, f"{owner}/{repo}")
                if issue is None or issue["issue_id"] in seen_numbers:
                    continue
                seen_numbers.add(issue["issue_id"])
                collected.append(issue)
                if len(collected) >= max_issues:
                    break

            print(f"Page {page}: {len(collected)} valid issues collected")
            if len(collected) >= max_issues or len(records) < 100:
                break
    finally:
        if own_client:
            client.close()
    return collected


def save_jsonl(issues: list[dict[str, Any]], output: Path = OUTPUT_PATH) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for issue in issues:
            handle.write(json.dumps(issue, ensure_ascii=False) + "\n")


def main() -> int:
    owner = os.getenv("GITHUB_OWNER", "psf").strip()
    repo = os.getenv("GITHUB_REPO", "requests").strip()
    token = os.getenv("GITHUB_TOKEN", "").strip() or None
    print(f"Collecting issues from {owner}/{repo} ...")
    try:
        issues = fetch_issues(
            owner=owner,
            repo=repo,
            token=token,
            max_issues=int(os.getenv("MAX_ISSUES", "300")),
            max_pages=int(os.getenv("MAX_PAGES", "15")),
        )
        if not issues:
            print("No valid issues found. Try another repository or increase MAX_PAGES.")
            return 1
        save_jsonl(issues)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        print(f"GitHub API HTTP {code}: {exc.response.text[:250]}", file=sys.stderr)
        if code == 403:
            print("Tip: API rate limit may be reached; set GITHUB_TOKEN in .env.", file=sys.stderr)
        return 1
    except (httpx.RequestError, ValueError, OSError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(issues)} issues -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
