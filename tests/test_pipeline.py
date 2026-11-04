import json
from pathlib import Path

import httpx

from scripts.collect_issues import fetch_issues, normalize_issue, save_jsonl
from scripts.inspect_dataset import inspect


def sample(number=1, **kwargs):
    issue = {
        "number": number,
        "title": f"Issue {number}",
        "body": "Useful description",
        "state": "open",
        "labels": [{"name": "bug"}],
        "html_url": f"https://github.com/test/repo/issues/{number}",
        "created_at": "2026-01-01T00:00:00Z",
    }
    issue.update(kwargs)
    return issue


def test_normalize_filters_prs_and_missing_text():
    assert normalize_issue(sample(pull_request={}), "test/repo") is None
    assert normalize_issue(sample(body=""), "test/repo") is None
    assert normalize_issue(sample(), "test/repo")["labels"] == ["bug"]


def test_fetch_mocked_pagination_and_jsonl(tmp_path: Path):
    def handle(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
        if page == 1:
            return httpx.Response(200, json=[sample(1), sample(2), sample(3, pull_request={})])
        return httpx.Response(200, json=[])

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        issues = fetch_issues("test", "repo", max_issues=10, client=client)
    assert len(issues) == 2
    out = tmp_path / "issues.jsonl"
    save_jsonl(issues, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    df = inspect(out)
    assert len(df) == 2
    assert df["issue_id"].is_unique
