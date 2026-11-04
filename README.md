# IssuePilot AI — Day 1: GitHub Issue Data Pipeline

A working starter for the **first day** of the 7-day IssuePilot project.
This is the data-collection stage, **not yet** a RAG or AI Agent application.

## Goal
Collect public GitHub issues, filter out pull requests, normalize issue fields,
write a local JSONL dataset, and inspect basic dataset quality.

## Requirements
- Windows PowerShell (commands below), Python **3.11** or **3.12**
- Internet connection to reach `api.github.com` for collection
- Optional GitHub token for higher API rate limits

## Windows setup
Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not installed, use `py -3.12` if available.

Edit `.env` to change `GITHUB_OWNER`, `GITHUB_REPO`, `MAX_ISSUES`, etc.
The `GITHUB_TOKEN` field may remain blank for public repositories.
Do **not** commit `.env` or share your token.

## Run
From the **root** folder:

```powershell
python -m scripts.collect_issues
python -m scripts.inspect_dataset
python -m pytest -q
```

Expected files:

- `data/processed/issues.jsonl` — actual downloaded records, one JSON per line
- GitHub Issues: `issue_id`, `repository`, `title`, `body`, `text`, `state`, `labels`, `url`, `created_at`

The number collected depends on the selected repository, issue content, and available API pages.
If API returns 403/rate-limit, optionally set a token or retry after your rate limit resets.
If 404, verify the repository name and access. The script skips PRs and entries without meaningful body text.

## Offline tests
`pytest` uses a mocked GitHub API, so **tests do not need Internet**.
The data collector itself needs Internet; there is no fabricated sample data presented as GitHub results.

## Next days
- Day 2: TF-IDF + cosine similarity baseline and labeled retrieval queries.
- Day 3: sentence embeddings, ChromaDB, Recall@5 and MRR@10.
- Day 4: RAG + LLM tool calling to GitHub API.
- Day 5–7: FastAPI, UI, evaluation, packaging, demo and CV.

## Notes
- Only fetch public issues you are allowed to use; review the relevant repository's terms and remove sensitive data when publishing.
- GitHub REST API: https://docs.github.com/en/rest/issues/issues
