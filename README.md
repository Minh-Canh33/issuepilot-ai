# IssuePilot AI

**GitHub Issue Retrieval & RAG Assistant**

IssuePilot AI is a Python-based project designed to help developers discover related GitHub issues and, in future iterations, generate evidence-grounded troubleshooting suggestions using retrieval-augmented generation (RAG).

The project is being developed incrementally, with each feature implemented, tested, and documented.

## Current Implementation

The current implementation focuses on building a reliable GitHub issue data collection pipeline.

### Features

- Fetches real GitHub issues using the GitHub REST API.
- Supports API authentication through environment variables.
- Handles paginated API responses.
- Extracts and normalizes relevant issue fields.
- Stores processed issue data locally for future retrieval experiments.
- Provides basic dataset inspection and validation.

## Dataset

The initial dataset contains **300 real GitHub issues** collected from the [`psf/requests`](https://github.com/psf/requests) repository.

The collected data includes:

- Issue ID and repository
- Title and description
- Issue state and labels
- GitHub issue URL
- Creation timestamp

### Initial Data Validation

| Metric              | Result |
| ------------------- | -----: |
| Total issues        |    300 |
| Closed issues       |    263 |
| Open issues         |     37 |
| Missing titles      |      0 |
| Missing bodies      |      0 |
| Duplicate issue IDs |      0 |

These statistics describe the initial collected dataset and are not retrieval or RAG performance metrics.

## Tech Stack

**Currently implemented:**

- Python 3.11
- HTTPX
- Pandas
- python-dotenv
- GitHub REST API

## Project Structure

```text
issuepilot-ai/
├── scripts/
│   ├── collect_issues.py
│   └── inspect_dataset.py
├── tests/
│   └── test_pipeline.py
├── data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Local datasets and credentials are excluded from version control.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Minh-Canh33/issuepilot-ai.git
cd issuepilot-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure the GitHub repository and personal access token as needed.

Never commit your `.env` file or API tokens.

### 5. Collect issues

```bash
python -m scripts.collect_issues
```

### 6. Inspect the dataset

```bash
python -m scripts.inspect_dataset
```

## Planned Features

The following components are planned and are **not yet implemented**:

- TF-IDF retrieval baseline
- Sentence Transformer embeddings and ChromaDB vector search
- Retrieval evaluation using Recall@5, MRR@10, and latency
- LLM-powered RAG issue analysis with source references
- FastAPI backend and Streamlit interface
- Experimental comparisons and benchmark reporting

## Project Goal

The goal is to build an end-to-end AI application that combines real-world data collection, information retrieval, LLM-based analysis, and measurable evaluation.

Performance claims and benchmark results will be added only after they have been verified through experiment

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
