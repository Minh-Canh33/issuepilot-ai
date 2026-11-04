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
