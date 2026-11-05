
# IssuePilot AI

**GitHub Issue Retrieval & RAG Assistant**

IssuePilot AI is a Python-based project designed to help developers discover related GitHub issues and, in future iterations, generate evidence-grounded troubleshooting suggestions using retrieval-augmented generation (RAG).

The project is developed incrementally, with each feature implemented, tested, and documented.

## Current Implementation

IssuePilot AI currently provides:

- **GitHub Issue Data Collection:** Fetching, processing, and validating real GitHub issues.
- **TF-IDF Issue Retrieval:** Finding related issues using lexical similarity.
- **Evaluation Preparation:** Generating evaluation queries and collecting preliminary relevance labels.
- **Automated Testing:** Testing the data pipeline and retrieval functionality.

Semantic vector search, RAG-powered analysis, and the application interface are planned for future iterations.

## Features

### GitHub Issue Data Collection

- Fetches GitHub issues using the GitHub REST API.
- Supports API authentication through environment variables.
- Handles paginated API responses.
- Extracts and normalizes relevant issue fields.
- Stores processed issues locally in JSONL format.
- Provides dataset inspection and validation.

### TF-IDF Issue Retrieval

Implemented a lexical retrieval baseline using **TF-IDF vectorization and cosine similarity**.

**Implementation details:**

- Indexed **300 real GitHub issues** from the `psf/requests` repository.
- Built a TF-IDF document matrix with **7,556 features**.
- Implemented Top-5 issue retrieval with similarity scores, issue titles, and GitHub URLs.
- Added automated retrieval tests.

**Retrieval pipeline:**

```text
GitHub Issues Dataset
        |
        v
Text Preprocessing
        |
        v
TF-IDF Vectorization
        |
        v
Cosine Similarity <--- User Query
        |
        v
Top-5 Related Issues
        |
        v
Issue Title + Similarity Score + GitHub URL
```

### Example Retrieval

**Input query:**

```text
SSL certificate verification error
```

**Actual retrieval results:**

| Rank | GitHub Issue                                        | Similarity |
| ---- | --------------------------------------------------- | ---------: |
| 1    | [#6832](https://github.com/psf/requests/issues/6832) |     0.3725 |
| 2    | [#7547](https://github.com/psf/requests/issues/7547) |     0.3290 |
| 3    | [#6742](https://github.com/psf/requests/issues/6742) |     0.2513 |
| 4    | [#6886](https://github.com/psf/requests/issues/6886) |     0.2435 |
| 5    | [#6892](https://github.com/psf/requests/issues/6892) |     0.2138 |

These scores represent **lexical similarity**, not retrieval accuracy or relevance judgments.

### Evaluation Preparation

An initial set of **15 evaluation queries** has been prepared.

The project includes:

- A script to generate evaluation queries.
- A manual issue relevance labeling tool.
- An experimental LLM-assisted labeling script using the Gemini API.
- Separate storage for preliminary AI-generated labels.

**Evaluation status:** Ground-truth labeling is incomplete. Recall@5, MRR@10, and latency benchmarks have not yet been reported.

LLM-generated labels are treated as suggestions and are not automatically considered verified ground truth.

## Dataset

The initial dataset contains **300 real GitHub issues**, collected from the [`psf/requests`](https://github.com/psf/requests) repository.

Each record includes:

- Issue ID and repository
- Title and description
- Issue state and labels
- GitHub issue URL
- Creation timestamp

### Data Validation

| Metric              | Result |
| ------------------- | -----: |
| Total issues        |    300 |
| Closed issues       |    263 |
| Open issues         |     37 |
| Missing titles      |      0 |
| Missing bodies      |      0 |
| Duplicate issue IDs |      0 |

These statistics describe the initial dataset and are not retrieval or RAG quality metrics.

## Tech Stack

**Currently implemented:**

- **Language:** Python 3.11
- **API Integration:** HTTPX, GitHub REST API
- **Data Processing:** Pandas, JSONL
- **Configuration:** python-dotenv
- **Information Retrieval:** Scikit-learn (TF-IDF, Cosine Similarity)
- **Testing:** pytest
- **Experimental LLM Labeling:** Gemini API via HTTPX

## Project Structure

```text
issuepilot-ai/
├── scripts/
│   ├── __init__.py
│   ├── collect_issues.py
│   ├── inspect_dataset.py
│   ├── tfidf_search.py
│   ├── prepare_eval.py
│   ├── label_issues.py
│   └── auto_label.py
├── tests/
│   ├── test_pipeline.py
│   └── test_tfidf.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Local datasets, environment variables, and API credentials should be excluded from version control. Evaluation artifacts can be reviewed before publication.

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

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure the GitHub repository and authentication token as needed.

For the experimental LLM-assisted labeling script, configure a Gemini API key and supported model.

Never commit `.env`, GitHub tokens, or Gemini API keys.

### 5. Collect GitHub issues

```bash
python -m scripts.collect_issues
```

### 6. Inspect the dataset

```bash
python -m scripts.inspect_dataset
```

### 7. Run TF-IDF Retrieval

```bash
python -m scripts.tfidf_search
```

Enter a bug report title or description to retrieve up to five related GitHub issues.

### 8. Run automated tests

```bash
python -m pytest -q
```

**Latest local test result:** 4 tests passed in 2.30 seconds.

## Planned Features

The following components are **planned and not yet implemented**:

- Sentence Transformer embeddings for semantic retrieval.
- ChromaDB vector storage and similarity search.
- Comparative retrieval benchmarks using Recall@5, MRR@10, and latency.
- Title-only versus title-and-body embedding experiments.
- LLM-powered RAG issue analysis with GitHub source references.
- FastAPI backend and Streamlit user interface.
- End-to-end testing, reproducible experiments, and benchmark documentation.

## Project Goal

The goal of IssuePilot AI is to build an end-to-end AI application that combines real-world data collection, information retrieval, LLM-based analysis, and measurable evaluation.

The intended system will help developers discover relevant historical GitHub issues and generate evidence-grounded troubleshooting suggestions.

Performance improvements and benchmark metrics will be published only after they have been measured and verified through experiments.
