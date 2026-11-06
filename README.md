
# IssuePilot AI

**GitHub Issue Retrieval & RAG Assistant**

IssuePilot AI is a Python-based project designed to help developers discover related GitHub issues and, in future iterations, generate evidence-grounded troubleshooting suggestions using Retrieval-Augmented Generation (RAG).

The project is developed incrementally, with each feature implemented, tested, and documented.

## Current Implementation

IssuePilot AI currently provides:

- **GitHub Issue Data Collection:** Fetching, processing, and validating real GitHub issues.
- **TF-IDF Retrieval:** Finding related issues using lexical similarity.
- **Semantic Retrieval:** Using Sentence Transformers and ChromaDB to search issues by semantic similarity.
- **Evaluation Preparation:** Preparing evaluation queries and collecting preliminary relevance labels.
- **Automated Testing:** Testing data collection and retrieval components.

The RAG-powered analysis system and application interface are planned for future iterations.

## Features

### 1. GitHub Issue Data Collection

- Fetches real GitHub issues using the GitHub REST API.
- Supports authentication through environment variables.
- Handles paginated API responses.
- Filters out pull requests.
- Extracts and normalizes relevant issue fields.
- Saves processed issues locally in JSONL format.
- Provides dataset inspection and validation.

### 2. TF-IDF Issue Retrieval

Implemented a lexical retrieval baseline using **TF-IDF vectorization and cosine similarity**.

**Implementation details:**

- Indexed **300 GitHub issues** from the `psf/requests` repository.
- Built a TF-IDF document matrix with **7,556 features**.
- Implemented Top-5 issue retrieval with similarity scores, titles, and GitHub URLs.
- Added automated retrieval tests.

**Run TF-IDF Retrieval:**

```bash
python -m scripts.tfidf_search
```

### 3. Semantic Issue Retrieval

Implemented dense semantic retrieval using **Sentence Transformers and ChromaDB**.

**Implementation details:**

- Uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` model.
- Runs embedding inference locally on CPU.
- Generates **384-dimensional embeddings** for GitHub issues.
- Uses issue titles and truncated descriptions as embedding input.
- Stores issue embeddings in a persistent ChromaDB collection.
- Uses cosine distance for vector similarity search.
- Retrieves Top-5 related issues with similarity scores and source URLs.
- Includes unit tests for semantic search functionality.

**Build the vector database:**

```bash
python -m scripts.build_vector_db
```

**Run Semantic Search:**

```bash
python -m scripts.semantic_search
```

The vector database is stored locally under `data/chroma/` and does not need to be committed to GitHub.

### 4. Retrieval Comparison

Both TF-IDF and Semantic Search were tested with the same query:

**Input query:**

```text
SSL certificate verification error
```

**TF-IDF Results:**

| Rank | GitHub Issue                                        | Similarity |
| ---- | --------------------------------------------------- | ---------: |
| 1    | [#6832](https://github.com/psf/requests/issues/6832) |     0.3725 |
| 2    | [#7547](https://github.com/psf/requests/issues/7547) |     0.3290 |
| 3    | [#6742](https://github.com/psf/requests/issues/6742) |     0.2513 |
| 4    | [#6886](https://github.com/psf/requests/issues/6886) |     0.2435 |
| 5    | [#6892](https://github.com/psf/requests/issues/6892) |     0.2138 |

**Semantic Search Results:**

| Rank | GitHub Issue                                        | Similarity |
| ---- | --------------------------------------------------- | ---------: |
| 1    | [#6969](https://github.com/psf/requests/issues/6969) |     0.6428 |
| 2    | [#6729](https://github.com/psf/requests/issues/6729) |     0.5882 |
| 3    | [#7547](https://github.com/psf/requests/issues/7547) |     0.5801 |
| 4    | [#6728](https://github.com/psf/requests/issues/6728) |     0.5607 |
| 5    | [#6893](https://github.com/psf/requests/issues/6893) |     0.5372 |

**Initial observations:**

- The two retrieval methods produced different rankings.
- Only one issue, **#7547**, appeared in both Top-5 result sets.
- Dense retrieval returned several issues involving SSL certificate validation and self-signed certificates.
- Similarity scores from TF-IDF and Dense Retrieval should not be compared directly because they use different representations.

These results demonstrate retrieval behavior, not measured retrieval accuracy. A formal comparison requires verified relevance labels.

### 5. Evaluation Preparation

An initial set of **15 evaluation queries** has been prepared.

The project includes:

- Query preparation with `prepare_eval.py`.
- Manual relevance labeling with `label_issues.py`.
- Experimental LLM-assisted labeling with `auto_label.py`.
- Separate storage of preliminary AI-generated labels.

**Current status:**

- Ground-truth labeling is incomplete.
- LLM-generated labels require human verification.
- Recall@5, MRR@10, and retrieval latency benchmarks have not yet been completed.

Evaluation will compare both retrieval methods using the same queries and a shared pool of reviewed candidates.

## Dataset

The initial dataset contains **300 real GitHub issues** collected from the [`psf/requests`](https://github.com/psf/requests) repository.

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

These statistics describe the initial collected dataset, not retrieval or RAG performance.

## Tech Stack

**Currently implemented:**

- **Language:** Python 3.11
- **API Integration:** HTTPX, GitHub REST API
- **Data Processing:** Pandas, JSONL
- **Configuration:** python-dotenv
- **Lexical Retrieval:** Scikit-learn, TF-IDF, Cosine Similarity
- **Dense Retrieval:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
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
│   ├── build_vector_db.py
│   ├── semantic_search.py
│   ├── prepare_eval.py
│   ├── label_issues.py
│   └── auto_label.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_tfidf.py
│   └── test_semantic_search.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   └── chroma/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Local datasets, vector databases, Python environments, and credentials are excluded from version control.

Evaluation artifacts should be reviewed before publication.

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

The first use of Sentence Transformers may download the pretrained model from Hugging Face.

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure the GitHub repository and authentication token as needed.

For experimental LLM-assisted labeling, configure a Gemini API key and an available model.

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

### 8. Build the Semantic Vector Database

```bash
python -m scripts.build_vector_db
```

This step generates issue embeddings and stores them in ChromaDB.

### 9. Run Semantic Retrieval

```bash
python -m scripts.semantic_search
```

Enter a bug report title or description to retrieve the Top-5 semantically similar issues.

### 10. Run Automated Tests

```bash
python -m pytest -q
```

The previously verified test result was **4 tests passed** before the semantic retrieval tests were added. The updated full test suite should be run again to confirm its current status.

## Planned Features

The following components are **not yet completed**:

- Comparative evaluation using Recall@5, MRR@10, and latency.
- Title-only versus title-and-body embedding ablation experiments.
- LLM-powered RAG issue analysis with source references.
- FastAPI backend for retrieval and analysis endpoints.
- Streamlit user interface.
- End-to-end testing and reproducible benchmark reporting.

## Project Goal

The goal of IssuePilot AI is to build an end-to-end AI application that combines real-world data collection, information retrieval, LLM-based analysis, and measurable evaluation.

The intended system will help developers discover related historical GitHub issues and generate evidence-grounded troubleshooting suggestions using RAG.

Performance improvements and benchmark metrics will be published only after they have been measured and verified through experiments.
