# IssuePilot AI

**GitHub Issue Retrieval & RAG Assistant**

IssuePilot AI is an AI engineering portfolio project designed to help developers discover related GitHub issues, investigate reported bugs, and eventually generate evidence-grounded answers using Retrieval-Augmented Generation (RAG).

The project compares a traditional **TF-IDF retrieval baseline** with **semantic search using Sentence Transformers and ChromaDB**, evaluates retrieval quality and latency, and provides a foundation for an LLM-powered issue assistant.

**Project status:** In development — Days 1–4 implemented. RAG generation, API integration, and the interactive demo are planned for Days 5–7.

**Repository:** https://github.com/Minh-Canh33/issuepilot-ai

---

## 1. Problem Statement

Developers often need to search through existing GitHub issues before investigating a new bug report.

Traditional keyword-based search can miss related issues when developers describe similar problems using different terminology.

For example, one issue might mention an "SSL certificate verification error," while another describes a TLS handshake failure or a certificate authority configuration problem.

IssuePilot AI investigates whether semantic retrieval can help surface useful historical issues beyond keyword overlap.

The long-term goal is to build an assistant that:

- Collects and normalizes real GitHub issues.
- Retrieves potentially related historical reports.
- Compares lexical and semantic retrieval.
- Measures retrieval quality and latency.
- Generates issue-analysis suggestions grounded in retrieved sources.
- Shows supporting GitHub issue links.

The assistant is intended to support debugging and investigation, not automatically verify root causes or fix bugs.

## 2. System Architecture

The implemented retrieval pipeline and planned RAG extension follow this architecture:

```text
                   GitHub REST API
                          |
                          v
                 Issue Collection
                          |
                          v
               Data Normalization
                          |
                          v
                  JSONL Dataset
                    300 Issues
                          |
             +------------+------------+
             |                         |
             v                         v
      TF-IDF Retrieval          Dense Retrieval
      scikit-learn              Sentence Transformers
             |                         |
      Sparse Vectors             384-D Embeddings
             |                         |
      Cosine Similarity                v
             |                      ChromaDB
             |                         |
             +------------+------------+
                          |
                          v
                 Top-K GitHub Issues
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Evaluation Metrics       RAG Context
         (Implemented)           (Planned)
              |                       |
        Recall@5, MRR@10               v
        Retrieval Latency          Gemini LLM
                                      |
                                      v
                             Grounded AI Answer
                                      |
                                      v
                            FastAPI + Streamlit
                                  (Planned)
```

## 3. Technology Stack

| Component               | Technologies                            | Status            |
| ----------------------- | --------------------------------------- | ----------------- |
| Data collection         | Python, GitHub REST API, HTTPX          | Implemented       |
| Data processing         | pandas, JSONL                           | Implemented       |
| Keyword retrieval       | scikit-learn, TF-IDF, cosine similarity | Implemented       |
| Semantic retrieval      | Sentence Transformers, PyTorch CPU      | Implemented       |
| Vector database         | ChromaDB                                | Implemented       |
| Retrieval evaluation    | pandas, Recall@5, MRR@10                | Pilot implemented |
| Latency benchmarking    | Python `time.perf_counter()`          | Implemented       |
| Automated testing       | pytest                                  | Implemented       |
| LLM generation          | Gemini API                              | Planned           |
| Backend API             | FastAPI                                 | Planned           |
| Interactive application | Streamlit                               | Planned           |

## 4. Dataset

IssuePilot AI currently uses real public GitHub issues collected from the `psf/requests` repository.

**Dataset summary**

| Property            |            Value |
| ------------------- | ---------------: |
| Repository          | `psf/requests` |
| Total issues        |              300 |
| Closed issues       |              263 |
| Open issues         |               37 |
| Duplicate issue IDs |                0 |
| Missing titles      |                0 |
| Missing bodies      |                0 |

The collection pipeline excludes pull requests and normalizes records into a consistent JSONL format.

Each issue record contains information such as:

- GitHub issue ID
- Repository
- Title
- Body
- Normalized text
- Issue state
- Labels
- Original GitHub URL
- Creation timestamp

The processed dataset is stored locally at:

`data/processed/issues.jsonl`

The dataset is a small, repository-specific sample and should not be treated as representative of every GitHub project.

## 5. Retrieval Methods

### 5.1. TF-IDF Baseline

The lexical retrieval baseline uses `TfidfVectorizer` from scikit-learn.

The system:

1. Loads normalized GitHub issue documents.
2. Converts them into sparse TF-IDF vectors.
3. Transforms the user's search query using the same vectorizer.
4. Computes cosine similarity between the query and issue documents.
5. Returns the Top-K ranked issues.

For the initial 300-issue dataset, the TF-IDF index contained **7,556 text features**.

**Strengths**

- Fast retrieval on a small dataset.
- Low computational requirements.
- Useful for matching exact error messages, function names, and technical identifiers.
- Does not require a neural embedding model.

**Limitations**

- Depends heavily on lexical overlap.
- May miss related issues that use different vocabulary.

Run TF-IDF search:

```powershell
python -m scripts.tfidf_search
```

### 5.2. Dense Semantic Retrieval

The semantic search pipeline uses:

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

**Vector database:** ChromaDB

**Embedding dimension:** 384

**Distance metric:** Cosine distance

Each issue is represented using its title and a truncated portion of its description:

```text
Title: <issue title>
Description: <issue body>
```

The pipeline:

1. Loads the normalized issue dataset.
2. Generates 384-dimensional sentence embeddings.
3. Normalizes embeddings.
4. Stores the vectors and issue metadata in a persistent ChromaDB collection.
5. Embeds each incoming query.
6. Retrieves the nearest issues using vector similarity.

The current ChromaDB collection is named:

`github_issues_dense_v1`

Build the vector database:

```powershell
python -m scripts.build_vector_db
```

Run semantic search:

```powershell
python -m scripts.semantic_search
```

**Strengths**

- Can capture semantic relationships beyond exact keyword overlap.
- Useful for finding issues with related descriptions.

**Limitations**

- Requires model inference for each query.
- Has higher latency on CPU in the current setup.
- Semantic similarity does not guarantee that two issues describe the same bug.

TF-IDF scores and dense similarity scores are produced differently and should not be directly compared as if they were calibrated probabilities.

## 6. Retrieval Evaluation

The evaluation pipeline was implemented to compare both retrieval methods using the same query set and relevance criteria.

### 6.1. Evaluation Dataset

The current evaluation setup contains:

| Property               |                          Value |
| ---------------------- | -----------------------------: |
| Evaluation queries     |                             15 |
| Unique candidate pairs |                            256 |
| Candidate sources      | TF-IDF Top-10 and Dense Top-10 |
| Self-retrieval         |                       Excluded |
| Relevance labels       |            `y`, `n`, `s` |

The candidate pool is built by combining retrieval results from both methods and removing duplicate query–candidate pairs.

This reduces evaluation bias compared with labeling results from only one retrieval method.

Generate the evaluation pool:

```powershell
python -m scripts.build_eval_pool
```

**Warning:** Regenerating the candidate pool may overwrite existing labels. Preserve reviewed annotations before rebuilding it.

### 6.2. Relevance Labels

The evaluation uses three labels:

| Label | Meaning                                                          |
| ----- | ---------------------------------------------------------------- |
| `y` | Relevant: same technical issue or directly related investigation |
| `n` | Not relevant                                                     |
| `s` | Uncertain or insufficient information                            |

Labels should be verified using the actual issue descriptions, not only title similarity.

The initial labeling process produced mostly provisional, title-based judgments. Consequently, the current evaluation is still a pilot rather than a validated retrieval-quality benchmark.

### 6.3. Evaluation Metrics

**Pooled Recall@5**

Measures the proportion of confirmed relevant issues in the evaluated candidate pool that appear within the first five retrieved results.

```text
Pooled Recall@5 =
Relevant issues found in Top-5
--------------------------------
All confirmed relevant issues in the pool
```

This is pooled recall rather than absolute recall over every potentially relevant issue in the full corpus.

**Mean Reciprocal Rank at 10 (MRR@10)**

Measures how highly the first relevant result is ranked.

```text
RR@10 = 1 / rank_of_first_relevant_issue
```

If no relevant issue appears in the Top-10, the reciprocal rank is 0.

MRR@10 is the average reciprocal rank across evaluated queries.

### 6.4. Current Relevance Evaluation Results

The evaluation script is implemented and runs successfully, but only **one query** currently has a complete set of relevance labels in the evaluation file.

| Metric            | TF-IDF | Dense Retrieval |
| ----------------- | -----: | --------------: |
| Pooled Recall@5   | 1.0000 |          1.0000 |
| MRR@10            | 1.0000 |          1.0000 |
| Evaluated queries |      1 |               1 |

**Interpretation:** Both methods ranked the issue marked relevant at position 1 for this pilot query.

These figures are **not evidence of perfect retrieval accuracy**. The evaluation covers only one completed query, and the initial labels require further verification.

A reliable comparison of retrieval relevance remains future work.

Run the relevance evaluation:

```powershell
python -m scripts.evaluate_retrieval
```

Per-query results are written to:

`data/evaluation/benchmark_results.csv`

## 7. Retrieval Latency Benchmark

Retrieval latency was measured on a CPU-only Windows development environment.

### 7.1. Benchmark Setup

| Parameter            | Setting        |
| -------------------- | -------------- |
| Dataset size         | 300 issues     |
| Query set            | 15 queries     |
| Warm-up calls        | 3 per method   |
| Measured repetitions | 10             |
| Total measured calls | 150 per method |
| Results returned     | Top-5          |
| Execution            | Local CPU      |

Timing includes query processing and retrieval but excludes initial model loading and index construction.

### 7.2. Measured Results

| Metric       |            TF-IDF | Dense Retrieval |
| ------------ | ----------------: | --------------: |
| Mean latency | **4.75 ms** |        37.92 ms |
| P50 latency  | **4.73 ms** |        34.44 ms |
| P95 latency  | **7.04 ms** |        59.52 ms |

TF-IDF achieved approximately **8× lower mean latency** than dense retrieval in this setup.

Dense retrieval includes Sentence Transformer query embedding inference and ChromaDB vector search, which increases computational cost on CPU.

### 7.3. Interpretation

The latency experiment demonstrates a practical trade-off:

- TF-IDF is substantially faster on this small dataset.
- Dense retrieval offers a semantic representation but requires more computation.
- The current relevance evaluation is too limited to establish whether dense retrieval provides a measurable accuracy benefit.

The results are local measurements, not guarantees for other hardware, datasets, or production deployments.

Reproduce the latency benchmark:

```powershell
python -m scripts.benchmark_latency
```

Results are saved to:

`data/evaluation/latency_results.json`

## 8. Project Structure

```text
issuepilot-ai/
|
|-- data/
|   |-- processed/
|   |   |-- issues.jsonl
|   |
|   |-- evaluation/
|   |   |-- queries.jsonl
|   |   |-- labeling_pool.csv
|   |   |-- benchmark_results.csv
|   |   |-- latency_results.json
|   |
|   |-- chroma/
|
|-- scripts/
|   |-- collect_issues.py
|   |-- inspect_dataset.py
|   |-- tfidf_search.py
|   |-- prepare_eval.py
|   |-- label_issues.py
|   |-- auto_label.py
|   |-- build_vector_db.py
|   |-- semantic_search.py
|   |-- build_eval_pool.py
|   |-- evaluate_retrieval.py
|   |-- benchmark_latency.py
|
|-- tests/
|   |-- test_pipeline.py
|   |-- test_tfidf.py
|   |-- test_semantic_search.py
|
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
```

The structure shows the implemented retrieval and evaluation components. Locally generated datasets and vector indexes may be excluded from version control.

## 9. Installation and Setup

### Requirements

- Python 3.11
- Git
- Windows PowerShell
- Internet access for GitHub issue collection and initial model download
- CPU execution supported; a GPU is not required

### Clone the Repository

```powershell
git clone https://github.com/Minh-Canh33/issuepilot-ai.git

cd issuepilot-ai
```

### Create a Virtual Environment

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Configure Environment Variables

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Example GitHub configuration:

```dotenv
GITHUB_OWNER=psf
GITHUB_REPO=requests
MAX_ISSUES=300
MAX_PAGES=15
GITHUB_TOKEN=
```

A GitHub token may improve API rate limits.

Never commit `.env`, API tokens, virtual environments, or private credentials. If a credential is accidentally published, revoke it and rotate it immediately.

## 10. How to Run

Run commands from the repository root with the virtual environment activated.

**Step 1 — Collect GitHub Issues**

```powershell
python -m scripts.collect_issues
```

**Step 2 — Inspect the Dataset**

```powershell
python -m scripts.inspect_dataset
```

**Step 3 — Run TF-IDF Retrieval**

```powershell
python -m scripts.tfidf_search
```

**Step 4 — Build the Dense Vector Index**

```powershell
python -m scripts.build_vector_db
```

**Step 5 — Run Semantic Search**

```powershell
python -m scripts.semantic_search
```

**Step 6 — Prepare Retrieval Evaluation**

```powershell
python -m scripts.prepare_eval

python -m scripts.build_eval_pool
```

These preparation commands can recreate evaluation files. Do not rerun them over reviewed labels without making a backup.

**Step 7 — Evaluate Retrieval Quality**

```powershell
python -m scripts.evaluate_retrieval
```

**Step 8 — Benchmark Retrieval Latency**

```powershell
python -m scripts.benchmark_latency
```

**Step 9 — Run Automated Tests**

```powershell
python -m pytest -q
```

## 11. Development Roadmap

| Day   | Focus                                                        | Status                                             |
| ----- | ------------------------------------------------------------ | -------------------------------------------------- |
| Day 1 | GitHub API ingestion, normalization, dataset quality         | Completed                                          |
| Day 2 | TF-IDF retrieval baseline                                    | Completed                                          |
| Day 3 | Sentence embeddings and ChromaDB                             | Completed                                          |
| Day 4 | Retrieval evaluation and CPU latency benchmarks              | Implemented; relevance labels need more validation |
| Day 5 | Gemini API and evidence-grounded RAG generation              | Planned                                            |
| Day 6 | FastAPI backend and Streamlit demo                           | Planned                                            |
| Day 7 | Integration tests, documentation, and final portfolio polish | Planned                                            |

### Planned Day 5 — RAG Assistant

The next stage will integrate Gemini to generate answers grounded in retrieved GitHub issues.

Planned workflow:

```text
User question
     |
     v
Semantic retrieval
     |
     v
Top-K GitHub issues
     |
     v
Context construction
     |
     v
Gemini API
     |
     v
Answer with GitHub source links
```

The assistant will be designed to distinguish evidence from suggestions and avoid claiming that an issue is resolved without supporting information.

## 12. Limitations and Future Improvements

Current limitations:

- Only 300 issues from one GitHub repository are indexed.
- Semantic embeddings are generated with a general-purpose sentence embedding model.
- The relevance dataset is small and only partially validated.
- Pooled Recall@5 does not measure absolute corpus-wide recall.
- Latency figures are specific to the measured local CPU setup.
- Dense retrieval currently runs without a dedicated GPU.
- RAG generation and the user-facing application have not yet been implemented.

Potential improvements:

- Expand the evaluation dataset with carefully verified relevance judgments.
- Evaluate title-only versus title-and-description embeddings.
- Introduce hybrid retrieval using lexical and dense scores.
- Add reranking for candidate issues.
- Evaluate answer faithfulness and source attribution once RAG is implemented.
- Test retrieval performance on larger repositories.

## 13. Engineering Lessons

The project demonstrates hands-on work with:

- REST API integration and pagination.
- Data cleaning and normalized JSONL pipelines.
- Sparse information retrieval using TF-IDF.
- Sentence embedding inference.
- Persistent vector storage with ChromaDB.
- Evaluation dataset construction and relevance labeling.
- Retrieval metrics and benchmark design.
- CPU latency measurement using mean, P50, and P95.
- Automated tests and incremental Git development.

An important lesson from Day 4 is that **implementing a metric is not the same as obtaining a trustworthy evaluation result**. Retrieval-quality claims depend on valid queries, relevance definitions, sufficient coverage, and verified ground-truth labels.

## 14. References

- [GitHub Issues REST API](https://docs.github.com/en/rest/issues/issues)
- [HTTPX Documentation](https://www.python-httpx.org/)
- [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Sentence Transformers](https://www.sbert.net/)
- [all-MiniLM-L6-v2 Model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Chroma Documentation](https://docs.trychroma.com/)
- [pytest Documentation](https://docs.pytest.org/)

---

**IssuePilot AI** — An evolving AI engineering portfolio project focused on building, measuring, and improving a practical GitHub issue retrieval and RAG syste

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
