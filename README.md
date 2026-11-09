# 🔎 IssuePilot AI

**GitHub Issue Retrieval & RAG Assistant**

IssuePilot AI is an AI engineering portfolio project that helps developers search for related GitHub issues and investigate technical problems using Retrieval-Augmented Generation (RAG).

The system combines **TF-IDF keyword search**, **Sentence Transformer embeddings**, **ChromaDB vector search**, and **Google Gemini** to generate technical investigation suggestions grounded in real GitHub issue reports.

It also provides a **FastAPI backend** and an interactive **Streamlit web interface**.

**Repository:** https://github.com/Minh-Canh33/issuepilot-ai

---

## 1. Project Overview

### Problem

When developers encounter a bug, they often search GitHub for similar issues.

However, traditional keyword-based search has limitations:

- Different developers may describe the same problem using different terminology.
- Exact keyword matches do not always indicate the same underlying bug.
- Searching through many issues can be time-consuming.
- Relevant information may be buried inside issue descriptions.

### Solution

IssuePilot AI combines lexical and semantic retrieval with an LLM-powered assistant.

The system can:

- Collect and normalize real GitHub issues.
- Search using TF-IDF and semantic embeddings.
- Retrieve potentially relevant historical issues.
- Generate AI-assisted investigation suggestions.
- Provide links to supporting GitHub issues.
- Expose functionality through a REST API.
- Offer a web interface for interactive use.

**Important:** IssuePilot AI is an investigation assistant, not an automated bug fixer. AI-generated suggestions must be independently verified.

---

## 2. System Architecture

```text
                   GitHub REST API
                          |
                          v
                 Issue Collection
                          |
                          v
                 Data Processing
                          |
                          v
                  JSONL Dataset
                    300 Issues
                          |
               +----------+----------+
               |                     |
               v                     v
         TF-IDF Search        Semantic Search
          scikit-learn        Sentence Transformers
               |                     |
               |                 Embeddings
               |                     |
               |                     v
               |                  ChromaDB
               |                     |
               +----------+----------+
                          |
                          v
                  Top-K Related Issues
                          |
                          v
                RAG Context Builder
                Title + Body + URL
                          |
                          v
                     Gemini API
                          |
                          v
               Evidence-Grounded Answer
                          |
                          v
                    FastAPI Backend
                          |
                          v
                   Streamlit Web UI
                          |
                          v
                   End User
```

The TF-IDF and dense retrieval methods can be benchmarked independently. The current RAG assistant uses dense retrieval to supply context to Gemini.

---

## 3. Technology Stack

| Component               | Technology                          |
| ----------------------- | ----------------------------------- |
| Programming language    | Python 3.11                         |
| Data collection         | GitHub REST API, HTTPX              |
| Data processing         | pandas, JSONL                       |
| Keyword retrieval       | scikit-learn, TF-IDF                |
| Semantic embeddings     | Sentence Transformers               |
| Embedding model         | `all-MiniLM-L6-v2`                |
| Vector database         | ChromaDB                            |
| LLM                     | Google Gemini API                   |
| Backend                 | FastAPI, Uvicorn                    |
| Frontend                | Streamlit                           |
| Evaluation              | Recall@5, MRR@10, latency           |
| Testing                 | pytest, FastAPI TestClient          |
| Development environment | Windows, Python virtual environment |
| Version control         | Git, GitHub                         |

The application supports CPU-only execution.

---

## 4. Dataset

The dataset contains **300 public GitHub issues** collected from the `psf/requests` repository.

### Dataset Statistics

| Metric              | Result |
| ------------------- | -----: |
| Total issues        |    300 |
| Closed issues       |    263 |
| Open issues         |     37 |
| Duplicate issue IDs |      0 |
| Missing titles      |      0 |
| Missing bodies      |      0 |

### Data Collection Pipeline

The collection pipeline:

1. Connects to the GitHub REST API.
2. Retrieves issues using pagination.
3. Excludes pull requests.
4. Normalizes issue records.
5. Saves processed records in JSONL format.
6. Inspects the dataset for missing and duplicate values.

Example issue fields:

```json
{
  "issue_id": 7547,
  "repository": "psf/requests",
  "title": "Example issue title",
  "body": "Example issue description",
  "state": "open",
  "url": "https://github.com/psf/requests/issues/7547"
}
```

The record above illustrates the data structure; it is not a verbatim copy of issue #7547.

### Dataset Location

```text
data/processed/issues.jsonl
```

The collected dataset is stored locally and is not required to be committed to GitHub.

---

## 5. Retrieval Methods

### 5.1 TF-IDF Retrieval

The baseline retrieval system uses `TfidfVectorizer` and cosine similarity.

**Pipeline:**

```text
User Query
    |
    v
TF-IDF Vectorizer
    |
    v
Sparse Query Vector
    |
    v
Cosine Similarity
    |
    v
Top-K GitHub Issues
```

The initial TF-IDF representation contains **7,556 text features** for the 300-issue corpus.

**Advantages:**

- Fast on small datasets.
- Computationally inexpensive.
- Effective at matching exact technical terms, function names, and error messages.

**Limitations:**

- Depends on shared vocabulary.
- Can miss semantically related issues expressed using different words.

Run:

```powershell
python -m scripts.tfidf_search
```

### 5.2 Semantic Retrieval

Semantic retrieval uses the Sentence Transformer model:

`sentence-transformers/all-MiniLM-L6-v2`

Each issue is embedded into a **384-dimensional vector**.

The embedding input combines the title with a truncated issue description.

```text
Title: <issue title>
Description: <issue body>
```

The embeddings are normalized and stored in ChromaDB using cosine distance.

**Pipeline:**

```text
User Query
    |
    v
Sentence Transformer
    |
    v
384-D Query Embedding
    |
    v
ChromaDB Vector Search
    |
    v
Top-K GitHub Issues
```

Build the vector database:

```powershell
python -m scripts.build_vector_db
```

Run semantic retrieval:

```powershell
python -m scripts.semantic_search
```

**Advantages:**

- Captures semantic relationships beyond exact keyword matches.
- Can retrieve issues using related descriptions.

**Limitations:**

- Requires neural model inference.
- Has greater CPU latency than TF-IDF in the current benchmark.
- Semantic similarity does not guarantee two issues describe the same bug.

TF-IDF similarity scores and dense retrieval scores are not directly comparable.

---

## 6. Retrieval Evaluation

The project implements an evaluation pipeline for comparing retrieval methods.

### Evaluation Dataset

| Property              |                          Value |
| --------------------- | -----------------------------: |
| Evaluation queries    |                             15 |
| Candidate issue pairs |                            256 |
| Candidate sources     | TF-IDF Top-10 and Dense Top-10 |
| Self-retrieval        |                       Excluded |
| Relevance labels      |            `y`, `n`, `s` |

Candidates from the two retrieval methods are merged and deduplicated.

The relevance labels mean:

| Label | Meaning                 |
| ----- | ----------------------- |
| `y` | Relevant to the query   |
| `n` | Not relevant            |
| `s` | Uncertain; needs review |

### Retrieval Metrics

**Pooled Recall@5**

Measures the proportion of confirmed relevant issues in the candidate pool that appear among the top five retrieval results.

```text
Pooled Recall@5 =
    Relevant issues retrieved in Top-5
    ----------------------------------
    All confirmed relevant issues in the pool
```

**MRR@10 — Mean Reciprocal Rank**

Measures how highly the first relevant result appears within the first ten results.

```text
Reciprocal Rank = 1 / Rank of First Relevant Result
```

MRR@10 averages this value across evaluable queries.

### Current Relevance Evaluation

The evaluation code is implemented and runs successfully. However, the current pilot contains only **one completed query**.

| Metric            | TF-IDF | Dense Retrieval |
| ----------------- | -----: | --------------: |
| Pooled Recall@5   | 1.0000 |          1.0000 |
| MRR@10            | 1.0000 |          1.0000 |
| Evaluated queries |      1 |               1 |

These results are preliminary.

The initial relevance labels included title-based suggestions that require further verification. Therefore, the current evaluation **cannot establish which retrieval method has better relevance quality**.

The reported recall is pooled recall, not exhaustive recall across the entire issue corpus.

Run the evaluation:

```powershell
python -m scripts.evaluate_retrieval
```

Evaluation output:

```text
data/evaluation/benchmark_results.csv
```

---

## 7. Retrieval Latency Benchmark

Both retrieval methods were benchmarked locally on CPU.

### Benchmark Configuration

| Parameter                 | Value      |
| ------------------------- | ---------- |
| Corpus size               | 300 issues |
| Evaluation queries        | 15         |
| Measured repetitions      | 10         |
| Measured calls per method | 150        |
| Warm-up calls             | 3          |
| Retrieval depth           | Top-5      |
| Hardware                  | Local CPU  |

Model loading and index construction were excluded from the timing measurements.

### Measured Results

| Metric       |            TF-IDF | Dense Retrieval |
| ------------ | ----------------: | --------------: |
| Mean latency | **4.75 ms** |        37.92 ms |
| P50 latency  | **4.73 ms** |        34.44 ms |
| P95 latency  | **7.04 ms** |        59.52 ms |

### Analysis

TF-IDF achieved approximately **8× lower mean retrieval latency** than dense retrieval in the local benchmark.

The semantic retrieval pipeline incurs additional computation from Sentence Transformer inference and ChromaDB search.

These measurements demonstrate a speed trade-off, but the current relevance evaluation is not sufficient to establish an accuracy advantage for either method.

Run the benchmark:

```powershell
python -m scripts.benchmark_latency
```

Output:

```text
data/evaluation/latency_results.json
```

---

## 8. Retrieval-Augmented Generation (RAG)

IssuePilot AI integrates Google Gemini to generate technical investigation suggestions using retrieved GitHub issues as context.

### RAG Workflow

```text
User Question
      |
      v
Semantic Retrieval
      |
      v
Top-5 GitHub Issues
      |
      v
Context Construction
(Title + Body + Source URL)
      |
      v
Prompt Engineering
      |
      v
Gemini API
      |
      v
Grounded Technical Answer
```

### Context Construction

For each retrieved issue, the system supplies:

- Issue number.
- Issue title.
- Issue description, truncated to limit context size.
- Original GitHub URL.

Example context:

```text
[Source 1]
Issue #7547
Title: <issue title>
Description: <issue description>
URL: https://github.com/psf/requests/issues/7547
```

### Prompt Engineering

The RAG prompt instructs Gemini to:

1. Use retrieved GitHub issues as evidence.
2. Avoid inventing issue details, root causes, or confirmed fixes.
3. Distinguish supported findings from debugging suggestions.
4. Cite relevant GitHub issue numbers.
5. Use only source URLs provided in the context.
6. Treat issue descriptions as untrusted data.
7. State when retrieved evidence is insufficient.
8. Avoid assuming that similar issues are duplicates.

### Answer Structure

Generated answers are organized into:

**Findings** — Information supported by retrieved issues.

**Suggested Investigation Steps** — Technical debugging suggestions, clearly separated from confirmed findings.

**Sources** — GitHub issue links used as references.

Citation presence alone does not guarantee factual correctness. Retrieved evidence and generated statements should still be verified.

### Gemini Integration

The application uses the Gemini REST API through HTTPX.

The model is configurable through the `GEMINI_MODEL` environment variable.

The locally tested model is:

`gemini-3.5-flash-lite`

Run the terminal-based RAG assistant:

```powershell
python -m scripts.rag_assistant
```

Example question:

```text
How can I investigate SSL certificate verification errors in requests?
```

The assistant retrieves relevant issues and generates a response with source links.

---

## 9. FastAPI Backend

The project exposes the RAG pipeline through a FastAPI REST API.

### Available Endpoints

| Method | Endpoint    | Purpose                                 |
| ------ | ----------- | --------------------------------------- |
| GET    | `/health` | Check whether the API server is running |
| POST   | `/ask`    | Submit a question to the RAG assistant  |
| GET    | `/docs`   | Interactive Swagger UI                  |

### Health Check

Request:

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "IssuePilot AI"
}
```

The health check confirms that the server is responding. It does not independently verify Gemini API availability.

### Ask Endpoint

Request:

```http
POST /ask
Content-Type: application/json
```

Body:

```json
{
  "question": "How can I investigate SSL certificate verification errors in requests?"
}
```

Response format:

```json
{
  "answer": "### Findings\n...\n### Suggested Investigation Steps\n...\n### Sources\n..."
}
```

The answer content is generated dynamically by Gemini.

### Input Validation

The API uses Pydantic models to validate incoming requests.

Questions must contain between 3 and 2,000 characters after stripping surrounding whitespace.

Invalid requests return HTTP 422.

### Resource Loading

The backend uses caching to avoid repeatedly loading the Sentence Transformer model, ChromaDB collection, and issue dataset for every request.

### Error Handling

The backend handles errors such as:

- Invalid request data.
- Gemini API HTTP failures.
- Network connection failures.
- Unexpected internal errors.

Upstream Gemini failures are returned to clients as HTTP 502 without exposing the API key.

### Run FastAPI

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Swagger UI:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health

The `--reload` option is intended for local development.

---

## 10. Streamlit Web Interface

IssuePilot AI includes a browser-based interface built with Streamlit.

### Features

- Technical question input.
- **Analyze Issue** button.
- Backend communication using HTTPX.
- Loading indicator during generation.
- Markdown rendering of the AI answer.
- Clickable GitHub source links.
- Validation and error messages.

### Application Flow

```text
Browser
   |
   v
Streamlit UI
   |
   | POST /ask
   v
FastAPI
   |
   v
RAG Pipeline
   |
   v
Gemini Response
   |
   v
FastAPI JSON
   |
   v
Streamlit Answer Display
```

The Streamlit application does not call Gemini directly. It communicates with FastAPI, keeping API credentials on the backend.

### Run Streamlit

With FastAPI already running, open a second terminal:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open:

http://localhost:8501

### Example Usage

1. Open the Streamlit web interface.
2. Enter a question about a technical issue.
3. Click **Analyze Issue**.
4. Review the generated findings.
5. Inspect suggested investigation steps.
6. Open the linked GitHub issues to verify the evidence.

Example question:

```text
How can I investigate SSL certificate verification errors in requests?
```

The Streamlit-to-FastAPI-to-Gemini workflow has been tested manually with a successful response.

---

## 11. Project Structure

```text
issuepilot-ai/
|
|-- api/
|   |-- __init__.py
|   |-- main.py
|
|-- app/
|   |-- __init__.py
|   |-- streamlit_app.py
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
|   |-- check_gemini.py
|   |-- check_gemini_rest.py
|   |-- rag_assistant.py
|
|-- tests/
|   |-- test_pipeline.py
|   |-- test_tfidf.py
|   |-- test_semantic_search.py
|   |-- test_rag.py
|   |-- test_api.py
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
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
```

Some data files, vector indexes, and local environment files are intentionally excluded from Git version control.

---

## 12. Installation

### Prerequisites

- Python 3.11
- Git
- Internet connection
- Gemini API key
- Windows PowerShell or an equivalent terminal

A GPU is not required.

### Step 1 — Clone Repository

```powershell
git clone https://github.com/Minh-Canh33/issuepilot-ai.git
cd issuepilot-ai
```

### Step 2 — Create Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Create a `.env` file in the repository root.

Example:

```dotenv
GITHUB_OWNER=psf
GITHUB_REPO=requests
MAX_ISSUES=300
MAX_PAGES=15
GITHUB_TOKEN=

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.5-flash-lite
```

Replace `YOUR_GEMINI_API_KEY` with a valid API key.

**Security:**

- Never commit `.env`.
- Never publish API keys in screenshots or logs.
- Do not commit `.venv` or the local ChromaDB directory.
- Revoke and rotate any exposed API key.

---

## 13. How to Run the Full Project

Run the commands from the repository root.

### Step 1 — Collect Issues

```powershell
python -m scripts.collect_issues
```

### Step 2 — Inspect Dataset

```powershell
python -m scripts.inspect_dataset
```

### Step 3 — Build the Vector Database

```powershell
python -m scripts.build_vector_db
```

### Step 4 — Test Semantic Search

```powershell
python -m scripts.semantic_search
```

### Step 5 — Test Gemini Connectivity

```powershell
python -m scripts.check_gemini_rest
```

### Step 6 — Start FastAPI

**Terminal 1:**

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Backend:

http://127.0.0.1:8000

### Step 7 — Start Streamlit

**Terminal 2:**

```powershell
python -m streamlit run app/streamlit_app.py
```

Frontend:

http://localhost:8501

Both processes must remain running for the web application to work.

### Step 8 — Use IssuePilot AI

Open the frontend, enter a technical question, and select **Analyze Issue**.

The backend retrieves relevant GitHub issues and calls Gemini to generate a grounded response.

---

## 14. Automated Testing

The project includes automated tests for data processing, retrieval, RAG, and the API.

### Test Coverage Areas

| Test file                   | Focus                                                          |
| --------------------------- | -------------------------------------------------------------- |
| `test_pipeline.py`        | Dataset collection and processing                              |
| `test_tfidf.py`           | Lexical retrieval                                              |
| `test_semantic_search.py` | Dense retrieval                                                |
| `test_rag.py`             | Context construction, prompt, mocked Gemini responses          |
| `test_api.py`             | Health check, request validation, API response, backend errors |

### RAG Tests

Five RAG unit tests have passed.

They cover:

- Inclusion of issue descriptions in RAG context.
- Prompt construction.
- Parsing the Gemini response.
- Using retrieved issues in the generated prompt.
- Avoiding Gemini calls when no issues are retrieved.

### FastAPI Tests

Five API tests have passed.

They cover:

- Successful `/health` response.
- Rejection of short questions.
- Rejection of missing question fields.
- Successful `/ask` response.
- Handling internal backend failures.

### Run Tests

```powershell
python -m pytest -q
```

Run the API tests separately:

```powershell
python -m pytest tests/test_api.py -v
```

The RAG and API unit tests use mocks, so they do not need to call Gemini for every test run.

---

## 15. Development Roadmap

| Day   | Objective                                                                 | Status                                                   |
| ----- | ------------------------------------------------------------------------- | -------------------------------------------------------- |
| Day 1 | GitHub REST API collection and data normalization                         | Completed                                                |
| Day 2 | TF-IDF retrieval baseline                                                 | Completed                                                |
| Day 3 | Sentence Transformers and ChromaDB                                        | Completed                                                |
| Day 4 | Retrieval evaluation and latency benchmarking                             | Implemented; relevance labels require further validation |
| Day 5 | Gemini RAG pipeline with source grounding                                 | Completed                                                |
| Day 6 | FastAPI backend and Streamlit web interface                               | Implemented and manually tested                          |
| Day 7 | Final tests, evaluation improvements, documentation, and portfolio polish | Planned                                                  |

---

## 16. Current Limitations

IssuePilot AI is a functional prototype with several limitations:

- The dataset contains only 300 issues from one repository.
- Retrieval relevance judgments are not yet sufficiently verified for a reliable comparison.
- Dense retrieval requires additional CPU processing.
- Retrieved issue content can contain incomplete or incorrect user reports.
- Gemini can generate plausible but unsupported technical suggestions.
- Source URLs are included in the response, but citations are not yet automatically verified against individual claims.
- The current web application is intended for local development and demonstration.
- Production authentication, rate limiting, deployment, and monitoring have not been implemented.

The application should not be used to make security-sensitive or production debugging decisions without independent validation.

---

## 17. Future Improvements

Potential improvements include:

- Expanding and verifying the relevance evaluation dataset.
- Adding hybrid TF-IDF + dense retrieval.
- Introducing cross-encoder reranking.
- Comparing title-only and title-plus-body embeddings.
- Validating generated citations against retrieved source IDs.
- Returning structured source metadata from FastAPI.
- Evaluating response faithfulness and answer usefulness.
- Adding caching and robust retry handling for transient API failures.
- Improving the Streamlit interface.
- Adding deployment configuration and monitoring.

---

## 18. Engineering Skills Demonstrated

This project demonstrates practical experience with:

- GitHub REST API integration.
- Pagination and data normalization.
- JSONL-based data pipelines.
- TF-IDF and cosine similarity.
- Neural sentence embeddings.
- Vector databases and semantic retrieval.
- Retrieval evaluation metrics.
- CPU latency benchmarking.
- Retrieval-Augmented Generation.
- LLM prompt engineering and source grounding.
- Gemini REST API integration.
- FastAPI request validation and response handling.
- Streamlit user-interface development.
- Mock-based testing with pytest.
- Git-based incremental development.

The project also highlights an important engineering principle:

**A working AI demo is not the same as a fully validated AI system.** Reliable evaluation, source verification, and transparent limitations are essential for trustworthy results.

---

## 19. References

- [GitHub REST API](https://docs.github.com/en/rest/issues/issues)
- [HTTPX](https://www.python-httpx.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [ChromaDB](https://docs.trychroma.com/)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)
- [pytest](https://docs.pytest.org/)

---

**IssuePilot AI — A practical AI engineering project combining information retrieval, RAG, API development, and interactive web applications**

# 🔎 IssuePilot AI

**GitHub Issue Retrieval & RAG Assistant**

IssuePilot AI is an AI engineering portfolio project that helps developers search for related GitHub issues and investigate technical problems using Retrieval-Augmented Generation (RAG).

The system combines **TF-IDF keyword search**, **Sentence Transformer embeddings**, **ChromaDB vector search**, and **Google Gemini** to generate technical investigation suggestions grounded in real GitHub issue reports.

It also provides a **FastAPI backend** and an interactive **Streamlit web interface**.

**Repository:** https://github.com/Minh-Canh33/issuepilot-ai

---

## 1. Project Overview

### Problem

When developers encounter a bug, they often search GitHub for similar issues.

However, traditional keyword-based search has limitations:

- Different developers may describe the same problem using different terminology.
- Exact keyword matches do not always indicate the same underlying bug.
- Searching through many issues can be time-consuming.
- Relevant information may be buried inside issue descriptions.

### Solution

IssuePilot AI combines lexical and semantic retrieval with an LLM-powered assistant.

The system can:

- Collect and normalize real GitHub issues.
- Search using TF-IDF and semantic embeddings.
- Retrieve potentially relevant historical issues.
- Generate AI-assisted investigation suggestions.
- Provide links to supporting GitHub issues.
- Expose functionality through a REST API.
- Offer a web interface for interactive use.

**Important:** IssuePilot AI is an investigation assistant, not an automated bug fixer. AI-generated suggestions must be independently verified.

---

## 2. System Architecture

```text
                   GitHub REST API
                          |
                          v
                 Issue Collection
                          |
                          v
                 Data Processing
                          |
                          v
                  JSONL Dataset
                    300 Issues
                          |
               +----------+----------+
               |                     |
               v                     v
         TF-IDF Search        Semantic Search
          scikit-learn        Sentence Transformers
               |                     |
               |                 Embeddings
               |                     |
               |                     v
               |                  ChromaDB
               |                     |
               +----------+----------+
                          |
                          v
                  Top-K Related Issues
                          |
                          v
                RAG Context Builder
                Title + Body + URL
                          |
                          v
                     Gemini API
                          |
                          v
               Evidence-Grounded Answer
                          |
                          v
                    FastAPI Backend
                          |
                          v
                   Streamlit Web UI
                          |
                          v
                   End User
```

The TF-IDF and dense retrieval methods can be benchmarked independently. The current RAG assistant uses dense retrieval to supply context to Gemini.

---

## 3. Technology Stack

| Component               | Technology                          |
| ----------------------- | ----------------------------------- |
| Programming language    | Python 3.11                         |
| Data collection         | GitHub REST API, HTTPX              |
| Data processing         | pandas, JSONL                       |
| Keyword retrieval       | scikit-learn, TF-IDF                |
| Semantic embeddings     | Sentence Transformers               |
| Embedding model         | `all-MiniLM-L6-v2`                |
| Vector database         | ChromaDB                            |
| LLM                     | Google Gemini API                   |
| Backend                 | FastAPI, Uvicorn                    |
| Frontend                | Streamlit                           |
| Evaluation              | Recall@5, MRR@10, latency           |
| Testing                 | pytest, FastAPI TestClient          |
| Development environment | Windows, Python virtual environment |
| Version control         | Git, GitHub                         |

The application supports CPU-only execution.

---

## 4. Dataset

The dataset contains **300 public GitHub issues** collected from the `psf/requests` repository.

### Dataset Statistic

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
