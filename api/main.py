
from functools import lru_cache

import chromadb
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from scripts.build_vector_db import (
    COLLECTION_NAME,
    DB_PATH,
    MODEL_NAME,
)
from scripts.rag_assistant import answer_question
from scripts.tfidf_search import DATA_PATH, load_issues


app = FastAPI(
    title="IssuePilot AI API",
    description="GitHub Issue Retrieval & RAG Assistant",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@lru_cache(maxsize=1)
def get_rag_resources():
    """
    Load the embedding model, ChromaDB collection,
    and local issue dataset once per server process.
    """
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    issues = load_issues(DATA_PATH)
    issue_by_id = {
        str(issue["issue_id"]): issue
        for issue in issues
    }

    return model, collection, issue_by_id


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "IssuePilot AI",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Question must contain at least 3 characters.",
        )

    try:
        model, collection, issue_by_id = get_rag_resources()

        answer = answer_question(
            question,
            model,
            collection,
            issue_by_id,
        )

        return AskResponse(answer=answer)

    except httpx.HTTPStatusError as error:
        print(
            "Gemini upstream HTTP status:",
            error.response.status_code,
        )

        raise HTTPException(
            status_code=502,
            detail="Gemini API request failed.",
        )

    except httpx.RequestError as error:
        print("Gemini connection error:", type(error).__name__)

        raise HTTPException(
            status_code=502,
            detail="Cannot connect to Gemini API.",
        )

    except Exception as error:
        print("IssuePilot internal error:", type(error).__name__)

        raise HTTPException(
            status_code=500,
            detail="IssuePilot could not process the question.",
        )
