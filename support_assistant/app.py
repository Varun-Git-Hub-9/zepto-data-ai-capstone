import os
from typing import List, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class SupportState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="zepto_support"
)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running",
        "mock_llm": MOCK_LLM
    }
