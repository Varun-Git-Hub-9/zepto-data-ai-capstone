# Module 3 - Zepto Support Assistant

This module implements a small RAG-based support assistant using the eight Zepto policy documents provided in the assignment.

The graded baseline runs in deterministic mock mode and does not call an external LLM.

## Main Components

- 8 Zepto policy documents
- Sentence Transformers using `all-MiniLM-L6-v2`
- ChromaDB using cosine similarity
- LangGraph for intent routing
- Pydantic for structured JSON output
- FastAPI `/ask` endpoint
- Dockerfile for local container execution

## MOCK_LLM

Mock mode is enabled by default.

```text
MOCK_LLM=1


pip install -r requirements.txt
cd ~/zepto-data-ai-capstone

