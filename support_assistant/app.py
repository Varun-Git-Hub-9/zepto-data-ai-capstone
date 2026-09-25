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
    name="zepto_support",
    metadata={"hnsw:space": "cosine"}
)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running",
        "mock_llm": MOCK_LLM
    }
from pathlib import Path
from langgraph.graph import StateGraph, END


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


PROMPT_TEMPLATE = """
Role:
You are a Zepto support assistant.

Context:
Use only the retrieved Zepto policy context provided below.

Task:
Answer the user's question accurately.

Format:
Return a concise answer with source references.

Length:
Keep the answer under 120 words.

Negative constraint:
Do not answer using information that is not present in the provided context.

Few-shot example:
Question: Can I cancel an order after it is packed?
Answer: No. Once an order is packed, it can no longer be cancelled through the app.

Retrieved context:
{context}

User question:
{query}
"""


def load_documents():
    docs_dir = Path(__file__).parent / "docs"

    documents = []
    ids = []

    for path in sorted(docs_dir.glob("doc_*.txt")):
        documents.append(path.read_text(encoding="utf-8"))
        ids.append(path.stem)

    return ids, documents


def index_documents():
    ids, documents = load_documents()

    existing = collection.get()

    if len(existing["ids"]) == 0:
        embeddings = embedding_model.encode(
            documents
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )

    return len(ids)


DOCUMENT_COUNT = index_documents()


def classify_intent(state: SupportState):
    query = state["query"].lower()

    if MOCK_LLM:
        intent = (
            "policy_question"
            if any(keyword in query for keyword in POLICY_KEYWORDS)
            else "general_question"
        )

        return {
            "intent": intent
        }

    # Optional real-LLM extension can be added here.
    return {
        "intent": "general_question"
    }


def retrieve_and_answer(state: SupportState):
    query = state["query"]

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    top_chunk = documents[0]

    if MOCK_LLM:
        answer = (
            "Based on the retrieved context: "
            + top_chunk[:200]
        )

        return {
            "answer": answer,
            "sources": ids,
            "confidence": 1.0
        }

    context = "\n\n".join(documents)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        query=query
    )

    # Optional real LLM call would use prompt here.
    return {
        "answer": "Real LLM mode is not configured.",
        "sources": ids,
        "confidence": 0.0
    }


def direct_answer(state: SupportState):
    if MOCK_LLM:
        return {
            "answer": (
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0
        }

    return {
        "answer": "Real LLM mode is not configured.",
        "sources": [],
        "confidence": 0.0
    }


def route_intent(state: SupportState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


graph_builder = StateGraph(SupportState)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.set_entry_point(
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

graph = graph_builder.compile()


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):
    result = graph.invoke(
        {
            "query": request.query
        }
    )

    response = AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )

    return response
# ---------------------------------------------------------
# Optional real-LLM JSON validation/retry helper
# ---------------------------------------------------------

def validate_real_llm_response(call_llm_function, prompt):
    """
    Optional MOCK_LLM=0 helper.

    If a real LLM is configured later, its raw response is validated
    against AskResponse. Validation failures are retried up to two
    additional times with a corrective instruction.
    """

    current_prompt = prompt

    for attempt in range(3):
        raw_response = call_llm_function(current_prompt)

        try:
            return AskResponse.model_validate_json(raw_response)

        except Exception as exc:
            if attempt == 2:
                return AskResponse(
                    answer=(
                        "The real LLM response could not be validated "
                        "against the required JSON schema."
                    ),
                    sources=[],
                    confidence=0.0
                )

            current_prompt = (
                prompt
                + "\n\nCorrective instruction: Return only valid JSON "
                + "with fields answer (string), sources (list of strings), "
                + "and confidence (number from 0 to 1). "
                + f"Previous validation error: {exc}"
            )
