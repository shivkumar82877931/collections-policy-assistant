"""
The main chat endpoint. This is the literal implementation of the system
diagram in guide file 00 — input guardrails -> retrieval (hybrid + rerank)
-> generation -> output guardrails -> response, with full tracing logged
per request (guide file 07, section 5: "trace-level visibility is what
makes a flagged failure debuggable").
"""

import time
import uuid
from fastapi import APIRouter
from pydantic import BaseModel

from app.guardrails.input_guardrails import run_input_guardrails
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank
from app.generation.generate import generate_answer
from app.guardrails.output_guardrails import run_output_guardrails
from app.config import settings

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    include_superseded: bool = False  # exposed for demo purposes — lets you
                                       # show the metadata filter working live


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    sources: list[dict]
    guardrails: dict
    latency_ms: dict


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    trace_id = str(uuid.uuid4())
    timings = {}
    t0 = time.time()

    # --- Input guardrails ---
    t_start = time.time()
    input_result = {"safe_query": request.query, "pii_detected": [], "injection_flagged": False, "block": False}
    if settings.ENABLE_INPUT_GUARDRAILS:
        input_result = run_input_guardrails(request.query)
    timings["input_guardrails_ms"] = round((time.time() - t_start) * 1000, 1)

    if input_result["block"]:
        return ChatResponse(
            trace_id=trace_id,
            answer="This query was flagged by input safety checks and cannot be processed. "
                   "If you believe this is an error, please rephrase your question.",
            sources=[],
            guardrails={"input": input_result, "output": None},
            latency_ms=timings,
        )

    safe_query = input_result["safe_query"]

    # --- Retrieval: hybrid search ---
    t_start = time.time()
    candidates = hybrid_search(
        safe_query,
        top_k=settings.TOP_K_RETRIEVE,
        only_current=not request.include_superseded,
    )
    timings["retrieval_ms"] = round((time.time() - t_start) * 1000, 1)

    # --- Re-ranking ---
    t_start = time.time()
    top_chunks = rerank(safe_query, candidates, top_k=settings.TOP_K_FINAL)
    timings["rerank_ms"] = round((time.time() - t_start) * 1000, 1)

    # --- Generation ---
    t_start = time.time()
    answer = generate_answer(safe_query, top_chunks)
    timings["generation_ms"] = round((time.time() - t_start) * 1000, 1)

    # --- Output guardrails ---
    t_start = time.time()
    output_result = {"needs_caveat": False, "groundedness": None, "citations": None}
    if settings.ENABLE_OUTPUT_GUARDRAILS:
        output_result = run_output_guardrails(answer, top_chunks)
    timings["output_guardrails_ms"] = round((time.time() - t_start) * 1000, 1)

    if output_result["needs_caveat"]:
        answer += ("\n\n⚠️ Note: parts of this answer could not be fully verified against "
                   "the source documents. Please confirm critical details directly with "
                   "the cited policy document.")

    timings["total_ms"] = round((time.time() - t0) * 1000, 1)

    sources = [
        {
            "source_file": c.get("metadata", {}).get("source_file", "unknown"),
            "section": c.get("metadata", {}).get("section", ""),
            "version": c.get("metadata", {}).get("version", ""),
            "rerank_score": c.get("rerank_score"),
            "excerpt": c["text"][:200],
        }
        for c in top_chunks
    ]

    return ChatResponse(
        trace_id=trace_id,
        answer=answer,
        sources=sources,
        guardrails={"input": input_result, "output": output_result},
        latency_ms=timings,
    )
