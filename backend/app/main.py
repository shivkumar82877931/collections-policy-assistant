"""
FastAPI app entrypoint. Wires up routes, CORS (for the static frontend),
and builds the in-memory BM25 index at startup (see retrieval/hybrid_search.py).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import chat, health
from app.retrieval.hybrid_search import build_bm25_index
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # BM25 needs the vector store already populated — run
    # `python -m app.ingestion.ingest` once before starting the server.
    build_bm25_index()
    yield


app = FastAPI(
    title="Collections Policy Assistant",
    description="RAG-based policy lookup assistant for bank collections agents. "
                 "Built as an applied demonstration of hybrid retrieval, re-ranking, "
                 "and layered guardrails for a compliance-sensitive domain.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
