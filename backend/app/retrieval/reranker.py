"""
Cross-encoder reranking. See guide file 04, section 4, for the bi-encoder
vs cross-encoder distinction in full — short version: the embedding-based
hybrid_search.py retrieves a broad candidate set cheaply, this module
re-scores that smaller set more precisely by processing query+chunk
together, then we keep only the final top_k that actually go into the prompt.

Uses a local sentence-transformers cross-encoder (no extra API cost/latency
dependency on OpenAI for this step) — cross-encoder models are small enough
to run on CPU at this scale.
"""

from sentence_transformers import CrossEncoder
from app.config import settings

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        # ms-marco-MiniLM is a standard, small, well-validated reranking model —
        # trained specifically on query-passage relevance (MS MARCO dataset),
        # which is exactly the task we're using it for here.
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    candidates: list of dicts with at least a "text" key (from hybrid_search).
    Returns the top_k candidates re-sorted by cross-encoder relevance score,
    with the score attached for transparency/debugging.
    """
    if not settings.USE_RERANKER or not candidates:
        return candidates[:top_k]

    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    results = []
    for candidate, score in scored_candidates[:top_k]:
        candidate["rerank_score"] = float(score)
        results.append(candidate)
    return results
