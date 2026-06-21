"""
Hybrid search: dense (vector) + sparse (BM25), combined via Reciprocal Rank
Fusion (RRF). See guide file 04, section 3, for the full rationale —
short version: dense retrieval misses exact-match terms (regulation
citations like "1692c(a)(1)", specific dollar figures, version numbers)
because embeddings encode semantic similarity, not exact lexical match.
BM25 catches exactly those cases. RRF combines the two ranked lists
without needing to normalize cosine similarity and BM25 scores onto the
same scale — it works purely on rank position.

Scale note: rank_bm25 builds an in-memory index, rebuilt at startup from
whatever's in the vector store. Fine for this corpus (a handful of
documents). At real production document volume, you'd move BM25 to a
proper search engine (e.g., Elasticsearch/OpenSearch) rather than holding
it in process memory — flagging this explicitly because "would this
design choice survive 100x scale" is exactly the kind of follow-up
question guide file 08 section 3 prepares you for.
"""

from rank_bm25 import BM25Okapi
from app.retrieval.vector_store import query_collection, get_all_documents_for_bm25
from app.retrieval.embeddings import embed_text
from app.config import settings

_bm25_index = None
_bm25_doc_ids: list[str] = []
_bm25_doc_texts: list[str] = []


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index():
    """Called once at app startup (see routes/health.py's startup hook), and
    again any time the index is rebuilt via the ingestion script."""
    global _bm25_index, _bm25_doc_ids, _bm25_doc_texts
    docs = get_all_documents_for_bm25()
    if not docs:
        _bm25_index = None
        return
    _bm25_doc_ids, _bm25_doc_texts = zip(*docs)
    _bm25_doc_ids, _bm25_doc_texts = list(_bm25_doc_ids), list(_bm25_doc_texts)
    tokenized = [_tokenize(t) for t in _bm25_doc_texts]
    _bm25_index = BM25Okapi(tokenized)


def _bm25_search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Returns [(doc_id, bm25_score), ...] sorted descending by score."""
    if _bm25_index is None:
        return []
    scores = _bm25_index.get_scores(_tokenize(query))
    ranked = sorted(zip(_bm25_doc_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Standard RRF: score(doc) = sum over lists of 1 / (k + rank_in_that_list).
    k=60 is the commonly-used damping constant from the original RRF paper —
    it reduces the influence of any single very-high rank from one method
    dominating the fused result.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, top_k: int, only_current: bool = True) -> list[dict]:
    """
    Runs dense + sparse retrieval in parallel (conceptually — sequential here
    for simplicity), fuses rankings with RRF, and returns full chunk records
    for the fused top_k, re-attaching text/metadata from the vector store
    results (which already carries everything we need).
    """
    query_embedding = embed_text(query, is_query=True)

    # Over-fetch from both methods so RRF has enough signal to work with —
    # fusing two top-5 lists is much weaker than fusing two top-20 lists.
    fetch_k = max(top_k * 3, 20)

    dense_results = query_collection(query_embedding, top_k=fetch_k, only_current=only_current)
    dense_ranked_ids = [r["id"] for r in dense_results]

    bm25_results = _bm25_search(query, top_k=fetch_k)
    bm25_ranked_ids = [doc_id for doc_id, _ in bm25_results]

    fused = _reciprocal_rank_fusion([dense_ranked_ids, bm25_ranked_ids])
    top_fused_ids = [doc_id for doc_id, _ in fused[:top_k]]

    # Re-attach full records (text/metadata) using the dense results as the
    # lookup table, since dense retrieval already carries everything needed —
    # for any id that came ONLY from BM25 and not dense, fall back to a
    # direct vector-store lookup by id.
    dense_by_id = {r["id"]: r for r in dense_results}
    final_results = []
    for doc_id in top_fused_ids:
        if doc_id in dense_by_id:
            final_results.append(dense_by_id[doc_id])
        else:
            # BM25-only hit — still surface it, with a placeholder distance
            # since it didn't come through the dense path.
            idx = _bm25_doc_ids.index(doc_id)
            final_results.append({
                "id": doc_id,
                "text": _bm25_doc_texts[idx],
                "metadata": {},
                "distance": None,
            })
    return final_results
