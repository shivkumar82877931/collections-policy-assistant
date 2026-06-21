"""
ChromaDB wrapper. Single place that owns the collection lifecycle, so the
rest of the app never talks to Chroma's API directly — if we ever swap to
pgvector or a managed service (guide file 03, section 3), only this file changes.
"""

import chromadb
from app.config import settings
from app.ingestion.chunking import Chunk

_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)


def get_collection(reset: bool = False):
    if reset:
        try:
            _client.delete_collection(settings.COLLECTION_NAME)
        except Exception:
            pass
    return _client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # explicit distance metric — see guide file 02 section 1
    )


def index_chunks(chunks: list[Chunk], embeddings: list[list[float]]):
    collection = get_collection()
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.raw_text for c in chunks],  # what the LLM/user actually sees
        metadatas=[
            {
                # Chroma metadata values must be str/int/float/bool — flatten accordingly
                "title": c.metadata.get("title", ""),
                "section": c.metadata.get("section", ""),
                "source_file": c.metadata.get("source_file", ""),
                "version": c.metadata.get("version", ""),
                "effective_date": c.metadata.get("effective_date", ""),
                "is_current": bool(c.metadata.get("is_current", True)),
                "embedded_text": c.text,  # the contextually-prefixed text, kept for debugging/inspection
            }
            for c in chunks
        ],
    )


def query_collection(
    query_embedding: list[float],
    top_k: int,
    only_current: bool = True,
):
    """
    Dense vector query, with an optional metadata pre-filter for is_current.
    This is the concrete implementation of the pre-filtering pattern from
    guide file 03 section 4 — filtering out superseded policy versions
    BEFORE similarity search runs, not after.
    """
    collection = get_collection()
    where_filter = {"is_current": True} if only_current else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    return [
        {
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
        }
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def get_all_documents_for_bm25():
    """
    Returns every indexed chunk's id/text, used to build the in-memory BM25
    index at startup (see hybrid_search.py). For a corpus this size, rebuilding
    BM25 in memory at startup is the right call — see hybrid_search.py for the
    scale note on when this stops being true.
    """
    collection = get_collection()
    results = collection.get()
    return list(zip(results["ids"], results["documents"]))
