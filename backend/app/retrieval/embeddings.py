"""
Embedding generation via Ollama (local, free, no API key).

Important difference from the OpenAI version this replaced: nomic-embed-text
is an ASYMMETRIC embedding model — it expects a "search_query: " or
"search_document: " prefix depending on whether you're embedding a query or
a document/chunk. Using the wrong prefix (or no prefix) silently degrades
retrieval quality — this is a real, documented failure mode, not a
theoretical one (see the original learning guide, 02_Embeddings.md, section 3,
failure mode #3). is_query is therefore NOT optional here the way it was
cosmetic for OpenAI's symmetric models.

Requires Ollama running locally with the model pulled:
    ollama pull nomic-embed-text
"""

import ollama
from app.config import settings

_client = ollama.Client(host=settings.OLLAMA_HOST)


def embed_text(text: str, is_query: bool = False) -> list[float]:
    prefix = "search_query: " if is_query else "search_document: "
    response = _client.embeddings(model=settings.EMBEDDING_MODEL, prompt=prefix + text)
    return response["embedding"]


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Ollama's embeddings endpoint doesn't support true batch requests the way
    OpenAI's does — we loop here. For this project's corpus size (a few
    dozen chunks) this is fine; at real scale you'd want a queue/parallelism
    layer, but that's true of any local-inference setup, not specific to
    this design choice.
    """
    return [embed_text(t, is_query=is_query) for t in texts]
