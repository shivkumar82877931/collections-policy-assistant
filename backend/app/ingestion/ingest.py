"""
Run this once (and again any time documents change) to build the index:

    python -m app.ingestion.ingest

This is the ingestion-time half of the pipeline — chunk every document,
embed every chunk (locally via Ollama, see retrieval/embeddings.py), and
write everything into Chroma. The BM25 index is rebuilt separately at app
startup (app/main.py's lifespan hook) since it's cheap to rebuild from
whatever's already in Chroma.
"""

from app.ingestion.chunking import chunk_all_documents
from app.retrieval.embeddings import embed_batch
from app.retrieval.vector_store import index_chunks, get_collection
from app.config import settings


def run_ingestion():
    print(f"Reading documents from {settings.DATA_DIR} ...")
    chunks = chunk_all_documents(settings.DATA_DIR)
    print(f"Produced {len(chunks)} chunks from the document set.")

    if not chunks:
        print("No documents found — nothing to ingest.")
        return

    print(f"Embedding {len(chunks)} chunks via {settings.EMBEDDING_MODEL} ...")
    # Reset the collection so re-running ingestion doesn't duplicate/stale entries.
    get_collection(reset=True)
    texts_to_embed = [c.text for c in chunks]  # the contextually-prefixed text
    embeddings = embed_batch(texts_to_embed)

    print("Writing to vector store ...")
    index_chunks(chunks, embeddings)

    collection = get_collection()
    print(f"Done. Collection now has {collection.count()} chunks indexed.")

    # Quick sanity printout — useful when first setting this up locally
    by_file = {}
    for c in chunks:
        by_file[c.metadata["source_file"]] = by_file.get(c.metadata["source_file"], 0) + 1
    for filename, count in by_file.items():
        print(f"  {filename}: {count} chunks")


if __name__ == "__main__":
    run_ingestion()
