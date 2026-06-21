from fastapi import APIRouter
from app.retrieval.vector_store import get_collection

router = APIRouter()


@router.get("/health")
def health():
    try:
        collection = get_collection()
        count = collection.count()
        return {"status": "ok", "indexed_chunks": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
