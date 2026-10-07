from fastapi import APIRouter

from ..config import EMBEDDING_MODEL, QA_MODEL
from ..store import documents_store

router = APIRouter()


@router.get("/")
async def root():
    return {
        "name": "RAG Note",
        "version": "5.0.0",
        "embedding_model": EMBEDDING_MODEL,
        "generator": QA_MODEL or None,
        "documents": len(documents_store),
    }


@router.get("/api/stats")
async def stats():
    return {
        "documents": len(documents_store),
        "nepali_documents": sum(1 for d in documents_store.values() if d.get("is_nepali")),
        "total_chunks": sum(len(d["chunks"]) for d in documents_store.values()),
        "images": sum(d.get("item_counts", {}).get("image", 0) for d in documents_store.values()),
        "tables": sum(d.get("item_counts", {}).get("table", 0) for d in documents_store.values()),
        "equations": sum(d.get("item_counts", {}).get("equation", 0) for d in documents_store.values()),
        "embedding_model": EMBEDDING_MODEL,
        "generator": QA_MODEL or None,
    }
