import asyncio
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from ..config import ALLOWED_EXTS, MAX_DOCS, MAX_FILE_SIZE
from ..ingest import ingest_document, to_info
from ..schemas import DocumentInfo, DocumentListResponse
from ..store import (
    documents_store, store_lock, save_store, remove_document, clear_all,
)

router = APIRouter(prefix="/api")


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    if len(documents_store) >= MAX_DOCS:
        raise HTTPException(400, f"Maximum of {MAX_DOCS} documents reached.")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 50 MB).")

    async with store_lock:
        try:
            doc = await asyncio.to_thread(ingest_document, file.filename, content)
        except ValueError as e:
            raise HTTPException(400, str(e))
        await asyncio.to_thread(save_store)
    return to_info(doc)


@router.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results, errors = [], []
    async with store_lock:
        for f in files:
            ext = Path(f.filename).suffix.lower()
            if ext not in ALLOWED_EXTS:
                errors.append({"filename": f.filename, "error": f"Unsupported file type {ext}"})
                continue
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                errors.append({"filename": f.filename, "error": "File too large (max 50 MB)"})
                continue
            if len(documents_store) >= MAX_DOCS:
                errors.append({"filename": f.filename, "error": "Document limit reached"})
                continue
            try:
                doc = await asyncio.to_thread(ingest_document, f.filename, content)
                results.append(to_info(doc))
            except Exception as e:
                errors.append({"filename": f.filename, "error": str(e)})
        await asyncio.to_thread(save_store)
    return {"documents": results, "errors": errors, "total": len(results)}


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    async with store_lock:
        docs = sorted(documents_store.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True)
    items = [to_info(d) for d in docs]
    return DocumentListResponse(documents=items, total=len(items))


@router.get("/documents/{doc_id}/content")
async def get_content(doc_id: str):
    async with store_lock:
        doc = documents_store.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {
        "id": doc_id,
        "filename": doc["filename"],
        "content": doc["content"],
        "chunk_count": len(doc["chunks"]),
        "is_nepali": doc.get("is_nepali", False),
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    async with store_lock:
        if doc_id not in documents_store:
            raise HTTPException(404, "Document not found")
        doc = remove_document(doc_id)
        await asyncio.to_thread(save_store)
    return {"message": f"Deleted '{doc['filename']}'"}


@router.delete("/documents")
async def delete_all_documents():
    async with store_lock:
        count = clear_all()
        await asyncio.to_thread(save_store)
    return {"message": f"All {count} documents deleted"}
