"""Document ingestion: parse -> modal items -> embeddings -> knowledge graph."""

import hashlib
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import UPLOAD_DIR, EMB_DIR, MEDIA_DIR
from .graph import knowledge_graph
from .model_loader import embed_passages
from .modal_processors import build_items
from .parsers import parse_file
from .schemas import DocumentInfo
from .store import documents_store, embeddings_store
from .text_utils import is_nepali_text


def ingest_content_list(
    filename: str,
    content_list: List[dict],
    *,
    file_id: Optional[str] = None,
    digest: Optional[str] = None,
    size: int = 0,
) -> dict:
    """Blocking. Turns a RAG-Anything style content_list into an indexed document."""
    file_id = file_id or str(uuid.uuid4())
    chunks, items = build_items(content_list, file_id)
    if not chunks:
        raise ValueError("Could not extract anything searchable from this file.")

    vecs = embed_passages(chunks)
    np.save(EMB_DIR / f"{file_id}.npy", vecs)

    text_parts = [c for c, it in zip(chunks, items) if it["type"] == "text"]
    full_text = "\n".join(text_parts) if text_parts else "\n".join(chunks)
    doc = {
        "id": file_id,
        "filename": filename,
        "size": size,
        "sha256": digest or hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "content": full_text,
        "chunks": chunks,
        "items": items,
        "chunk_count": len(chunks),
        "item_counts": dict(Counter(it["type"] for it in items)),
        "pages": max((it["page"] for it in items), default=0) + 1,
        "is_nepali": is_nepali_text(full_text),
        "content_preview": full_text[:200].replace("\n", " ").strip(),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    documents_store[file_id] = doc
    embeddings_store[file_id] = vecs
    knowledge_graph.rebuild(documents_store)
    return {**doc, "duplicate": False}


def ingest_document(filename: str, content: bytes) -> dict:
    """Blocking. Run in a thread. Raises ValueError for unreadable files."""
    ext = Path(filename).suffix.lower()
    digest = hashlib.sha256(content).hexdigest()

    for d in documents_store.values():  # identical file already indexed
        if d.get("sha256") == digest:
            return {**d, "duplicate": True}

    file_id = str(uuid.uuid4())
    fpath = UPLOAD_DIR / f"{file_id}{ext}"
    fpath.write_bytes(content)
    try:
        content_list = parse_file(fpath, MEDIA_DIR / file_id)
        if not content_list:
            raise ValueError("Could not extract any content from this file.")
        return ingest_content_list(filename, content_list, file_id=file_id, digest=digest, size=len(content))
    except Exception:
        fpath.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(MEDIA_DIR / file_id, ignore_errors=True)
        raise


def to_info(d: dict) -> DocumentInfo:
    return DocumentInfo(
        id=d["id"],
        filename=d["filename"],
        size=d["size"],
        content_preview=d["content_preview"],
        uploaded_at=d.get("uploaded_at", ""),
        chunk_count=d.get("chunk_count", len(d.get("chunks", []))),
        is_nepali=d.get("is_nepali", False),
        duplicate=d.get("duplicate", False),
        item_counts=d.get("item_counts", {"text": len(d.get("chunks", []))}),
        pages=d.get("pages", 0),
    )
