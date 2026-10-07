"""In-memory document store mirrored to disk (JSON + .npy embeddings)."""

import asyncio
import json
from typing import Dict

import numpy as np

import shutil

from .config import STORE_FILE, EMB_DIR, MEDIA_DIR, UPLOAD_DIR, logger
from .graph import knowledge_graph
from .model_loader import embed_passages

documents_store: Dict[str, dict] = {}      # id -> document (chunks included)
embeddings_store: Dict[str, np.ndarray] = {}  # id -> [n_chunks, dim]
store_lock = asyncio.Lock()


def save_store() -> None:
    tmp = STORE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(documents_store, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STORE_FILE)


def load_store() -> None:
    if not STORE_FILE.exists():
        return
    try:
        data = json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Could not read store: {e}")
        return

    missing = []
    for doc_id, doc in data.items():
        documents_store[doc_id] = doc
        npy = EMB_DIR / f"{doc_id}.npy"
        if npy.exists():
            embeddings_store[doc_id] = np.load(npy)
        else:
            missing.append(doc_id)

    for doc_id in missing:  # re-embed only what is missing
        chunks = documents_store[doc_id]["chunks"]
        if chunks:
            vecs = embed_passages(chunks)
            np.save(EMB_DIR / f"{doc_id}.npy", vecs)
            embeddings_store[doc_id] = vecs

    for doc in documents_store.values():       # older saves had no item metadata
        if "items" not in doc:
            doc["items"] = [{"type": "text", "page": 0, "section": None, "media": None,
                             "caption": "", "raw": "", "description": ""} for _ in doc["chunks"]]
    knowledge_graph.rebuild(documents_store)
    logger.info(f"[store] Restored {len(documents_store)} document(s) from disk")


def remove_document(doc_id: str) -> dict:
    """Remove a document from memory and disk. Caller must hold store_lock."""
    doc = documents_store.pop(doc_id)
    embeddings_store.pop(doc_id, None)
    (EMB_DIR / f"{doc_id}.npy").unlink(missing_ok=True)
    for f in UPLOAD_DIR.glob(f"{doc_id}*"):
        f.unlink(missing_ok=True)
    shutil.rmtree(MEDIA_DIR / doc_id, ignore_errors=True)
    knowledge_graph.rebuild(documents_store)
    return doc


def clear_all() -> int:
    """Remove every document. Caller must hold store_lock."""
    count = len(documents_store)
    documents_store.clear()
    embeddings_store.clear()
    for f in EMB_DIR.glob("*.npy"):
        f.unlink(missing_ok=True)
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)
    for d in MEDIA_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    knowledge_graph.clear()
    return count
