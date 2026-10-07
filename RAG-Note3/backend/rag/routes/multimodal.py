"""RAG-Anything style extras: multimodal queries, direct content_list insertion, graph inspection."""

import asyncio
import csv
import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import IMAGE_EXTS, MAX_FILE_SIZE
from ..graph import knowledge_graph
from ..ingest import ingest_content_list, to_info
from ..modal_processors import describe_equation, describe_image, describe_table_groups
from ..model_loader import vlm_available
from ..parsers.common import rows_to_markdown
from ..qa import answer_question
from ..schemas import AnswerResponse, ContentListRequest, DocumentInfo, QuestionRequest
from ..store import documents_store, store_lock, save_store

router = APIRouter(prefix="/api")


def _build_aux(image_path: Optional[Path], table_data: str, latex: str) -> str:
    """Turn the user's attachment(s) into searchable text."""
    parts = []
    if image_path is not None:
        _, vlm_desc = describe_image(str(image_path))
        if not vlm_desc:
            raise HTTPException(400, "The vision model is not available, so the image cannot be understood. "
                                     "Check ENABLE_VLM / HF_VLM_MODEL in backend/.env.")
        parts.append(vlm_desc)
    if table_data.strip():
        md = table_data.strip()
        if "|" not in md:                      # CSV -> markdown
            rows = list(csv.reader(io.StringIO(md)))
            md = rows_to_markdown(rows) if len(rows) >= 2 else md
        groups = describe_table_groups(md)
        parts.extend(t for t, _ in groups[:2])
    if latex.strip():
        parts.append(describe_equation(latex.strip()))
    return " ".join(parts)


@router.post("/ask-multimodal", response_model=AnswerResponse)
async def ask_multimodal(
    question: str = Form(...),
    mode: str = Form("hybrid"),
    document_ids: str = Form(""),
    vlm_enhanced: bool = Form(False),
    table_data: str = Form(""),
    latex: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """Ask a question together with an image, a table (markdown/CSV) and/or an equation (LaTeX)."""
    if not documents_store:
        raise HTTPException(400, "No documents uploaded yet.")
    tmp_path = None
    try:
        if image is not None and image.filename:
            ext = Path(image.filename).suffix.lower()
            if ext not in IMAGE_EXTS:
                raise HTTPException(400, f"Unsupported image type: {ext}")
            data = await image.read()
            if len(data) > MAX_FILE_SIZE:
                raise HTTPException(413, "Image too large.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                f.write(data)
                tmp_path = Path(f.name)
        aux = await asyncio.to_thread(_build_aux, tmp_path, table_data, latex)
        ids = [d for d in document_ids.split(",") if d] or None
        req = QuestionRequest(question=question, document_ids=ids, mode=mode, vlm_enhanced=vlm_enhanced)
        return await asyncio.to_thread(answer_question, req, aux)
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


@router.post("/insert-content-list", response_model=DocumentInfo)
async def insert_content_list(req: ContentListRequest):
    """Index a pre-parsed content_list (text / image / table / equation blocks) without parsing a file."""
    async with store_lock:
        try:
            doc = await asyncio.to_thread(ingest_content_list, req.file_name, req.content_list)
        except ValueError as e:
            raise HTTPException(400, str(e))
        await asyncio.to_thread(save_store)
    return to_info(doc)


@router.get("/documents/{doc_id}/items")
async def list_items(doc_id: str):
    doc = documents_store.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {
        "id": doc_id, "filename": doc["filename"],
        "items": [{"index": i, "text": c[:300], **it} for i, (c, it) in enumerate(zip(doc["chunks"], doc["items"]))],
    }


@router.get("/graph")
async def graph_summary(limit: int = 40):
    return knowledge_graph.summary(limit=max(5, min(limit, 200)))


@router.get("/capabilities")
async def capabilities():
    from ..config import EMBEDDING_MODEL, VLM_MODEL, QA_MODEL
    return {"embedding_model": EMBEDDING_MODEL, "vlm_model": VLM_MODEL if vlm_available() else None,
            "llm_model": QA_MODEL or None, "modes": ["naive", "local", "global", "hybrid"]}
