"""Modal processors (RAG-Anything style): turn text / image / table / equation blocks into
retrievable items. Each item has searchable `text` plus display metadata.

Item metadata (stored per chunk):  type, page, section, media, caption, raw, description
"""

import shutil
from pathlib import Path
from typing import List, Tuple

from .config import (
    MEDIA_DIR, CONTEXT_CHARS, TABLE_ROWS_PER_ITEM, MAX_VLM_IMAGES_PER_DOC, logger,
)
from .model_loader import vlm_generate, vlm_available
from .parsers.common import markdown_to_rows, rows_to_markdown
from .text_utils import smart_chunk

IMAGE_PROMPT = (
    "Describe this image for a search index. State what kind of image it is (chart, diagram, "
    "photo, screenshot, scanned text...), read any visible text, labels, numbers and axis titles, "
    "and summarise what it shows. Be factual and concise."
)


# ─── Individual modalities (reused for query-time attachments) ──
def describe_image(img_path: str, caption: str = "", footnote: str = "", use_vlm: bool = True) -> Tuple[str, str]:
    """Return (searchable_text, vlm_description)."""
    vlm_desc = ""
    if use_vlm and vlm_available():
        vlm_desc = vlm_generate(img_path, IMAGE_PROMPT) or ""
    parts = []
    if caption:
        parts.append(caption)
    if vlm_desc:
        parts.append(vlm_desc)
    if footnote:
        parts.append(footnote)
    return " ".join(parts).strip(), vlm_desc


def describe_table_groups(table_md: str, caption: str = "") -> List[Tuple[str, str]]:
    """Split a table into row groups. Returns [(searchable_text, markdown_of_group)]."""
    rows = markdown_to_rows(table_md)
    if len(rows) < 2:
        return []
    header, body = rows[0], rows[1:]
    out = []
    for start in range(0, len(body), TABLE_ROWS_PER_ITEM):
        group = body[start:start + TABLE_ROWS_PER_ITEM]
        serial = []
        for r in group:
            serial.append("; ".join(f"{h}: {v}" for h, v in zip(header, r) if v))
        end = start + len(group)
        label = f"Table: {caption}." if caption else "Table."
        span = f" (rows {start + 1}-{end} of {len(body)})" if len(body) > TABLE_ROWS_PER_ITEM else ""
        text = f"{label}{span} Columns: {', '.join(h for h in header if h)}. " + " | ".join(serial)
        out.append((text, rows_to_markdown([header] + group)))
    return out


def describe_equation(latex: str, caption: str = "") -> str:
    latex = " ".join(latex.split())
    return (f"Equation: {caption}. " if caption else "Equation. ") + f"Formula: {latex}"


# ─── content_list -> items ──────────────────────────────────────
def _localize_image(img_path: str, doc_id: str) -> Path:
    """Make sure the image lives under MEDIA_DIR/<doc_id>/ so it can be served."""
    src = Path(img_path)
    dest_dir = MEDIA_DIR / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        src.resolve().relative_to(dest_dir.resolve())
        return src
    except ValueError:
        if not src.is_file():
            raise ValueError(f"Image not found (use an absolute path): {img_path}")
        dest = dest_dir / src.name
        if dest.exists():
            dest = dest_dir / f"{abs(hash(str(src))) % 10**8}_{src.name}"
        shutil.copy(src, dest)
        return dest


def build_items(content_list: List[dict], doc_id: str) -> Tuple[List[str], List[dict], List[dict]]:
    """Return (chunks, items, warnings[{code,text}]): chunk i is the searchable text of item i."""
    chunks: List[str] = []
    items: List[dict] = []
    warnings: List[dict] = []
    buf: List[str] = []
    buf_key = None            # (page, section) of the text currently buffered
    recent_text = ""          # surrounding text for multimodal items
    vlm_used = 0

    def add(text: str, **meta):
        chunks.append(text)
        items.append({"type": meta.pop("type"), "page": meta.pop("page", 0), "section": meta.pop("section", None),
                      "media": None, "caption": "", "raw": "", "description": "", **meta})

    def flush():
        nonlocal buf, buf_key
        if buf:
            page, section = buf_key
            for ch in smart_chunk("\n".join(buf)):
                add(ch, type="text", page=page, section=section)
        buf, buf_key = [], None

    for block in content_list:
        kind = block.get("type", "text")
        page = int(block.get("page_idx", 0) or 0)
        section = block.get("section")
        context = recent_text[-CONTEXT_CHARS:].strip()

        if kind == "warning":
            warnings.append({"code": block.get("code", "info"), "text": str(block.get("text", ""))})
            continue

        if kind == "text":
            text = (block.get("text") or "").strip()
            if not text:
                continue
            key = (page, section)
            if buf_key is not None and key != buf_key:
                flush()
            buf_key = key
            buf.append(text)
            recent_text = (recent_text + " " + text)[-2 * CONTEXT_CHARS:]
            continue

        flush()

        if kind == "image":
            try:
                path = _localize_image(block["img_path"], doc_id)
            except Exception as e:
                logger.warning(f"[ingest] skipping image: {e}")
                continue
            caption = " ".join(block.get("image_caption") or []).strip()
            footnote = " ".join(block.get("image_footnote") or []).strip()
            use_vlm = vlm_used < MAX_VLM_IMAGES_PER_DOC
            text, vlm_desc = describe_image(str(path), caption, footnote, use_vlm=use_vlm)
            if vlm_desc:
                vlm_used += 1
            if not text:
                continue                          # no caption and no AI description: nothing to search on (logos, icons)
            body = text
            if context:                           # context-aware: link the figure to its surrounding text
                body = f"{body} (near: {context})"
            add(f"Figure: {body}", type="image", page=page, section=section,
                media=f"{doc_id}/{path.name}", caption=caption, description=vlm_desc)

        elif kind == "table":
            caption = " ".join(block.get("table_caption") or []).strip()
            groups = describe_table_groups(block.get("table_body", ""), caption)
            for text, md in groups:
                add(text, type="table", page=page, section=section, caption=caption, raw=md)

        elif kind == "equation":
            latex = (block.get("latex") or block.get("text") or "").strip()
            if not latex:
                continue
            desc = (block.get("text") or "").strip() if block.get("latex") else ""
            body = describe_equation(latex, desc)
            if context:
                body += f" (near: {context})"
            add(body, type="equation", page=page, section=section, raw=latex, caption=desc)

        else:  # generic / custom content type
            text = str(block.get("content") or block.get("text") or "").strip()
            if text:
                add(text, type="text", page=page, section=section)

    flush()
    return chunks, items, warnings
