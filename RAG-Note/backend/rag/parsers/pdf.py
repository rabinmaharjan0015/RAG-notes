"""PDF parser (pdfplumber): reading-order text, tables, and figures (cropped as PNG)."""

from pathlib import Path
from typing import List

from ..config import MIN_IMAGE_PT, logger
from .common import (
    rows_to_markdown, is_equation_text, attach_captions, FIGURE_CAPTION_RE, TABLE_CAPTION_RE,
)


def _inside(obj, boxes) -> bool:
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _paragraphs(lines: List[dict]) -> List[dict]:
    """Group text lines into paragraphs using vertical gaps."""
    paras, cur = [], None
    for ln in lines:
        h = max(ln["bottom"] - ln["top"], 1)
        math_break = cur is not None and is_equation_text(cur["text"]) != is_equation_text(ln["text"])
        if cur and not math_break and ln["top"] - cur["bottom"] <= 0.9 * h:
            joiner = "" if cur["text"].endswith("-") and not cur["text"].endswith(" -") else " "
            cur["text"] = (cur["text"][:-1] if joiner == "" else cur["text"]) + joiner + ln["text"].strip()
            cur["bottom"] = ln["bottom"]
        else:
            if cur:
                paras.append(cur)
            cur = {"text": ln["text"].strip(), "top": ln["top"], "bottom": ln["bottom"]}
    if cur:
        paras.append(cur)
    return paras


def parse_pdf(path: Path, media_dir: Path) -> List[dict]:
    import pdfplumber

    media_dir.mkdir(parents=True, exist_ok=True)
    content: List[dict] = []

    with pdfplumber.open(str(path)) as pdf:
        for pno, page in enumerate(pdf.pages):
            elems = []          # (top, block)
            boxes = []          # regions already taken by tables / figures

            # --- tables ---
            try:
                for t in page.find_tables():
                    rows = t.extract()
                    if rows and len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                        boxes.append(t.bbox)
                        elems.append((t.bbox[1], {
                            "type": "table", "table_body": rows_to_markdown(rows),
                            "table_caption": [], "table_footnote": [], "page_idx": pno,
                        }))
            except Exception as e:
                logger.warning(f"[pdf] table detection failed on page {pno + 1}: {e}")

            # --- figures (embedded images, rendered from the page so format never matters) ---
            px0, ptop, px1, pbot = page.bbox
            for k, im in enumerate(page.images):
                x0, top, x1, bottom = im["x0"], im["top"], im["x1"], im["bottom"]
                if (x1 - x0) < MIN_IMAGE_PT or (bottom - top) < MIN_IMAGE_PT:
                    continue
                bbox = (max(x0, px0), max(top, ptop), min(x1, px1), min(bottom, pbot))
                if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
                    continue
                out = media_dir / f"p{pno + 1}_img{k + 1}.png"
                try:
                    page.crop(bbox).to_image(resolution=150).save(str(out))
                except Exception as e:
                    logger.warning(f"[pdf] could not render image on page {pno + 1}: {e}")
                    continue
                boxes.append(bbox)
                elems.append((bbox[1], {
                    "type": "image", "img_path": str(out), "image_caption": [],
                    "image_footnote": [], "page_idx": pno,
                }))

            # --- text (everything outside tables / figures) ---
            try:
                filtered = page.filter(lambda o: o.get("object_type") != "char" or not _inside(o, boxes))
                lines = filtered.extract_text_lines(return_chars=False)
            except Exception as e:
                logger.warning(f"[pdf] text extraction failed on page {pno + 1}: {e}")
                lines = []
            for para in _paragraphs(lines):
                txt = para["text"]
                if not txt:
                    continue
                if is_equation_text(txt):
                    elems.append((para["top"], {"type": "equation", "latex": txt, "text": "", "page_idx": pno}))
                else:
                    elems.append((para["top"], {"type": "text", "text": txt, "page_idx": pno}))

            elems.sort(key=lambda e: e[0])
            content.extend(attach_captions([e[1] for e in elems]))

    return content
