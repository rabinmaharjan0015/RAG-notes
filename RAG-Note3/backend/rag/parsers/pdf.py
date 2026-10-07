"""PDF parser (pdfplumber): reading-order text, tables, figures; OCR fallback for damaged Nepali pages."""

from pathlib import Path
from typing import List

from ..config import MIN_IMAGE_PT, OCR_DPI, logger
from .common import (
    rows_to_markdown, is_equation_text, attach_captions, clean_text, devanagari_damage,
)
from .ocr import ocr_languages, ocr_blocks


def _inside(obj, boxes) -> bool:
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _clamp(bbox, page):
    x0, top, x1, bottom = bbox
    px0, ptop, px1, pbot = page.bbox
    return (max(x0, px0), max(top, ptop), min(x1, px1), min(bottom, pbot))


def _paragraphs(lines: List[dict]) -> List[dict]:
    """Group text lines into paragraphs using vertical gaps."""
    paras, cur = [], None
    for ln in lines:
        text = clean_text(ln["text"]).strip()
        if not text:
            continue
        h = max(ln["bottom"] - ln["top"], 1)
        math_break = cur is not None and is_equation_text(cur["text"]) != is_equation_text(text)
        if cur and not math_break and ln["top"] - cur["bottom"] <= 0.9 * h:
            joiner = "" if cur["text"].endswith("-") and not cur["text"].endswith(" -") else " "
            cur["text"] = (cur["text"][:-1] if joiner == "" else cur["text"]) + joiner + text
            cur["bottom"] = ln["bottom"]
        else:
            if cur:
                paras.append(cur)
            cur = {"text": text, "top": ln["top"], "bottom": ln["bottom"]}
    if cur:
        paras.append(cur)
    return paras


def _valid_table(table, rows, page) -> bool:
    """Reject page-layout artefacts that pdfplumber mistakes for tables.

    Printed web pages often draw the whole page as one huge box (observed: 743% of the page,
    starting above the page) whose single cell holds all the text.
    """
    if not rows or len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    cells = [clean_text(str(c or "")).strip() for r in rows for c in r]
    total = sum(len(c) for c in cells)
    if total == 0 or sum(1 for c in cells if c) / len(cells) < 0.15:
        return False
    if total > 300 and max(len(c) for c in cells) / total > 0.6:      # one giant cell = page text
        return False
    x0, top, x1, bottom = table.bbox
    area = (x1 - x0) * (bottom - top) / (page.width * page.height)
    if area > 2.0 and len(rows) <= 3:                                  # far larger than the page
        return False
    return True


def _page_ranges(pages: List[int]) -> str:
    pages = sorted(set(p + 1 for p in pages))
    out, start, prev = [], None, None
    for p in pages + [None]:
        if start is None:
            start = prev = p
        elif p is not None and p == prev + 1:
            prev = p
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = p
    return ", ".join(out)


def parse_pdf(path: Path, media_dir: Path) -> List[dict]:
    import pdfplumber

    media_dir.mkdir(parents=True, exist_ok=True)
    content: List[dict] = []
    damaged_pages: List[int] = []
    ocr_pages: List[int] = []
    langs = None

    with pdfplumber.open(str(path)) as pdf:
        for pno, page in enumerate(pdf.pages):
            elems = []          # (top, block)
            boxes = []          # regions already taken by tables / figures

            # --- tables (validated) ---
            try:
                for t in page.find_tables():
                    rows = t.extract()
                    if _valid_table(t, rows, page):
                        boxes.append(_clamp(t.bbox, page))
                        elems.append((max(t.bbox[1], 0), {
                            "type": "table", "table_body": rows_to_markdown(rows),
                            "table_caption": [], "table_footnote": [], "page_idx": pno,
                        }))
            except Exception as e:
                logger.warning(f"[pdf] table detection failed on page {pno + 1}: {e}")

            # --- figures ---
            for k, im in enumerate(page.images):
                x0, top, x1, bottom = im["x0"], im["top"], im["x1"], im["bottom"]
                if (x1 - x0) < MIN_IMAGE_PT or (bottom - top) < MIN_IMAGE_PT:
                    continue
                bbox = _clamp((x0, top, x1, bottom), page)
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

            # --- text outside tables / figures ---
            try:
                filtered = page.filter(lambda o: o.get("object_type") != "char" or not _inside(o, boxes))
                lines = filtered.extract_text_lines(return_chars=False)
            except Exception as e:
                logger.warning(f"[pdf] text extraction failed on page {pno + 1}: {e}")
                lines = []

            raw = "\n".join(l["text"] for l in lines)
            if devanagari_damage(raw):
                if langs is None:
                    langs = ocr_languages() or ""
                if langs:
                    try:
                        scale = OCR_DPI / 72.0
                        img = page.to_image(resolution=OCR_DPI).original.convert("RGB")
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(img)
                        for b in boxes:                       # keep tables/figures out of the OCR text
                            draw.rectangle([b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale], fill="white")
                        text_elems = [(b["top"], {"type": "text", "text": b["text"], "page_idx": pno})
                                      for b in ocr_blocks(img, langs, scale)]
                        ocr_len = sum(len(e[1]["text"]) for e in text_elems)
                        if text_elems and ocr_len >= 0.3 * len(clean_text(raw)):   # OCR must recover most of the page
                            elems.extend(text_elems)
                            ocr_pages.append(pno)
                            lines = []
                        else:
                            damaged_pages.append(pno)
                    except Exception as e:
                        logger.warning(f"[pdf] OCR failed on page {pno + 1}: {e}")
                        damaged_pages.append(pno)
                else:
                    damaged_pages.append(pno)

            for para in _paragraphs(lines):
                txt = para["text"]
                if is_equation_text(txt):
                    elems.append((para["top"], {"type": "equation", "latex": txt, "text": "", "page_idx": pno}))
                else:
                    elems.append((para["top"], {"type": "text", "text": txt, "page_idx": pno}))

            elems.sort(key=lambda e: e[0])
            content.extend(attach_captions([e[1] for e in elems]))

    if ocr_pages:
        content.append({"type": "warning", "code": "ocr_used", "text":
            f"Page(s) {_page_ranges(ocr_pages)} had a damaged Nepali text layer (missing conjuncts); "
            f"they were re-read with OCR ({langs})."})
    if damaged_pages:
        content.append({"type": "warning", "code": "damaged_text", "text":
            f"Page(s) {_page_ranges(damaged_pages)} have a damaged Nepali text layer: conjuncts such as "
            f"र्य and क्ष are missing from the PDF itself, so searches may miss words and answers may show "
            f"broken spellings. Install Tesseract with Nepali data (tesseract-ocr-nep) and pytesseract to "
            f"recover them automatically."})
    return content
