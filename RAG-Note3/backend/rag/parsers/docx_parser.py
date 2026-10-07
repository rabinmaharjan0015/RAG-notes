"""DOCX parser: headings, paragraphs, tables, inline images and Word equations, in document order."""

import re
from pathlib import Path
from typing import List

from ..config import logger
from .common import rows_to_markdown, attach_captions, is_equation_text

_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/gif": "gif",
        "image/bmp": "bmp", "image/webp": "webp"}


def parse_docx(path: Path, media_dir: Path) -> List[dict]:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    media_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(str(path))
    content: List[dict] = []
    section = None
    n_img = 0

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            para = Paragraph(child, doc)

            # inline images
            for blip in child.iter(qn("a:blip")):
                rid = blip.get(qn("r:embed"))
                part = doc.part.related_parts.get(rid) if rid else None
                ext = _EXT.get(getattr(part, "content_type", ""), None)
                if part is None or ext is None:
                    continue  # EMF/WMF etc. cannot be shown in a browser
                n_img += 1
                out = media_dir / f"img{n_img}.{ext}"
                out.write_bytes(part.blob)
                content.append({"type": "image", "img_path": str(out), "image_caption": [],
                                "image_footnote": [], "page_idx": 0})

            # Word (OMML) equations
            for om in child.iter(qn("m:oMath")):
                latex = "".join(t.text or "" for t in om.iter(qn("m:t"))).strip()
                if latex:
                    content.append({"type": "equation", "latex": latex, "text": "", "page_idx": 0})

            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "") if para.style is not None else ""
            if style.lower().startswith(("heading", "title")):
                section = text
            if is_equation_text(text):
                content.append({"type": "equation", "latex": text, "text": "", "page_idx": 0})
            else:
                content.append({"type": "text", "text": text, "page_idx": 0, "section": section})

        elif tag == "tbl":
            try:
                table = Table(child, doc)
                rows = []
                for row in table.rows:
                    cells, prev = [], None
                    for c in row.cells:
                        if c._tc is prev:        # merged cell repeats: keep once
                            continue
                        prev = c._tc
                        cells.append(c.text)
                    rows.append(cells)
                if rows and len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    content.append({"type": "table", "table_body": rows_to_markdown(rows),
                                    "table_caption": [], "table_footnote": [], "page_idx": 0})
            except Exception as e:
                logger.warning(f"[docx] table skipped: {e}")

    return attach_captions(content)
