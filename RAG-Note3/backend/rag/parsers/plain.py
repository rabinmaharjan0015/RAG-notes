"""TXT / Markdown parser: paragraphs, headings, pipe tables and $$ equations."""

import re
from pathlib import Path
from typing import List

from .common import attach_captions, is_equation_text, markdown_to_rows, rows_to_markdown


def parse_text(path: Path) -> List[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    content: List[dict] = []
    section = None
    para: List[str] = []
    i = 0

    def flush():
        nonlocal para
        text = " ".join(l.strip() for l in para).strip()
        para = []
        if not text:
            return
        if is_equation_text(text):
            content.append({"type": "equation", "latex": text, "text": "", "page_idx": 0})
        else:
            content.append({"type": "text", "text": text, "page_idx": 0, "section": section})

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # $$ equation $$   (single line or block)
        if stripped.startswith("$$"):
            flush()
            body = stripped.strip("$").strip()
            if stripped.count("$$") < 2 and not body:
                buf = []
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    buf.append(lines[i])
                    i += 1
                body = " ".join(buf).strip()
            if body:
                content.append({"type": "equation", "latex": body, "text": "", "page_idx": 0})
            i += 1
            continue

        # pipe table: header row + separator row
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$", lines[i + 1]):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = markdown_to_rows("\n".join(block))
            if len(rows) >= 2:
                content.append({"type": "table", "table_body": rows_to_markdown(rows),
                                "table_caption": [], "table_footnote": [], "page_idx": 0})
            continue

        m = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if m:
            flush()
            section = m.group(1).strip()
            content.append({"type": "text", "text": section, "page_idx": 0, "section": section})
            i += 1
            continue

        if not stripped:
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return attach_captions(content)
