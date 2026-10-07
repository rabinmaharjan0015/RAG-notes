"""Helpers shared by all parsers. Parsers return a RAG-Anything style `content_list`:

    {"type": "text",     "text": str, "page_idx": int, "section": str|None}
    {"type": "image",    "img_path": str, "image_caption": [str], "image_footnote": [str], "page_idx": int}
    {"type": "table",    "table_body": "<markdown>", "table_caption": [str], "table_footnote": [str], "page_idx": int}
    {"type": "equation", "latex": str, "text": str, "page_idx": int}
"""

import re
from typing import List

_FIG = r"(figure|fig\.?|chart|diagram|graph|image|photo|चित्र|नक्सा|ग्राफ)"
_TAB = r"(table|तालिका)"
_NUM = r"\s*[\dIVXivx]+[\.\:\-–)]?"
FIGURE_CAPTION_RE = re.compile(rf"^\s*{_FIG}{_NUM}", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(rf"^\s*{_TAB}{_NUM}", re.IGNORECASE)
ANY_CAPTION_RE = re.compile(rf"^\s*({_FIG}|{_TAB}){_NUM}", re.IGNORECASE)

_STRONG_MATH = set("∑∫√∞≈≠≤≥∂∇∈∀∃±×÷αβγδθλμπσωΔΣΩ^_{}\\=")
_LATEX_RE = re.compile(r"\\(frac|sum|int|sqrt|alpha|beta|gamma|mathbb|mathbf|begin|cdot|times|left|right)|\$\$")


def rows_to_markdown(rows: List[List]) -> str:
    """List of rows -> GitHub markdown table (first row is the header)."""
    clean = []
    for r in rows:
        clean.append([" ".join(str(c if c is not None else "").split()).replace("|", "\\|") for c in r])
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    header, body = clean[0], clean[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def markdown_to_rows(md: str) -> List[List[str]]:
    rows = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip("|"))]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # separator row
        rows.append(cells)
    return rows


def is_equation_text(s: str) -> bool:
    """Heuristic: short line that is mostly math symbols / LaTeX."""
    s = s.strip()
    if len(s) < 3 or len(s) > 180:
        return False
    if _LATEX_RE.search(s):
        return True
    strong = sum(c in _STRONG_MATH for c in s)
    long_words = sum(1 for w in re.findall(r"[A-Za-z]{6,}", s))
    if long_words > 2:
        return False
    if strong >= 2 and strong / len(s) > 0.08:
        return True
    return "=" in s and sum(c in "()/^_*" for c in s) >= 3 and len(s) <= 120


def attach_captions(items: List[dict]) -> List[dict]:
    """Move a caption-like text block next to an image/table into that item's caption.

    Figures are usually captioned below (next block), tables above (previous block).
    """
    drop = set()
    for i, it in enumerate(items):
        if it["type"] == "image":
            key, rx, order = "image_caption", FIGURE_CAPTION_RE, (i + 1, i - 1)
        elif it["type"] == "table":
            key, rx, order = "table_caption", TABLE_CAPTION_RE, (i - 1, i + 1)
        else:
            continue
        for j in order:
            if j < 0 or j >= len(items) or j in drop:
                continue
            nb = items[j]
            if (nb["type"] == "text" and len(nb["text"]) <= 250 and rx.match(nb["text"])
                    and nb.get("page_idx") == it.get("page_idx")):
                it[key] = [nb["text"].strip()]
                drop.add(j)
                break
    return [it for k, it in enumerate(items) if k not in drop]
