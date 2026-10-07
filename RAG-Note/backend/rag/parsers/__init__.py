"""Parser dispatch: file -> RAG-Anything style content_list."""

from pathlib import Path
from typing import List

from ..config import IMAGE_EXTS
from .docx_parser import parse_docx
from .image import parse_image
from .pdf import parse_pdf
from .plain import parse_text


def parse_file(path: Path, media_dir: Path) -> List[dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, media_dir)
    if ext == ".docx":
        return parse_docx(path, media_dir)
    if ext in (".txt", ".md"):
        return parse_text(path)
    if ext in IMAGE_EXTS:
        return parse_image(path, media_dir)
    raise ValueError(f"Unsupported file type: {ext}")
