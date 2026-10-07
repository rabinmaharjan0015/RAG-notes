"""Standalone image files -> one image block (described later by the vision model)."""

from pathlib import Path
from typing import List


def parse_image(path: Path, media_dir: Path) -> List[dict]:
    from PIL import Image

    media_dir.mkdir(parents=True, exist_ok=True)
    out = media_dir / "image1.png"
    with Image.open(path) as im:
        im.convert("RGB").save(out)
    return [{"type": "image", "img_path": str(out), "image_caption": [], "image_footnote": [], "page_idx": 0}]
