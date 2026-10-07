"""Optional Tesseract OCR, used only for pages whose PDF text layer is damaged."""

from functools import lru_cache
from typing import List, Optional

from ..config import OCR_ENABLED, OCR_LANGS, logger

_DEVANAGARI = ("nep", "hin", "mar", "san")


@lru_cache(maxsize=1)
def ocr_languages() -> Optional[str]:
    """Tesseract language string to use, or None if OCR cannot read Devanagari here."""
    if not OCR_ENABLED:
        return None
    try:
        import pytesseract

        installed = set(pytesseract.get_languages(config=""))
    except Exception as e:
        logger.info(f"[ocr] Tesseract not available: {e}")
        return None
    wanted = [l for l in OCR_LANGS.split("+") if l in installed]
    if not set(wanted) & set(_DEVANAGARI):
        for l in _DEVANAGARI:                     # any installed Devanagari model beats none
            if l in installed:
                wanted.insert(0, l)
                break
    if not set(wanted) & set(_DEVANAGARI):
        logger.warning("[ocr] No Devanagari Tesseract data installed (need nep / hin / mar).")
        return None
    if "eng" in installed and "eng" not in wanted:
        wanted.append("eng")
    return "+".join(wanted)


def ocr_blocks(image, langs: str, scale: float) -> List[dict]:
    """OCR a PIL image into paragraphs: [{"top": y_in_pdf_points, "text": str}]."""
    import pytesseract

    data = pytesseract.image_to_data(image, lang=langs, output_type=pytesseract.Output.DICT)
    paras = {}
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word or float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        entry = paras.setdefault(key, {"top": data["top"][i], "lines": {}})
        entry["top"] = min(entry["top"], data["top"][i])
        entry["lines"].setdefault(data["line_num"][i], []).append(word)
    out = []
    for entry in paras.values():
        text = " ".join(" ".join(words) for _, words in sorted(entry["lines"].items())).strip()
        if text:
            out.append({"top": entry["top"] / scale, "text": text})
    return sorted(out, key=lambda b: b["top"])
