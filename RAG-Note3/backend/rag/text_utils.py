"""Sentence splitting, chunking, tokenisation and question analysis (Devanagari + Latin)."""

import re
from pathlib import Path
from typing import List

from .config import CHUNK_CHARS, CHUNK_OVERLAP, logger

# ─── Script detection ───────────────────────────────────────────
def is_nepali_text(text: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in text)


# ─── Sentences & chunks ─────────────────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=।)\s*|(?<=[\.!\?])\s+|\n+")   # danda ends a sentence even with no space after it


def split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in parts if len(s) > 2]


def smart_chunk(text: str, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sentence-aware chunks with real sentence overlap."""
    text = re.sub(r"[ \t]+", " ", text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in split_sentences(text):
        if len(s) > max_chars:  # very long sentence: hard split
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            step = max_chars - overlap
            for i in range(0, len(s), step):
                chunks.append(s[i : i + max_chars].strip())
            continue

        if cur_len + len(s) + 1 > max_chars and cur:
            chunks.append("\n".join(cur))
            keep, k_len = [], 0  # carry trailing sentences over as overlap
            for prev in reversed(cur):
                if k_len + len(prev) > overlap:
                    break
                keep.insert(0, prev)
                k_len += len(prev) + 1
            cur, cur_len = keep, k_len

        cur.append(s)
        cur_len += len(s) + 1

    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if len(c.strip()) > 10]


# ─── Keyword matching ───────────────────────────────────────────
_TOKEN = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "what", "who", "how", "why",
    "when", "where", "which", "does", "do", "for", "on", "with", "this", "that",
    "को", "का", "की", "मा", "लाई", "ले", "हो", "छ", "के", "कसरी", "कहिले", "कहाँ",
    "र", "वा", "यो", "त्यो", "भएको", "गर्ने",
}


_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def tokens(text: str) -> set:
    """Lower-cased words; Nepali digits are mapped to ASCII so 2082/83 matches २०८२/८३."""
    return {t.lower().translate(_DIGITS) for t in _TOKEN.findall(text)
            if t.lower() not in _STOP and len(t) > 1}


def lexical_score(query_tokens: set, text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & tokens(text)) / len(query_tokens)


# ─── Question analysis helpers ──────────────────────────────────
GENERIC_WORDS = {
    # English filler that carries no topic
    "tell", "me", "about", "explain", "describe", "information", "info", "details", "detail",
    "more", "something", "anything", "everything", "this", "that", "it", "these", "those",
    "document", "documents", "file", "files", "give", "show", "please", "can", "you", "i",
    "want", "know", "say", "write", "there", "any", "some", "all", "main", "topic", "content",
    # Nepali filler
    "बारे", "बारेमा", "जानकारी", "विवरण", "बताउनु", "बताउनुहोस्", "भन्नुहोस्", "सबै", "केही",
    "यसको", "यसका", "यसमा", "कागजात", "फाइल", "मुख्य", "विषय", "देऊ", "दिनुहोस्", "बुझाउनुहोस्",
}
MODALITY_FILLER = {
    "image", "images", "figure", "figures", "fig", "chart", "charts", "diagram", "diagrams", "picture",
    "photo", "graph", "plot", "illustration", "table", "tables", "tabular", "equation", "equations",
    "formula", "formulas", "formulae", "चित्र", "तस्बिर", "ग्राफ", "नक्सा", "तालिका", "तालिकामा",
    "सूत्र", "समीकरण", "show", "shows", "shown", "see",
}
SUMMARY_WORDS = {"summarize", "summarise", "summary", "overview", "gist", "सारांश", "संक्षेप", "सार"}


def content_tokens(text: str) -> set:
    """Topic-bearing words only (no stop words, no 'tell me about this' filler)."""
    return {t for t in tokens(text)
            if t not in GENERIC_WORDS and t not in SUMMARY_WORDS and t not in MODALITY_FILLER}


def wants_summary(text: str) -> bool:
    return bool(tokens(text) & SUMMARY_WORDS)


def token_match(a: str, b: str) -> bool:
    """Equal, or one is a short-suffix variant of the other (नेपाल/नेपालको, fee/fees)."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 3:
        return True
    if len(short) >= 4 and len(long_) - len(short) <= 4:
        p = 0
        for x, y in zip(a, b):
            if x != y:
                break
            p += 1
        return p >= max(4, int(0.8 * len(short)))
    return False


def _is_subsequence(short: str, long_: str) -> bool:
    it = iter(long_)
    return all(ch in it for ch in short)


def _is_devanagari(tok: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in tok)


def damaged_match(q: str, t: str) -> bool:
    """For PDFs whose text layer lost conjuncts: the damaged word is the correct word with letters
    missing (लेखापरीणको <- लेखापरीक्षणको), so it is a subsequence of what the user types.
    The ि matra is ignored because broken fonts often move it in front of the consonant."""
    if not (_is_devanagari(q) and _is_devanagari(t)):
        return False
    q, t = q.replace("\u093f", ""), t.replace("\u093f", "")
    if q == t:
        return True
    if len(q) < 4 or len(t) < 3 or not _is_subsequence(t, q):
        return False
    if q[0] == t[0] and len(t) >= 0.35 * len(q):
        return True
    return len(t) >= 4 and len(t) >= 0.6 * len(q)      # first letter itself may have been lost


def keyword_coverage(query_terms: set, text: str, damaged: bool = False) -> float:
    """Share of the question's topic words that appear in the text (suffix-tolerant).

    damaged=True additionally tolerates missing letters (see damaged_match)."""
    if not query_terms:
        return 0.0
    text_tokens = tokens(text)
    hit = 0
    for q in query_terms:
        if q in text_tokens or any(token_match(q, t) or (damaged and damaged_match(q, t)) for t in text_tokens):
            hit += 1
    return hit / len(query_terms)


# ─── Light stemming (Nepali suffixes + English plurals) ─────────
_NE_SUFFIXES = ("हरूको", "हरूले", "हरूमा", "हरू", "बाट", "सँग", "प्रति", "लाई", "मा", "को", "का", "की", "ले", "मै")


def stem(tok: str) -> str:
    tok = tok.lower()
    if any("\u0900" <= c <= "\u097F" for c in tok):
        for suf in _NE_SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: -len(suf)]
        return tok
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


# ─── Which modality does the question ask for? ──────────────────
_MODALITY_WORDS = {
    "image": {"image", "images", "figure", "figures", "fig", "chart", "charts", "diagram", "diagrams",
              "picture", "photo", "graph", "plot", "illustration", "चित्र", "तस्बिर", "ग्राफ", "नक्सा"},
    "table": {"table", "tables", "tabular", "तालिका", "तालिकामा"},
    "equation": {"equation", "equations", "formula", "formulas", "formulae", "सूत्र", "समीकरण"},
}


def query_modalities(text: str) -> set:
    toks = {t.lower() for t in _TOKEN.findall(text)}
    return {m for m, words in _MODALITY_WORDS.items() if toks & words}


# ─── Formula questions ("what is the formula F = m a") ──────────
def extract_formula(question: str) -> str:
    """The formula-looking part of a question, or "" if it has none."""
    keep = [t for t in question.replace("?", " ").split() if not (t.isalpha() and len(t) >= 2)]
    f = " ".join(keep)
    return f if ("=" in f and len(re.sub(r"\s+", "", f)) >= 3) else ""


def norm_formula(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()
