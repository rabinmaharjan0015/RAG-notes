"""Clarifying questions: generic questions, ambiguous questions, summary requests."""

from collections import Counter
from typing import List, Optional

import numpy as np

from .config import AMBIG_RATIO, AMBIG_MAX_TERMS, MAX_OPTIONS
from .retrieval import Retrieval, Hit
from .schemas import AnswerResponse, Option
from .store import documents_store
from .text_utils import (
    GENERIC_WORDS, SUMMARY_WORDS, split_sentences, tokens, is_nepali_text,
)

T = {
    "generic": {
        "en": "Your question is a bit broad, so I can't give a precise answer. What exactly would you like to know? You can pick a topic below or type a more specific question.",
        "ne": "तपाईंको प्रश्न धेरै सामान्य छ, त्यसैले सही जवाफ दिन सकिँदैन। तपाईं ठ्याक्कै के जान्न चाहनुहुन्छ? तल विषय छान्नुहोस् वा थप स्पष्ट प्रश्न लेख्नुहोस्।",
    },
    "ambiguous": {
        "en": "Your question matches several different parts of your documents. Which one do you mean?",
        "ne": "तपाईंको प्रश्न कागजातका धेरै फरक भागसँग मिल्छ। तपाईंले कुन बुझ्नुभएको हो?",
    },
    "which_doc": {
        "en": "Which document would you like me to summarize?",
        "ne": "कुन कागजातको सारांश चाहिन्छ?",
    },
    "not_found": {
        "en": "I couldn't find an answer to that in your documents. Try rephrasing, or pick a topic they cover:",
        "ne": "तपाईंका कागजातहरूमा यस प्रश्नको जवाफ फेला परेन। फरक तरिकाले सोध्नुहोस्, वा कागजातमा भएको विषय छान्नुहोस्:",
    },
    "summary_title": {"en": "Main points of", "ne": "मुख्य बुँदाहरू:"},
}


def _lang(question: str) -> str:
    return "ne" if is_nepali_text(question) else "en"


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _text_indices(doc: dict) -> List[int]:
    """Indices of plain-text items (tables/figures/equations make poor topic labels)."""
    items = doc.get("items") or []
    idx = [i for i in range(len(doc["chunks"])) if i >= len(items) or items[i]["type"] == "text"]
    return idx or list(range(len(doc["chunks"])))


def _chunk_df(doc_ids: List[str]) -> Counter:
    df: Counter = Counter()
    for did in doc_ids:
        for chunk in documents_store[did]["chunks"]:
            df.update(tokens(chunk))
    return df


def _distinct(chunk: str, df: Counter, avoid: set = frozenset(), k: int = 3) -> List[str]:
    """Words that make this chunk different from the rest (low document frequency)."""
    skip = GENERIC_WORDS | SUMMARY_WORDS | set(avoid)
    words = [t for t in tokens(chunk) if len(t) >= 4 and t not in skip and not t.isdigit()]
    words.sort(key=lambda t: (df[t], -len(t)))
    return words[:k]


def topic_options(doc_ids: List[str], n: int = MAX_OPTIONS) -> List[Option]:
    """Real topics from the documents: one per spread-out section, round-robin over documents."""
    df = _chunk_df(doc_ids)
    multi = len(doc_ids) > 1
    per_doc = {}
    for did in doc_ids:
        pool = _text_indices(documents_store[did])
        count = min(n, len(pool))
        per_doc[did] = sorted({pool[int(i)] for i in np.linspace(0, len(pool) - 1, num=count)})
    opts: List[Option] = []
    seen_queries = set()
    depth = 0
    while len(opts) < n and any(depth < len(v) for v in per_doc.values()):
        for did in doc_ids:
            if depth >= len(per_doc[did]) or len(opts) >= n:
                continue
            doc = documents_store[did]
            chunk = doc["chunks"][per_doc[did][depth]]
            words = _distinct(chunk, df)
            q = " ".join(words)
            if not words or q in seen_queries:
                continue
            seen_queries.add(q)
            sents = split_sentences(chunk)
            first = sents[0] if sents else chunk
            label = _short(f"{doc['filename']}: {first}" if multi else first)
            opts.append(Option(label=label, query=q, document_ids=[did]))
        depth += 1
    return opts


# ─── 1. Generic question ("tell me about this", "यो के हो?") ───────
def generic_clarification(question: str, doc_ids: List[str]) -> AnswerResponse:
    opts = []
    if len(doc_ids) <= 3:
        for did in doc_ids:
            opts.append(Option(label=f"Summary of {documents_store[did]['filename']}",
                               query="summarize", document_ids=[did]))
    opts += topic_options(doc_ids)
    return AnswerResponse(
        answer=T["generic"][_lang(question)], sources=[], source_count=0, confidence=0.0,
        found=False, status="clarify", options=opts[: MAX_OPTIONS + 3],
    )


# ─── 2. Summary request ───────────────────────────────────────────
def which_document(question: str, doc_ids: List[str]) -> AnswerResponse:
    opts = [Option(label=documents_store[d]["filename"], query="summarize", document_ids=[d])
            for d in doc_ids[:8]]
    return AnswerResponse(
        answer=T["which_doc"][_lang(question)], sources=[], source_count=0, confidence=0.0,
        found=False, status="clarify", options=opts,
    )


def summarize_document(question: str, doc_id: str) -> AnswerResponse:
    """Extractive overview: the opening sentence of evenly spaced chunks."""
    doc = documents_store[doc_id]
    chunks = doc["chunks"]
    pool = _text_indices(doc)
    picks = sorted({pool[int(i)] for i in np.linspace(0, len(pool) - 1, num=min(6, len(pool)))})
    lines = []
    for i in picks:
        sents = [s for s in split_sentences(chunks[i]) if len(s) >= 20]
        if sents:
            lines.append("• " + _short(sents[0], 220))
    body = "\n".join(lines) if lines else _short(chunks[0], 400)
    title = f"{T['summary_title'][_lang(question)]} {doc['filename']}"
    return AnswerResponse(
        answer=f"**{title}**\n\n{body}", sources=[doc["filename"]], source_count=1,
        confidence=1.0, found=True, status="answer",
    )


# ─── 3. Ambiguous question (short term matching several unrelated places) ─
def ambiguity_options(question: str, r: Retrieval) -> Optional[List[Option]]:
    """Return choices if several *different* places answer a very short question equally well."""
    hits = r.vector_hits
    if not hits or len(r.terms) > AMBIG_MAX_TERMS or not r.terms:
        return None
    best = hits[0]
    if best.coverage <= 0:
        return None

    chosen: List[Hit] = []
    for h in hits:
        if h.hybrid < best.hybrid * AMBIG_RATIO or h.coverage < best.coverage or h.coverage <= 0:
            continue
        def place(x):          # a table and its paragraph share a place; a different section/page does not
            return (x.doc_id, x.item.get("section") or x.item.get("page", 0))

        same_place = any(
            place(h) == place(c) or (h.doc_id == c.doc_id and abs(h.idx - c.idx) <= 1) for c in chosen
        )
        near_duplicate = any(float(h.vec @ c.vec) > 0.95 for c in chosen)  # same content repeated
        if not same_place and not near_duplicate:
            chosen.append(h)
        if len(chosen) >= MAX_OPTIONS:
            break
    if len(chosen) < 2:
        return None

    multi_doc = len({c.doc_id for c in chosen}) > 1
    df = _chunk_df(list({h.doc_id for h in hits}))
    opts = []
    for c in chosen:
        first = split_sentences(c.chunk)[0] if split_sentences(c.chunk) else c.chunk
        label = _short(f"{c.filename}: {first}" if multi_doc else first)
        extra = _distinct(c.chunk, df, avoid=r.terms)
        opts.append(Option(label=label, query=" ".join([question.strip().rstrip("?"), *extra]),
                           document_ids=[c.doc_id]))
    return opts


def ambiguity_response(question: str, options: List[Option]) -> AnswerResponse:
    return AnswerResponse(
        answer=T["ambiguous"][_lang(question)], sources=[], source_count=0, confidence=0.0,
        found=False, status="clarify", options=options,
    )


# ─── 4. Nothing relevant ───────────────────────────────────────────
def not_found_response(question: str, doc_ids: List[str], diagnostics=None) -> AnswerResponse:
    return AnswerResponse(
        answer=T["not_found"][_lang(question)], sources=[], source_count=0, confidence=0.0,
        found=False, status="not_found", options=topic_options(doc_ids)[:MAX_OPTIONS],
        diagnostics=diagnostics,
    )
