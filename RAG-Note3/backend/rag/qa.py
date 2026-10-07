"""Question answering with an explicit decision step:

    generic question      -> ask what the user means (clarify)
    summary request       -> ask which document, or summarise it
    short, ambiguous term -> offer the different places it matches (clarify)
    no real evidence      -> "not found" (never guess)
    otherwise             -> answer from the documents: sentences, a table, a figure or an equation
"""

from typing import List, Optional

from fastapi import HTTPException

from .clarify import (
    generic_clarification, which_document, summarize_document,
    ambiguity_options, ambiguity_response, not_found_response,
)
from .config import (
    QA_MODEL, Z_STRONG, Z_MIN, Z_CROSS, MIN_KEYWORD_COVERAGE, MIN_PARTIAL_COVERAGE, MEDIA_DIR, logger,
)
from .model_loader import get_generator, vlm_generate, vlm_available
from .retrieval import retrieve, extract_answer, Retrieval, Hit, _make_hit
from .schemas import QuestionRequest, AnswerResponse, Source, VisualNote
from .store import documents_store
from .text_utils import content_tokens, wants_summary, extract_formula, norm_formula


def generate_grounded(question: str, passages: List[str]) -> Optional[str]:
    """Optional LLM answer limited to the passages. Returns None on any failure."""
    gen = get_generator()
    if gen is None:
        return None
    try:
        import torch

        tok, model = gen
        context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages[:4]))
        messages = [
            {"role": "system", "content":
                "Answer ONLY from the given passages, in the same language as the question. "
                "If the passages do not contain the answer, reply exactly: NOT_FOUND."},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return None if (not ans or "NOT_FOUND" in ans) else ans
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None


def is_supported(r: Retrieval) -> bool:
    """Do the documents contain real evidence for this question? (vector evidence only)

    A) the question's words appear in the best items AND the item stands out a little   (Z_MIN)
    B) at least some of its words appear AND the item stands out clearly                (Z_STRONG)
    C) no shared words (e.g. a question in another language) but it stands out hugely   (Z_CROSS)
    Tiny corpora (z-scores meaningless) rely on the question's words alone.
    """
    if not r.vector_hits:
        return False
    cov = max(h.coverage for h in r.vector_hits[:3])
    if r.z is None:
        return cov >= MIN_KEYWORD_COVERAGE
    if cov >= MIN_KEYWORD_COVERAGE and r.z >= Z_MIN:
        return True
    if cov >= MIN_PARTIAL_COVERAGE and r.z >= Z_STRONG:
        return True
    return r.z >= Z_CROSS


# ─── answer text for non-text items ──────────────────────────────
def _page_label(h: Hit) -> str:
    pages = documents_store.get(h.doc_id, {}).get("pages", 0)
    return f" (page {h.item.get('page', 0) + 1})" if pages > 1 else ""


def modal_answer(h: Hit) -> str:
    it, kind = h.item, h.item.get("type")
    if kind == "table":
        cap = f": {it['caption']}" if it.get("caption") else ""
        return f"**Table**{_page_label(h)}{cap}\n\n{it.get('raw', '')}"
    if kind == "equation":
        return f"**Equation**{_page_label(h)}\n\n```\n{it.get('raw', '')}\n```"
    if kind == "image":
        lines = [f"**Figure**{_page_label(h)}"]
        if it.get("caption"):
            lines.append(it["caption"])
        if it.get("description"):
            lines.append(it["description"])
        if len(lines) == 1:
            lines.append(h.chunk[:300])
        return "\n\n".join(lines)
    return h.chunk


def _source(h: Hit) -> Source:
    it = h.item
    pages = documents_store.get(h.doc_id, {}).get("pages", 0)
    return Source(
        filename=h.filename, snippet=h.chunk[:320], score=round(h.dense, 3),
        type=it.get("type", "text"), page=(it.get("page", 0) + 1) if pages > 1 else None,
        section=it.get("section"), media_url=f"/media/{it['media']}" if it.get("media") else None,
        raw=it.get("raw") or None, caption=it.get("caption") or None, via=h.via,
    )


def answer_question(req: QuestionRequest, aux_text: str = "") -> AnswerResponse:
    """Blocking. Run in a thread.  `aux_text` = description of an attached image/table/equation."""
    q = req.question.strip()
    ids = [i for i in (req.document_ids or list(documents_store.keys())) if i in documents_store]
    if not ids:
        raise HTTPException(404, "No matching documents.")

    # 0) Formula question: match the equation text exactly (most accurate possible)
    formula = extract_formula(q)
    if formula and not aux_text:
        n = norm_formula(formula)
        for did in ids:
            for i, it in enumerate(documents_store[did].get("items") or []):
                if it.get("type") == "equation" and n in norm_formula(it.get("raw", "")):
                    h = _make_hit(did, i, 1.0, set(), "vector")
                    return AnswerResponse(
                        answer=modal_answer(h), sources=[h.filename], source_count=1, confidence=1.0,
                        found=True, status="answer", evidence=[_source(h)], mode=req.mode,
                    )
        return not_found_response(q, ids)

    terms = content_tokens(q) or (content_tokens(aux_text) if aux_text else set())

    # 1) Summary request
    if wants_summary(q) and not content_tokens(q) and not aux_text:
        from .clarify import summarize_document as _sum
        return _sum(q, ids[0]) if len(ids) == 1 else which_document(q, ids)

    # 2) Generic question: nothing specific to search for -> ask
    if not terms:
        return generic_clarification(q, ids)

    # 3) Retrieve and judge the evidence
    r = retrieve(q, ids, mode=req.mode, aux=aux_text, terms=terms)
    diag = None
    if req.debug:
        top = r.vector_hits[0] if r.vector_hits else None
        diag = {
            "mode": r.mode, "topic_words": sorted(terms), "items_searched": r.n_chunks,
            "z_score": None if r.z is None else round(r.z, 2),
            "best_dense": None if top is None else round(top.dense, 3),
            "best_keyword_coverage": None if top is None else round(top.coverage, 2),
            "supported": is_supported(r),
            "top_hits": [{"type": h.item.get("type"), "via": h.via, "fused": round(h.fused, 4),
                          "text": h.chunk[:60]} for h in r.hits[:5]],
            "thresholds": {"Z_MIN": Z_MIN, "Z_STRONG": Z_STRONG, "Z_CROSS": Z_CROSS,
                           "MIN_KEYWORD_COVERAGE": MIN_KEYWORD_COVERAGE, "MIN_PARTIAL_COVERAGE": MIN_PARTIAL_COVERAGE},
        }

    if not is_supported(r):
        return not_found_response(q, ids, diag)

    # 4) Supported but too short/ambiguous: ask which meaning
    if not aux_text:
        options = ambiguity_options(q, r)
        if options:
            resp = ambiguity_response(q, options)
            resp.diagnostics = diag
            return resp

    # 5) Answer from the documents
    hits = r.hits
    top = hits[0]
    if top.item.get("type", "text") != "text":
        answer = modal_answer(top)                       # show the actual table / figure / equation
    else:
        answer = generate_grounded(q, [h.chunk for h in hits if h.item.get("type") == "text"]) if QA_MODEL else None
        if not answer:
            sentences = extract_answer(q, hits)
            answer = "\n\n".join(sentences) if sentences else top.chunk

    evidence, seen = [], set()
    for h in hits[: req.max_sources * 2]:
        modal = h.item.get("type", "text") != "text"
        relevant = (h is top
                    or (not modal and h.coverage > 0)
                    or (modal and (h.coverage >= MIN_KEYWORD_COVERAGE or h.via == "structure")))
        if not relevant:
            continue
        key = (h.filename, h.chunk[:60])
        if key in seen:
            continue
        seen.add(key)
        evidence.append(_source(h))
        if len(evidence) >= req.max_sources:
            break

    kinds = {e.type for e in evidence if e.type != "text"} - {top.item.get("type")}
    if kinds and top.item.get("type", "text") == "text":
        answer += "\n\n_Related " + ", ".join(sorted(kinds)) + " from the document is shown in the sources below._"

    # Optional: let the vision model look at the retrieved images
    notes: List[VisualNote] = []
    if req.vlm_enhanced and vlm_available():
        for h in [h for h in hits if h.item.get("media")][:2]:
            path = MEDIA_DIR / h.item["media"]
            out = vlm_generate(
                path,
                f"Answer the question using only what is visible in this image. "
                f"If the image does not show the answer, reply exactly: NOT_VISIBLE. Question: {q}",
                max_new_tokens=100,
            )
            if out and "NOT_VISIBLE" not in out.upper():
                notes.append(VisualNote(media_url=f"/media/{h.item['media']}", note=out))

    sources = list(dict.fromkeys(e.filename for e in evidence))
    conf = max(h.coverage for h in r.vector_hits[:3]) if r.z is None else min(1.0, max(0.0, r.z / (Z_STRONG + 1.0)))
    return AnswerResponse(
        answer=answer, sources=sources, source_count=len(sources), confidence=round(conf, 2),
        found=True, status="answer", evidence=evidence, visual_notes=notes, mode=r.mode, diagnostics=diag,
    )
