"""Modality-aware hybrid retrieval: vector search fused with knowledge-graph search.

Modes (as in RAG-Anything / LightRAG)
  naive   vector search only
  local   vector + entity-centric graph search (specific facts)
  global  vector + relation-centric graph search (themes, connected topics)
  hybrid  vector + local + global, fused with Reciprocal Rank Fusion  (default)

Confidence ("is this question answerable?") is ALWAYS judged on the vector evidence only,
so the graph can add context but can never make an unsupported question look answerable.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    DENSE_WEIGHT, LEXICAL_WEIGHT, CANDIDATES, SMALL_CORPUS_CHUNKS, RRF_K, MODALITY_BOOST,
)
from .graph import knowledge_graph
from .model_loader import embed_query, embed_passages
from .store import documents_store, embeddings_store
from .text_utils import content_tokens, keyword_coverage, split_sentences, query_modalities

MODES = ("naive", "local", "global", "hybrid")


@dataclass
class Hit:
    hybrid: float            # dense + keyword evidence (vector view)
    dense: float
    coverage: float          # share of question topic-words found in this item
    chunk: str               # searchable text of the item
    doc_id: str
    filename: str
    idx: int                 # item position inside its document
    item: dict = field(default_factory=dict)
    vec: np.ndarray = field(repr=False, default=None)
    via: str = "vector"      # vector | graph | structure
    fused: float = 0.0


@dataclass
class Retrieval:
    hits: List[Hit]          # final order (fused)
    vector_hits: List[Hit]   # vector order - used for answerability / ambiguity decisions
    n_chunks: int
    z: Optional[float]       # how far the best item stands out from ALL items (None if tiny corpus)
    terms: set               # topic words of the question
    mode: str = "hybrid"


def _make_hit(doc_id: str, idx: int, dense: float, terms: set, via: str) -> Hit:
    doc = documents_store[doc_id]
    chunk = doc["chunks"][idx]
    cov = keyword_coverage(terms, chunk)
    items = doc.get("items") or []
    return Hit(
        hybrid=DENSE_WEIGHT * dense + LEXICAL_WEIGHT * cov, dense=dense, coverage=cov, chunk=chunk,
        doc_id=doc_id, filename=doc["filename"], idx=idx,
        item=items[idx] if idx < len(items) else {"type": "text", "page": 0},
        vec=embeddings_store[doc_id][idx], via=via,
    )


def retrieve(query: str, doc_ids: List[str], mode: str = "hybrid", aux: str = "",
             terms: Optional[set] = None) -> Retrieval:
    mode = mode if mode in MODES else "hybrid"
    terms = terms if terms is not None else content_tokens(query)
    q_emb = embed_query((query + " " + aux).strip())

    owners, parts = [], []
    for doc_id in doc_ids:
        vecs = embeddings_store.get(doc_id)
        if vecs is None or doc_id not in documents_store:
            continue
        parts.append(vecs @ q_emb)
        owners.extend((doc_id, i) for i in range(len(vecs)))
    if not owners:
        return Retrieval([], [], 0, None, terms, mode)

    dense = np.concatenate(parts)
    n = len(dense)
    std = float(dense.std())
    z = float((dense.max() - dense.mean()) / std) if (n >= SMALL_CORPUS_CHUNKS and std > 1e-6) else None

    vector_hits: List[Hit] = []
    for k in np.argsort(-dense)[:CANDIDATES]:
        doc_id, i = owners[int(k)]
        vector_hits.append(_make_hit(doc_id, i, float(dense[k]), terms, "vector"))
    vector_hits.sort(key=lambda h: h.hybrid, reverse=True)

    # ── fusion ──
    mods = query_modalities(query)
    boost = lambda h: MODALITY_BOOST if h.item.get("type") in mods else 0.0
    vec_order = sorted(vector_hits, key=lambda h: h.hybrid + boost(h), reverse=True)
    by_key: Dict[Tuple[str, int], Hit] = {(h.doc_id, h.idx): h for h in vector_hits}
    fused: Dict[Tuple[str, int], float] = defaultdict(float)

    def add_ranking(order_keys, weight):
        for r, key in enumerate(order_keys):
            fused[key] += weight / (RRF_K + r + 1)

    dense_by_key = {owners[i]: float(dense[i]) for i in range(n)}

    def graph_ranking(scores: Dict[Tuple[str, int], float], weight: float):
        top = sorted(scores, key=scores.get, reverse=True)[:CANDIDATES // 2]
        for key in top:
            if key not in by_key:
                by_key[key] = _make_hit(key[0], key[1], dense_by_key.get(key, 0.0), terms, "graph")
        add_ranking(top, weight)

    graph_weight = {"naive": 0.0, "local": 1.0, "global": 1.0, "hybrid": 1.0}[mode]
    add_ranking([(h.doc_id, h.idx) for h in vec_order], 0.5 if mode in ("local", "global") else 1.0)
    if graph_weight:
        ids = set(doc_ids)
        if mode in ("local", "hybrid"):
            graph_ranking(knowledge_graph.local_scores(query, ids), graph_weight)
        if mode in ("global", "hybrid"):
            graph_ranking(knowledge_graph.global_scores(query, ids), graph_weight)

    ordered = sorted(fused, key=fused.get, reverse=True)
    hits = []
    for key in ordered:
        h = by_key[key]
        h.fused = fused[key]
        hits.append(h)

    # ── relational coherence: bring along the figure/table/equation that belongs to the text ──
    if mode != "naive":
        present = {(h.doc_id, h.idx) for h in hits}
        extra = []
        for h in hits[:3]:
            for nb in knowledge_graph.neighbors((h.doc_id, h.idx), kinds=("describes",)):
                if nb not in present and nb[0] in set(doc_ids):
                    nh = _make_hit(nb[0], nb[1], dense_by_key.get(nb, 0.0), terms, "structure")
                    nh.fused = h.fused * 0.5
                    extra.append(nh)
                    present.add(nb)
        hits.extend(extra)
        hits.sort(key=lambda h: h.fused, reverse=True)

    return Retrieval(hits[:CANDIDATES], vector_hits, n, z, terms, mode)


def extract_answer(query: str, hits: List[Hit], max_sentences: int = 3) -> List[str]:
    """Pick the best real sentences from the top TEXT items, in reading order."""
    text_hits = [h for h in hits if h.item.get("type", "text") == "text"][:3]
    cand = []
    for rank, h in enumerate(text_hits):
        for j, s in enumerate(split_sentences(h.chunk)):
            if len(s) >= 15:
                cand.append((rank, j, s))
    if not cand:
        return []

    q_emb = embed_query(query)
    terms = content_tokens(query)
    s_embs = embed_passages([c[2] for c in cand])

    scored = []
    for (rank, j, s), e in zip(cand, s_embs):
        score = DENSE_WEIGHT * float(e @ q_emb) + LEXICAL_WEIGHT * keyword_coverage(terms, s)
        scored.append((score, rank, j, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    best = scored[0][0]
    keep = [x for x in scored[:max_sentences] if x[0] >= best - 0.06]
    keep.sort(key=lambda x: (x[1], x[2]))

    seen, out = set(), []
    for _, _, _, s in keep:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
