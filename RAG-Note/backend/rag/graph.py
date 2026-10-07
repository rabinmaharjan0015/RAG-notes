"""Lightweight multimodal knowledge graph (no LLM needed) used for local / global retrieval.

Nodes   : items (text / image / table / equation) and entities (key terms of the items)
Edges   : item --mentions--> entity          (weight = TF-IDF)
          entity <--co-occurs--> entity      (weight = shared items)
          item <--describes/next--> item     (a figure/table/equation <-> the text around it)
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from .config import ENTITIES_PER_ITEM
from .text_utils import content_tokens, stem, tokens, GENERIC_WORDS, SUMMARY_WORDS

Key = Tuple[str, int]  # (doc_id, item_index)

_CAP_PHRASE = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,})*")
_SKIP = GENERIC_WORDS | SUMMARY_WORDS


def _candidates(text: str) -> Counter:
    """Entity candidates: stemmed keywords, adjacent keyword pairs, and capitalised phrases."""
    toks = [t for t in re.findall(r"[\w\u0900-\u097F]+", text.lower())]
    keep = [stem(t) for t in toks if len(t) >= 4 and t not in _SKIP and not t.isdigit()]
    c: Counter = Counter(keep)
    for a, b in zip(keep, keep[1:]):
        c[f"{a} {b}"] += 1
    for m in _CAP_PHRASE.findall(text):
        if " " in m:
            c[" ".join(stem(w) for w in m.split())] += 2
    return c


class KnowledgeGraph:
    def __init__(self):
        self.clear()

    def clear(self):
        self.ent_items: Dict[str, Dict[Key, float]] = defaultdict(dict)
        self.item_ents: Dict[Key, Dict[str, float]] = {}
        self.co: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.struct: Dict[Key, Dict[Key, str]] = defaultdict(dict)
        self.n_items = 0

    # ─── build ──────────────────────────────────────────────────
    def rebuild(self, documents: Dict[str, dict]) -> None:
        self.clear()
        cands: Dict[Key, Counter] = {}
        df: Counter = Counter()
        for doc_id, doc in documents.items():
            for i, chunk in enumerate(doc["chunks"]):
                c = _candidates(chunk)
                cands[(doc_id, i)] = c
                df.update(c.keys())
        n = max(len(cands), 1)
        self.n_items = len(cands)
        too_common = 0.4 * n if n >= 10 else n + 1

        for key, c in cands.items():
            scored = []
            for ent, tf in c.items():
                if df[ent] > too_common:
                    continue
                scored.append((tf * math.log(1 + n / df[ent]) * (1.3 if " " in ent else 1.0), ent))
            scored.sort(reverse=True)
            ents = {e: w for w, e in scored[:ENTITIES_PER_ITEM]}
            self.item_ents[key] = ents
            for e, w in ents.items():
                self.ent_items[e][key] = w
            names = list(ents)
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    x, y = names[a], names[b]
                    self.co[x][y] = self.co[x].get(y, 0) + 1
                    self.co[y][x] = self.co[y].get(x, 0) + 1

        # structural edges: sequence + multimodal <-> surrounding text on the same page
        for doc_id, doc in documents.items():
            items = doc.get("items") or []
            for i, it in enumerate(items):
                if i + 1 < len(items):
                    self.struct[(doc_id, i)].setdefault((doc_id, i + 1), "next")
                    self.struct[(doc_id, i + 1)].setdefault((doc_id, i), "next")
                if it["type"] != "text":
                    for j in (i - 1, i + 1):
                        if 0 <= j < len(items) and items[j]["type"] == "text" and items[j]["page"] == it["page"]:
                            self.struct[(doc_id, i)][(doc_id, j)] = "describes"
                            self.struct[(doc_id, j)][(doc_id, i)] = "describes"

    # ─── query ──────────────────────────────────────────────────
    def match_entities(self, query: str) -> Set[str]:
        terms = [stem(t) for t in content_tokens(query)]
        matched = {t for t in terms if t in self.ent_items}
        toks = [stem(t) for t in re.findall(r"[\w\u0900-\u097F]+", query.lower())]
        for a, b in zip(toks, toks[1:]):          # adjacent pairs ("revenue growth")
            if f"{a} {b}" in self.ent_items:
                matched.add(f"{a} {b}")
        return matched

    def local_scores(self, query: str, doc_ids: Set[str]) -> Dict[Key, float]:
        """Entity-centric: items that mention the entities named in the question."""
        scores: Dict[Key, float] = defaultdict(float)
        for e in self.match_entities(query):
            for key, w in self.ent_items[e].items():
                if key[0] in doc_ids:
                    scores[key] += w
        return scores

    def global_scores(self, query: str, doc_ids: Set[str], per_entity: int = 8) -> Dict[Key, float]:
        """Relation-centric: items about entities that co-occur with the question's entities."""
        scores: Dict[Key, float] = defaultdict(float)
        matched = self.match_entities(query)
        for e in matched:
            neigh = sorted(self.co.get(e, {}).items(), key=lambda kv: kv[1], reverse=True)[:per_entity]
            for n, cw in neigh:
                if n in matched:
                    continue
                for key, w in self.ent_items[n].items():
                    if key[0] in doc_ids:
                        scores[key] += 0.5 * w * math.log(1 + cw)
        return scores

    def neighbors(self, key: Key, kinds=("describes",)) -> List[Key]:
        return [k for k, kind in self.struct.get(key, {}).items() if kind in kinds]

    # ─── inspection ─────────────────────────────────────────────
    def summary(self, limit: int = 40) -> dict:
        degree = {e: sum(w for w in nb.values()) for e, nb in self.co.items()}
        top = sorted(degree, key=degree.get, reverse=True)[:limit]
        topset = set(top)
        edges = [{"source": a, "target": b, "weight": w}
                 for a in top for b, w in self.co[a].items() if b in topset and a < b]
        return {
            "items": self.n_items,
            "entities": len(self.ent_items),
            "entity_links": sum(len(v) for v in self.co.values()) // 2,
            "structure_links": sum(len(v) for v in self.struct.values()) // 2,
            "top_entities": [{"name": e, "items": len(self.ent_items[e])} for e in top],
            "edges": edges,
        }


knowledge_graph = KnowledgeGraph()
