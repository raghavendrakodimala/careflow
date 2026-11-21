"""BM25 sparse vectors for the keyword half of hybrid search (M4).

Dense embeddings match meaning but blur exact tokens: "GAST-COLO", "OP-CS-014", "MERIDIAN-BRONZE" and
"5 business days" are where policy questions are won or lost. BM25 matches those exactly.

Split of work with Qdrant:
- a document vector holds, per term, the BM25 term-frequency part tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl));
- the collection's sparse vector is configured with Modifier.IDF, so Qdrant applies the IDF part at query
  time from the collection itself (no IDF table to build, store or keep in sync);
- a query vector is 1.0 per distinct query term.

Terms are lower-cased words and numbers; a hyphenated code is kept whole and also split into its parts
("gast-colo", "gast", "colo"), so both "GAST-COLO" and "colonoscopy (GAST-COLO)" match. A short stop-word
list drops words that carry no meaning. Term IDs are stable hashes, so no vocabulary file is needed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or our so "
    "that the their them then there these this to under was we what when where which who will with within "
    "you your".split()
)


def terms(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.strip(".")
        if not tok or tok in STOP_WORDS:
            continue
        out.append(tok)
        if "-" in tok:
            out.extend(p for p in tok.split("-") if p and p not in STOP_WORDS)
    return out


def term_id(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big") & 0x7FFFFFFF


class Bm25Encoder:
    """`avgdl` is the corpus's average document length in terms; set by fit() when the index is built."""

    name = "bm25-v1"  # stored with the index: a change here forces a rebuild

    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float = 100.0):
        self.k1, self.b, self.avgdl = k1, b, avgdl

    def fit(self, texts: list[str]) -> Bm25Encoder:
        lengths = [len(terms(t)) for t in texts]
        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 100.0
        return self

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        counts = Counter(terms(text))
        length = sum(counts.values())
        norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
        weights: dict[int, float] = {}
        for term, tf in counts.items():
            tid = term_id(term)
            weights[tid] = weights.get(tid, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return list(weights), list(weights.values())

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        ids = sorted({term_id(t) for t in terms(text)})
        return ids, [1.0] * len(ids)
