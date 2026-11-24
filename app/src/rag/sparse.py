"""BM25 sparse vectors for the keyword half of hybrid search.

Dense embeddings match meaning but blur exact tokens: "GAST-COLO", "OP-CS-014", "MERIDIAN-BRONZE" and
"5 business days" are where policy questions are won or lost. BM25 matches those exactly.

Split of work with Qdrant:
- a document vector holds, per term, the BM25 term-frequency part tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl)),
  where tf is the term's frequency, k1 the saturation, b the length weight and avgdl the average document length;
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

# A lower-case word or number; inner hyphens and dots stay, so "gast-colo", "op-cs-014" and "2.5" are one token.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_DEFAULT_AVERAGE_LENGTH = 100.0  # average document length assumed until fit() has seen the corpus
STOP_WORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so", "that", "the", "their", "them", "then", "there", "these", "this", "to", "under", "was", "we", "what", "when", "where", "which", "who", "will", "with", "within", "you", "your"]
)


def extract_terms(text: str) -> list[str]:
    """The BM25 terms of `text`, with repeats, in order (see the module docstring for the rules)."""
    found: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        token = token.strip(".")
        if not token or token in STOP_WORDS:
            continue
        found.append(token)
        if "-" in token:
            found.extend(part for part in token.split("-") if part and part not in STOP_WORDS)
    return found


def compute_term_id(term: str) -> int:
    """A stable 31-bit hash of the term: its index in the sparse vector. Two terms can share an ID."""
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big") & 0x7FFFFFFF


class Bm25Encoder:
    """`average_length` is the corpus's average document length in terms; set by fit() when the index is built."""

    name = "bm25-v1"  # stored with the index: a change here forces a rebuild

    # The saturation (BM25's k1, on term frequency) and the length weight (BM25's b, on length normalisation)
    # are the usual BM25 defaults.
    def __init__(self, saturation: float = 1.2, length_weight: float = 0.75,
                 average_length: float = _DEFAULT_AVERAGE_LENGTH):
        self.saturation, self.length_weight, self.average_length = saturation, length_weight, average_length

    def fit(self, texts: list[str]) -> Bm25Encoder:
        """Set `average_length` from the corpus being indexed. Returns self."""
        lengths = [len(extract_terms(text)) for text in texts]
        self.average_length = (sum(lengths) / len(lengths) if lengths else 0.0) or _DEFAULT_AVERAGE_LENGTH
        return self

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """(term IDs, weights) of a document: the BM25 term-frequency part, without IDF."""
        counts = Counter(extract_terms(text))
        length = sum(counts.values())
        norm = self.saturation * (1 - self.length_weight + self.length_weight * length / self.average_length)
        weights: dict[int, float] = {}
        for term, frequency in counts.items():
            identifier = compute_term_id(term)
            # colliding terms add up
            weights[identifier] = weights.get(identifier, 0.0) + frequency * (self.saturation + 1) / (frequency + norm)
        return list(weights), list(weights.values())

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """(term IDs, weights) of a query: 1.0 per distinct term; both empty when only stop words are left."""
        ids = sorted({compute_term_id(term) for term in extract_terms(text)})
        return ids, [1.0] * len(ids)
