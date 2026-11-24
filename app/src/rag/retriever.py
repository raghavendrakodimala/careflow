"""The policy retriever that search_policies calls: scope, hybrid search, rerank, diversify, abstain.

    question -> specialty scope (given, or inferred when the question names exactly one specialty)
             -> dense + BM25 candidates (Qdrant, RRF fusion)
             -> cross-encoder rerank
             -> diversify: the top few passages cover different documents (see _diversify)
             -> best score below the threshold? abstain (no passages) : top `limit` passages

Abstaining is the point of the threshold: a question the clinic's documents do not answer (valet
parking, uninsured tourists) gets no passages, so the agent says it does not have that information
instead of building an answer out of the nearest, unrelated policy. The threshold is a setting,
calibrated by scripts/run_retrieval_eval.py on answerable vs unanswerable questions.

Without a reranker the retriever returns the fused ranking and never abstains (fusion scores are
rank-based, so no fixed threshold means anything on them).

Stateless per call: retrieve() returns the hits and whether it abstained, so one retriever can serve
concurrent requests (several agents, the API's worker threads).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import Settings, get_settings
from .embeddings import build_embedder
from .index import PolicyIndex
from .rerank import Reranker, build_reranker

CANDIDATES = 20  # passages fetched from the index and handed to the reranker (the settings can override it)
MAX_PER_DOCUMENT = 2  # passages one document may hold among the diversified top results (see _diversify)
RRF_K = 60  # the usual Reciprocal Rank Fusion constant (also Qdrant's)
# A passage whose rerank score is within 1% of the best is a tie with it: spreading the results over
# documents must not push it out. Golden case EV-019: three sections of the cost-share schedule scored
# 0.998 to 0.9995 for "specialist visit co-pay"; one-per-document kept "Telehealth" and replaced the
# co-pay table itself with an intake-SOP passage scoring 0.001, so every rephrased search missed it.
NEAR_TIE = 0.99

# Words that tie a question to one service line: its name, its code prefix and its procedures
# (procedures.csv). Used only when the caller gave no specialty.
_SPECIALTY_TERMS = {
    "Cardiology": r"cardiolog\w*|cardiac|heart|echocardiogram|echo|catheteri[sz]ation|card-\w+",
    "Orthopaedics": r"orthop(?:a)?edic\w*|knee|arthroscopy|orth-\w+",
    "Dermatology": r"dermatolog\w*|skin|lesion|derm-\w+",
    "Endocrinology": r"endocrinolog\w*|thyroid|endo-\w+",
    "Gastroenterology": r"gastroenterolog\w*|gastro|colonoscopy|abdominal|gast-\w+",
    "Pulmonology": r"pulmonolog\w*|pulmonary|lungs?|bronchoscopy|pulm-\w+",
    "Neurology": r"neurolog\w*|nerve|brain|neur-\w+",
    "ENT": r"ear|nose|throat|audiometry|septoplasty|ent-\w+",
}
# Each specialty's terms as one whole-word, case-insensitive pattern.
_SPECIALTY_PATTERNS = {specialty: re.compile(rf"\b(?:{terms})\b", re.I) for specialty, terms in _SPECIALTY_TERMS.items()}
_ENT_WORD = re.compile(r"\bENT\b")  # upper case only: "ent" is not a word to match loosely


def infer_specialty(query: str) -> str | None:
    """The one specialty a question is about, or None when it names none or several."""
    found = {specialty for specialty, pattern in _SPECIALTY_PATTERNS.items() if pattern.search(query)}
    if _ENT_WORD.search(query):
        found.add("ENT")
    return found.pop() if len(found) == 1 else None


@dataclass
class Retrieval:
    """The outcome of one retrieve() call."""

    hits: list = field(default_factory=list)  # index.Hit objects, best first; empty when abstained
    abstained: bool = False  # the best score was under the threshold: the corpus does not answer the question
    top_score: float | None = None  # best reranked score (None without a reranker)
    specialty: str | None = None  # the scope used (given or inferred)


class PolicyRetriever:
    """Search, rerank, diversify and abstain over one PolicyIndex.

    `min_score`: abstain threshold on the reranker's probability (None = never abstain); `candidates`: how
    many passages are fetched for reranking; `mode`: the index search mode; `max_per_document`: see
    _diversify (None or 0 switches diversifying off); `fuse`: order by RRF of the retrieval and rerank
    ranks instead of by rerank score alone; `infer`: infer the specialty scope from the question when the
    caller gives none."""

    def __init__(
        self,
        index: PolicyIndex,
        reranker: Reranker | None = None,
        min_score: float | None = None,
        candidates: int = CANDIDATES,
        mode: str = "hybrid",
        max_per_document: int | None = MAX_PER_DOCUMENT,
        fuse: bool = False,
        infer: bool = True,
    ):
        self.index = index
        self.reranker = reranker
        self.min_score = min_score
        self.candidates = candidates
        self.mode = mode
        self.max_per_document = max_per_document
        self.fuse = fuse
        self.infer = infer

    def resolve_scope(self, query: str, specialty: str | None) -> str | None:
        """The specialty to search within: the one given, else the one inferred, else None (whole corpus)."""
        return specialty or (infer_specialty(query) if self.infer else None)

    def rank(self, query: str, specialty: str | None = None, limit: int | None = None) -> list:
        """Every candidate, best first, with rerank scores when a reranker is set (no threshold applied)."""
        scope = self.resolve_scope(query, specialty)
        hits = self.index.search(query, limit=max(self.candidates, limit or 0), specialty=scope, mode=self.mode)
        if self.reranker is not None and hits:
            hits = self._rerank(query, hits)
        hits = self._diversify(hits, scoped=scope is not None)
        return hits[:limit] if limit else hits

    def _rerank(self, query: str, hits: list) -> list:
        """`hits` (in retrieval order) with their scores replaced by the reranker's, reordered best first."""
        scores = self.reranker.score_passages(query, [hit.text for hit in hits])
        rescored = [replace(hit, score=score) for hit, score in zip(hits, scores, strict=True)]
        if not self.fuse:
            return sorted(rescored, key=lambda hit: hit.score, reverse=True)
        # Order by Reciprocal Rank Fusion of the retrieval rank and the rerank rank; the score shown
        # (and used for the abstain threshold) stays the cross-encoder probability.
        by_rerank = sorted(range(len(rescored)), key=lambda index: rescored[index].score, reverse=True)
        rerank_rank = {index: rank for rank, index in enumerate(by_rerank)}
        order = sorted(range(len(rescored)),
                       key=lambda index: -(1 / (RRF_K + index) + 1 / (RRF_K + rerank_rank[index])))
        return [rescored[index] for index in order]

    def _diversify(self, hits: list, scoped: bool) -> list:
        """Spread the top passages over documents; the best passage always stays first.

        Pass 1: one passage per document and, for an unscoped question, one specialty handbook in all.
        The eight handbooks are near-duplicates (each has a "plan coverage and pre-authorisation"
        section), so without the family limit a coverage question fills the top four with handbooks and
        pushes out the cost-share schedule and pre-authorisation matrix it actually needs.
        Pass 2: up to `max_per_document` per document. Then everything else, in order. Nothing is dropped.
        Before both passes, with a reranker: every passage that ties with the best (NEAR_TIE) keeps its
        place, whatever its document. Without a reranker the scores are rank-based, so no tie is defined."""
        if not self.max_per_document or not hits:
            return hits
        order: list[int] = []
        if self.reranker is not None:
            best = max(hit.score for hit in hits)
            if best > 0:
                order = [index for index, hit in enumerate(hits) if hit.score >= NEAR_TIE * best]
        for per_document, one_handbook in ((1, not scoped), (self.max_per_document, False)):
            for index, hit in enumerate(hits):
                if index in order:
                    continue
                if sum(hits[kept].doc_title == hit.doc_title for kept in order) >= per_document:
                    continue
                if one_handbook and hit.specialty and any(hits[kept].specialty for kept in order):
                    continue
                order.append(index)
        return [hits[index] for index in order] + [hit for index, hit in enumerate(hits) if index not in order]

    def retrieve(self, query: str, limit: int = 4, specialty: str | None = None) -> Retrieval:
        """The top `limit` passages, or none when abstaining. With a reranker and a threshold, no candidates
        at all also counts as abstaining."""
        scope = self.resolve_scope(query, specialty)
        ranked = self.rank(query, scope)
        top = max(hit.score for hit in ranked) if ranked and self.reranker is not None else None
        abstained = self.reranker is not None and self.min_score is not None and (top is None or top < self.min_score)
        return Retrieval(hits=[] if abstained else ranked[:limit], abstained=abstained, top_score=top, specialty=scope)

    def search(self, query: str, limit: int = 4, specialty: str | None = None) -> list:
        """retrieve() for callers that only want the passages."""
        return self.retrieve(query, limit, specialty).hits

    def count(self) -> int:
        return self.index.count()

    def close(self) -> None:
        self.index.close()


def open_policy_search(settings: Settings | None = None, path: Path | str | None = None) -> PolicyRetriever:
    """The configured retriever: Qdrant location, embedding model, reranker and threshold come from settings."""
    settings = settings or get_settings()
    index = PolicyIndex.open(build_embedder(settings), path, url=settings.careflow_qdrant_url or None)
    reranker = build_reranker(settings.careflow_reranker_model)
    return PolicyRetriever(
        index,
        reranker=reranker,
        min_score=settings.careflow_rerank_min_score if reranker else None,
        candidates=settings.careflow_retrieval_candidates,
    )
