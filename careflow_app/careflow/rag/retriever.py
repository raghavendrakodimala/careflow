"""The policy retriever that search_policies calls (M4): scope, hybrid search, rerank, diversify, abstain.

    question -> specialty scope (given, or inferred when the question names exactly one specialty)
             -> dense + BM25 candidates (Qdrant, RRF fusion)
             -> cross-encoder rerank
             -> diversify: the top few passages cover different documents (see _diversify)
             -> best score below the threshold? abstain (no passages) : top `limit` passages

Abstaining is the point of the threshold: a question the clinic's documents do not answer (valet
parking, uninsured tourists) gets no passages, so the agent says it does not have that information
instead of building an answer out of the nearest, unrelated policy. The threshold is a setting,
calibrated by scripts/run_m4_eval.py on answerable vs unanswerable questions.

Without a reranker the retriever returns the fused ranking and never abstains (fusion scores are
rank-based, so no fixed threshold means anything on them).

Stateless per call: retrieve() returns the hits and whether it abstained, so one retriever can serve
concurrent requests (M6 agents, M8 API workers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import Settings, get_settings
from .embeddings import default_embedder
from .index import PolicyIndex
from .rerank import Reranker, default_reranker

CANDIDATES = 20
MAX_PER_DOCUMENT = 2
RRF_K = 60  # the usual Reciprocal Rank Fusion constant (also Qdrant's)

# Words that tie a question to one service line: its name, its code prefix and its procedures
# (procedures.csv). Used only when the caller gave no specialty (feedback M4-3).
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
_SPECIALTY_RE = {s: re.compile(rf"\b(?:{p})\b", re.I) for s, p in _SPECIALTY_TERMS.items()}
_ENT_WORD = re.compile(r"\bENT\b")  # upper case only: "ent" is not a word to match loosely


def infer_specialty(query: str) -> str | None:
    """The one specialty a question is about, or None when it names none or several."""
    found = {s for s, rx in _SPECIALTY_RE.items() if rx.search(query)}
    if _ENT_WORD.search(query):
        found.add("ENT")
    return found.pop() if len(found) == 1 else None


@dataclass
class Retrieval:
    hits: list = field(default_factory=list)
    abstained: bool = False
    top_score: float | None = None  # best reranked score (None without a reranker)
    specialty: str | None = None  # the scope used (given or inferred)


class PolicyRetriever:
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

    def scope(self, query: str, specialty: str | None) -> str | None:
        return specialty or (infer_specialty(query) if self.infer else None)

    def rank(self, query: str, specialty: str | None = None, limit: int | None = None) -> list:
        """Every candidate, best first, with rerank scores when a reranker is set (no threshold applied)."""
        scope = self.scope(query, specialty)
        hits = self.index.search(query, limit=max(self.candidates, limit or 0), specialty=scope, mode=self.mode)
        if self.reranker is not None and hits:
            scores = self.reranker.score(query, [h.text for h in hits])
            rescored = [replace(h, score=s) for h, s in zip(hits, scores)]  # hits are in retrieval order
            if self.fuse:
                # Order by Reciprocal Rank Fusion of the retrieval rank and the rerank rank; the score shown
                # (and used for the abstain threshold) stays the cross-encoder probability.
                by_rerank = sorted(range(len(rescored)), key=lambda i: rescored[i].score, reverse=True)
                rerank_rank = {i: r for r, i in enumerate(by_rerank)}
                order = sorted(range(len(rescored)), key=lambda i: -(1 / (RRF_K + i) + 1 / (RRF_K + rerank_rank[i])))
                hits = [rescored[i] for i in order]
            else:
                hits = sorted(rescored, key=lambda h: h.score, reverse=True)
        hits = self._diversify(hits, scoped=scope is not None)
        return hits[:limit] if limit else hits

    def _diversify(self, hits: list, scoped: bool) -> list:
        """Spread the top passages over documents; the best passage always stays first.

        Pass 1: one passage per document and, for an unscoped question, one specialty handbook in all.
        The eight handbooks are near-duplicates (each has a "plan coverage and pre-authorisation"
        section), so without the family limit a coverage question fills the top four with handbooks and
        pushes out the cost-share schedule and pre-authorisation matrix it actually needs.
        Pass 2: up to `max_per_document` per document. Then everything else, in order. Nothing is dropped."""
        if not self.max_per_document or not hits:
            return hits
        order: list[int] = []
        for per_doc, one_handbook in ((1, not scoped), (self.max_per_document, False)):
            for i, h in enumerate(hits):
                if i in order:
                    continue
                if sum(hits[j].doc_title == h.doc_title for j in order) >= per_doc:
                    continue
                if one_handbook and h.specialty and any(hits[j].specialty for j in order):
                    continue
                order.append(i)
        return [hits[i] for i in order] + [h for i, h in enumerate(hits) if i not in order]

    def retrieve(self, query: str, limit: int = 4, specialty: str | None = None) -> Retrieval:
        scope = self.scope(query, specialty)
        ranked = self.rank(query, scope)
        top = ranked[0].score if ranked and self.reranker is not None else None
        abstained = self.reranker is not None and self.min_score is not None and (top is None or top < self.min_score)
        return Retrieval(hits=[] if abstained else ranked[:limit], abstained=abstained, top_score=top, specialty=scope)

    def search(self, query: str, limit: int = 4, specialty: str | None = None) -> list:
        return self.retrieve(query, limit, specialty).hits

    def count(self) -> int:
        return self.index.count()

    def close(self) -> None:
        self.index.close()


def open_policy_search(settings: Settings | None = None, path: Path | str | None = None) -> PolicyRetriever:
    """The configured retriever: Qdrant location, embedding model, reranker and threshold come from settings."""
    settings = settings or get_settings()
    index = PolicyIndex.open(default_embedder(settings), path, url=settings.careflow_qdrant_url or None)
    reranker = default_reranker(settings.careflow_reranker_model)
    return PolicyRetriever(
        index,
        reranker=reranker,
        min_score=settings.careflow_rerank_min_score if reranker else None,
        candidates=settings.careflow_retrieval_candidates,
    )
