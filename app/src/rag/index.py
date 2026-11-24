"""The Qdrant policy index: one collection, `careflow_policies`, of corpus chunks.

- Local mode: QdrantClient(path=out/qdrant) persists to disk with no server; ":memory:" for tests.
  A deployment can point the same code at a Qdrant server instead.
- Each point has two vectors: "dense" (the embedding model) and "bm25" (sparse keyword weights,
  rag/sparse.py; Qdrant applies IDF at query time).
- build() is idempotent. Chunk IDs are stable (uuid5 of source + slug + ordinal), so a rebuild
  overwrites in place and deletes chunks that no longer exist. If the embedder or the sparse encoding
  changed, the collection is recreated, because vectors from two encoders cannot be compared.
- search() modes: "dense", "sparse" or "hybrid" (both, fused with Reciprocal Rank Fusion inside Qdrant).
  A search can be scoped to one specialty, so a question about Cardiology never returns the
  near-duplicate Neurology handbook (the clinic-wide documents stay searchable).
  Reranking and the abstain threshold sit on top, in retriever.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from ..config import QDRANT_PATH
from .chunker import SPECIALTY_BY_SLUG, Chunk
from .embeddings import Embedder
from .sparse import Bm25Encoder

COLLECTION = "careflow_policies"
DENSE, SPARSE = "dense", "bm25"  # names of a point's two vectors
MODES = ("dense", "sparse", "hybrid")
PREFETCH = 20  # candidates each half contributes before fusion
_SCROLL_PAGE = 256  # point IDs read per request when listing the collection


class IndexOutOfDateError(RuntimeError):
    """The collection on disk was built by an older layout (one unnamed vector) or another embedder."""


@dataclass(frozen=True)
class Hit:
    """One search result. `score` is on the scale of whatever ranked it: cosine similarity (dense), BM25
    (sparse), an RRF score (hybrid) or, once the retriever has reranked, a probability in [0, 1]."""

    score: float
    doc_title: str
    doc_id: str | None
    heading: str
    specialty: str | None
    text: str
    chunk_id: str

    def to_dict(self) -> dict:
        """The passage as search_policies returns it to the agent (no chunk ID or specialty)."""
        return {
            "document": self.doc_title,
            "document_id": self.doc_id,
            "section": self.heading,
            "score": round(self.score, 3),
            "text": self.text,
        }


class PolicyIndex:
    """Builds and searches the collection. The embedder and sparse encoder given here must be the ones the
    stored collection was built with; build() and the first search() check that."""

    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = COLLECTION,
                 sparse: Bm25Encoder | None = None):
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.sparse = sparse or Bm25Encoder()
        self._schema_ok: bool | None = None  # checked once before the first search, set by build()

    @classmethod
    def open(cls, embedder: Embedder, path: Path | str | None = None, collection: str = COLLECTION,
             url: str | None = None) -> PolicyIndex:
        """`url` (CAREFLOW_QDRANT_URL): a Qdrant server, shared by several processes. Otherwise local
        mode at `path` (default out/qdrant; one process at a time) or `":memory:"` for a throwaway index."""
        if url:
            return cls(QdrantClient(url=url), embedder, collection)
        location = str(path or QDRANT_PATH)
        client = QdrantClient(":memory:") if location == ":memory:" else QdrantClient(path=location)
        return cls(client, embedder, collection)

    # ---- build -------------------------------------------------------------------------------
    def _is_compatible(self) -> bool:
        """Does the stored collection have this index's layout, embedder and sparse encoding? An empty
        collection with the right layout counts as compatible; a missing one does not."""
        if not self.client.collection_exists(self.collection):
            return False
        params = self.client.get_collection(self.collection).config.params
        dense = params.vectors.get(DENSE) if isinstance(params.vectors, dict) else None
        if dense is None or dense.size != self.embedder.dimensions or SPARSE not in (params.sparse_vectors or {}):
            return False  # an older collection (one unnamed vector) or another vector size
        sample, _ = self.client.scroll(self.collection, limit=1, with_payload=["embedder", "sparse"])
        return not sample or (
            sample[0].payload.get("embedder") == self.embedder.name and sample[0].payload.get("sparse") == self.sparse.name
        )

    def build(self, chunks: list[Chunk], batch: int = 64) -> dict:
        """Make the collection hold exactly `chunks`, embedding `batch` chunks per request. Returns the
        build report: chunks given, points stored, stale points removed, and whether it was recreated."""
        recreated = not self._is_compatible()
        if recreated:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: qdrant_models.VectorParams(size=self.embedder.dimensions,
                                                                  distance=qdrant_models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: qdrant_models.SparseVectorParams(modifier=qdrant_models.Modifier.IDF)},
            )
            # Payload indexes on specialty/doc_slug belong here when a Qdrant server is used; local mode
            # filters without them and warns if they are created.
        self.sparse.fit([chunk.text for chunk in chunks])  # BM25 length normalisation uses this corpus's average
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors = self.embedder.embed([chunk.text for chunk in part])
            points = []
            for chunk, vector in zip(part, vectors, strict=True):
                ids, weights = self.sparse.encode_document(chunk.text)
                points.append(qdrant_models.PointStruct(
                    id=chunk.chunk_id,
                    vector={DENSE: vector, SPARSE: qdrant_models.SparseVector(indices=ids, values=weights)},
                    payload={**chunk.to_payload(), "embedder": self.embedder.name, "sparse": self.sparse.name},
                ))
            self.client.upsert(self.collection, points)
        keep = {chunk.chunk_id for chunk in chunks}
        stale = [point.id for point in self._scan_points() if str(point.id) not in keep]
        if stale:
            self.client.delete(self.collection, qdrant_models.PointIdsList(points=stale))
        self._schema_ok = True
        return {"chunks": len(chunks), "points": self.count(), "removed": len(stale), "recreated": recreated}

    def _scan_points(self):
        """Every point in the collection (IDs only), read page by page."""
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=_SCROLL_PAGE, offset=offset,
                                                with_payload=False)
            yield from points
            if offset is None:
                return

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def get_source_format(self) -> str | None:
        """Which corpus copy the index was built from ("pdf" or "markdown"), or None when empty."""
        if self.count() == 0:
            return None
        sample, _ = self.client.scroll(self.collection, limit=1, with_payload=["source_format"])
        return sample[0].payload.get("source_format")

    # ---- search ------------------------------------------------------------------------------
    @staticmethod
    def _build_scope_filter(specialty: str | None) -> qdrant_models.Filter | None:
        if not specialty:
            return None
        # Leave out the other seven handbooks but keep the clinic-wide documents (pre-authorisation
        # matrix, cost-share schedule, ...), which also hold rules for this specialty.
        others = [other for other in SPECIALTY_BY_SLUG.values() if other != specialty]
        return qdrant_models.Filter(must_not=[
            qdrant_models.FieldCondition(key="specialty", match=qdrant_models.MatchAny(any=others))])

    def search(self, query: str, limit: int = 4, specialty: str | None = None, mode: str = "hybrid") -> list[Hit]:
        """The best `limit` chunks for `query`, best first. Empty for a blank query or an empty index; raises
        IndexOutOfDateError when the stored collection was built with another layout or embedder."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not query.strip() or self.count() == 0:
            return []
        if self._schema_ok is None:
            self._schema_ok = self._is_compatible()
        if not self._schema_ok:  # a clear error, not a Qdrant vector-name exception
            raise IndexOutOfDateError("the policy index is out of date; run python -m scripts.build_policy_index")
        scope_filter = self._build_scope_filter(specialty)
        ids, weights = self.sparse.encode_query(query)
        sparse_query = qdrant_models.SparseVector(indices=ids, values=weights)
        if mode == "dense":
            query_arguments = {"query": self.embedder.embed([query])[0], "using": DENSE}
        elif mode == "sparse":
            if not ids:  # every query word was a stop word: nothing to match on
                return []
            query_arguments = {"query": sparse_query, "using": SPARSE}
        else:
            per_half = max(PREFETCH, limit)
            prefetch = [qdrant_models.Prefetch(query=self.embedder.embed([query])[0], using=DENSE, filter=scope_filter,
                                               limit=per_half)]
            if ids:
                prefetch.append(qdrant_models.Prefetch(query=sparse_query, using=SPARSE, filter=scope_filter,
                                                       limit=per_half))
            query_arguments = {"prefetch": prefetch,
                               "query": qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF)}
        points = self.client.query_points(
            self.collection, query_filter=scope_filter, limit=limit, with_payload=True, **query_arguments
        ).points
        return [
            Hit(
                score=point.score,
                doc_title=point.payload["doc_title"],
                doc_id=point.payload.get("doc_id"),
                heading=point.payload["heading"],
                specialty=point.payload.get("specialty"),
                text=point.payload["text"],
                chunk_id=str(point.id),
            )
            for point in points
        ]

    def close(self) -> None:
        self.client.close()
        close = getattr(self.embedder, "close", None)  # e.g. the embedding cache (out/embeddings.sqlite)
        if callable(close):
            close()
