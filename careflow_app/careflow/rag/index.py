"""The Qdrant policy index: one collection, `careflow_policies`, of corpus chunks.

- Local mode: QdrantClient(path=out/qdrant) persists to disk with no server; ":memory:" for tests.
  M8's container can point the same code at a Qdrant server instead.
- Each point has two vectors (M4): "dense" (the embedding model) and "bm25" (sparse keyword weights,
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
from qdrant_client import models as qm

from ..config import QDRANT_PATH
from .chunker import SPECIALTY_BY_SLUG, Chunk
from .embeddings import Embedder
from .sparse import Bm25Encoder

COLLECTION = "careflow_policies"
DENSE, SPARSE = "dense", "bm25"
MODES = ("dense", "sparse", "hybrid")
PREFETCH = 20  # candidates each half contributes before fusion


class IndexOutOfDate(RuntimeError):
    """The collection on disk was built by an older layout or another embedder (e.g. the M3 index)."""


@dataclass(frozen=True)
class Hit:
    score: float
    doc_title: str
    doc_id: str | None
    heading: str
    specialty: str | None
    text: str
    chunk_id: str

    def as_dict(self) -> dict:
        return {
            "document": self.doc_title,
            "document_id": self.doc_id,
            "section": self.heading,
            "score": round(self.score, 3),
            "text": self.text,
        }


class PolicyIndex:
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
        """`url` (CAREFLOW_QDRANT_URL): a Qdrant server, shared by several processes (M6, M8). Otherwise local
        mode at `path` (default out/qdrant; one process at a time) or `":memory:"` for a throwaway index."""
        if url:
            return cls(QdrantClient(url=url), embedder, collection)
        location = str(path or QDRANT_PATH)
        client = QdrantClient(":memory:") if location == ":memory:" else QdrantClient(path=location)
        return cls(client, embedder, collection)

    # ---- build -------------------------------------------------------------------------------
    def _compatible(self) -> bool:
        if not self.client.collection_exists(self.collection):
            return False
        params = self.client.get_collection(self.collection).config.params
        dense = params.vectors.get(DENSE) if isinstance(params.vectors, dict) else None
        if dense is None or dense.size != self.embedder.dim or SPARSE not in (params.sparse_vectors or {}):
            return False  # an M3 collection (one unnamed vector) or another vector size
        sample, _ = self.client.scroll(self.collection, limit=1, with_payload=["embedder", "sparse"])
        return not sample or (
            sample[0].payload.get("embedder") == self.embedder.name and sample[0].payload.get("sparse") == self.sparse.name
        )

    def build(self, chunks: list[Chunk], batch: int = 64) -> dict:
        recreated = not self._compatible()
        if recreated:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: qm.VectorParams(size=self.embedder.dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
            # Payload indexes on specialty/doc_slug belong here when a Qdrant server is used (M8); local
            # mode filters without them and warns if they are created.
        self.sparse.fit([c.text for c in chunks])  # BM25 length normalisation uses this corpus's average
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors = self.embedder.embed([c.text for c in part])
            points = []
            for c, v in zip(part, vectors):
                ids, weights = self.sparse.encode_document(c.text)
                points.append(qm.PointStruct(
                    id=c.chunk_id,
                    vector={DENSE: v, SPARSE: qm.SparseVector(indices=ids, values=weights)},
                    payload={**c.payload(), "embedder": self.embedder.name, "sparse": self.sparse.name},
                ))
            self.client.upsert(self.collection, points)
        keep = {c.chunk_id for c in chunks}
        stale = [p.id for p in self._all_points() if str(p.id) not in keep]
        if stale:
            self.client.delete(self.collection, qm.PointIdsList(points=stale))
        self._schema_ok = True
        return {"chunks": len(chunks), "points": self.count(), "removed": len(stale), "recreated": recreated}

    def _all_points(self):
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=256, offset=offset, with_payload=False)
            yield from points
            if offset is None:
                return

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def source_format(self) -> str | None:
        """Which corpus copy the index was built from ("pdf" or "markdown"), or None when empty."""
        if self.count() == 0:
            return None
        sample, _ = self.client.scroll(self.collection, limit=1, with_payload=["source_format"])
        return sample[0].payload.get("source_format")

    # ---- search ------------------------------------------------------------------------------
    @staticmethod
    def _scope(specialty: str | None) -> qm.Filter | None:
        if not specialty:
            return None
        # Leave out the other seven handbooks but keep the clinic-wide documents (pre-authorisation
        # matrix, cost-share schedule, ...), which also hold rules for this specialty.
        others = [s for s in SPECIALTY_BY_SLUG.values() if s != specialty]
        return qm.Filter(must_not=[qm.FieldCondition(key="specialty", match=qm.MatchAny(any=others))])

    def search(self, query: str, limit: int = 4, specialty: str | None = None, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not query.strip() or self.count() == 0:
            return []
        if self._schema_ok is None:
            self._schema_ok = self._compatible()
        if not self._schema_ok:  # feedback M4-4: a clear error, not a Qdrant vector-name exception
            raise IndexOutOfDate("the policy index is out of date; run python -m scripts.build_policy_index")
        flt = self._scope(specialty)
        ids, weights = self.sparse.encode_query(query)
        sparse_query = qm.SparseVector(indices=ids, values=weights)
        if mode == "dense":
            kwargs = {"query": self.embedder.embed([query])[0], "using": DENSE}
        elif mode == "sparse":
            if not ids:
                return []
            kwargs = {"query": sparse_query, "using": SPARSE}
        else:
            prefetch = [qm.Prefetch(query=self.embedder.embed([query])[0], using=DENSE, filter=flt,
                                    limit=max(PREFETCH, limit))]
            if ids:
                prefetch.append(qm.Prefetch(query=sparse_query, using=SPARSE, filter=flt, limit=max(PREFETCH, limit)))
            kwargs = {"prefetch": prefetch, "query": qm.FusionQuery(fusion=qm.Fusion.RRF)}
        points = self.client.query_points(
            self.collection, query_filter=flt, limit=limit, with_payload=True, **kwargs
        ).points
        return [
            Hit(
                score=p.score,
                doc_title=p.payload["doc_title"],
                doc_id=p.payload.get("doc_id"),
                heading=p.payload["heading"],
                specialty=p.payload.get("specialty"),
                text=p.payload["text"],
                chunk_id=str(p.id),
            )
            for p in points
        ]

    def close(self) -> None:
        self.client.close()
        close = getattr(self.embedder, "close", None)  # e.g. the embedding cache (out/embeddings.sqlite)
        if callable(close):
            close()
