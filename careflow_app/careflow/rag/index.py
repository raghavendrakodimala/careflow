"""The Qdrant policy index (M3): one collection, `careflow_policies`, of corpus chunks.

- Local mode: QdrantClient(path=out/qdrant) persists to disk with no server; ":memory:" for tests.
  M8's container can point the same code at a Qdrant server instead.
- build() is idempotent. Chunk IDs are stable (uuid5 of slug + ordinal), so a rebuild overwrites in
  place and deletes chunks that no longer exist. If the embedder changed (different model or size),
  the collection is recreated, because vectors from two embedders cannot be compared.
- search() can be scoped to one specialty, so a question about Cardiology never returns the
  near-duplicate Neurology handbook (the clinic-wide documents stay searchable). M4 adds sparse (BM25) vectors, fusion and a reranker on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ..config import QDRANT_PATH
from .chunker import SPECIALTY_BY_SLUG, Chunk
from .embeddings import Embedder

COLLECTION = "careflow_policies"


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
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = COLLECTION):
        self.client = client
        self.embedder = embedder
        self.collection = collection

    @classmethod
    def open(cls, embedder: Embedder, path: Path | str | None = None) -> PolicyIndex:
        """`path=":memory:"` for a throwaway index; default out/qdrant (one process at a time in local mode)."""
        location = str(path or QDRANT_PATH)
        client = QdrantClient(":memory:") if location == ":memory:" else QdrantClient(path=location)
        return cls(client, embedder)

    # ---- build -------------------------------------------------------------------------------
    def _compatible(self) -> bool:
        if not self.client.collection_exists(self.collection):
            return False
        info = self.client.get_collection(self.collection)
        if info.config.params.vectors.size != self.embedder.dim:
            return False
        sample, _ = self.client.scroll(self.collection, limit=1, with_payload=["embedder"])
        return not sample or sample[0].payload.get("embedder") == self.embedder.name

    def build(self, chunks: list[Chunk], batch: int = 64) -> dict:
        recreated = not self._compatible()
        if recreated:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=self.embedder.dim, distance=qm.Distance.COSINE),
            )
            # Payload indexes on specialty/doc_slug belong here when a Qdrant server is used (M8); local
            # mode filters without them and warns if they are created.
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors = self.embedder.embed([c.text for c in part])
            self.client.upsert(
                self.collection,
                [
                    qm.PointStruct(id=c.chunk_id, vector=v, payload={**c.payload(), "embedder": self.embedder.name})
                    for c, v in zip(part, vectors)
                ],
            )
        keep = {c.chunk_id for c in chunks}
        stale = [p.id for p in self._all_points() if str(p.id) not in keep]
        if stale:
            self.client.delete(self.collection, qm.PointIdsList(points=stale))
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

    # ---- search ------------------------------------------------------------------------------
    def search(self, query: str, limit: int = 4, specialty: str | None = None) -> list[Hit]:
        if not query.strip() or self.count() == 0:
            return []
        flt = None
        if specialty:
            # Leave out the other seven handbooks but keep the clinic-wide documents (pre-authorisation
            # matrix, cost-share schedule, ...), which also hold rules for this specialty.
            others = [s for s in SPECIALTY_BY_SLUG.values() if s != specialty]
            flt = qm.Filter(must_not=[qm.FieldCondition(key="specialty", match=qm.MatchAny(any=others))])
        vector = self.embedder.embed([query])[0]
        points = self.client.query_points(
            self.collection, query=vector, query_filter=flt, limit=limit, with_payload=True
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
