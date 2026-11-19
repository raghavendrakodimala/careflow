"""M3 policy index: chunking, the Qdrant collection, specialty scoping, persistence, embedding cache.

Offline: the deterministic HashingEmbedder stands in for the real embedding model, and LiteLLM is faked.
Retrieval quality with the real model is measured live by scripts/run_m3_demo.py.
"""

import json
import re
from types import SimpleNamespace

import pytest

from careflow.config import DATA_DIR, Settings
from careflow.rag import embeddings as E
from careflow.rag.chunker import MAX_CHARS, SPECIALTY_BY_SLUG, load_corpus_chunks
from careflow.rag.index import PolicyIndex

INDEX = json.loads((DATA_DIR / "corpus" / "index.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunks():
    return load_corpus_chunks()


@pytest.fixture()
def index(chunks):
    ix = PolicyIndex.open(E.HashingEmbedder(), ":memory:")
    ix.build(chunks)
    yield ix
    ix.close()


# ---- chunking ----------------------------------------------------------------------------------
def test_every_document_is_chunked_with_its_exact_title(chunks):
    assert {c.doc_title for c in chunks} == {d["title"] for d in INDEX}  # eval must_cite uses these titles
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.chunk_id for c in load_corpus_chunks()] == [c.chunk_id for c in chunks]  # stable IDs


def test_no_text_is_lost(chunks):
    for doc in INDEX:
        source = (DATA_DIR / doc["markdown"]).read_text(encoding="utf-8")
        body = "\n".join(c.text for c in chunks if c.doc_slug == doc["slug"])
        for line in source.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line != "---":
                assert line in body, (doc["slug"], line[:60])


def test_each_chunk_starts_with_its_context_header(chunks):
    for c in chunks:
        header = c.text.split("\n", 1)[0]
        assert header.startswith(f"[{c.doc_title}") and c.heading in header


def test_tables_are_never_split(chunks):
    card = [c for c in chunks if c.doc_slug == "specialty-cardiology" and "| CARD-CONS |" in c.text]
    assert card and all("| CARD-CATH |" in c.text for c in card)
    for c in chunks:  # a chunk that opens a table row block also closes it
        rows = [ln for ln in c.text.splitlines() if ln.startswith("|")]
        if rows:
            assert re.match(r"\|[-| :]+\|", rows[1]) or len(rows) == 1


def test_chunks_stay_near_the_size_target(chunks):
    long = [c for c in chunks if len(c.text) > MAX_CHARS + 200]
    for c in long:  # only a single paragraph that is itself too long (kept whole) may exceed it
        body = c.text.split("\n", 1)[1]
        assert "\n\n" not in body, (c.doc_slug, c.heading)


def test_specialty_and_codes_are_tagged(chunks):
    for c in chunks:
        assert c.specialty == SPECIALTY_BY_SLUG.get(c.doc_slug)
    gast = [c for c in chunks if "GAST-COLO" in c.text]
    assert gast and all("GAST-COLO" in c.procedure_codes for c in gast)
    assert any("MERIDIAN-SILVER" in c.plan_codes for c in chunks if c.doc_slug == "copay-schedule")
    assert {c.doc_id for c in chunks if c.doc_slug == "clinical-escalation"} == {"OP-CS-014"}
    assert {c.doc_id for c in chunks if c.doc_slug == "specialty-neurology"} == {"MHP-SLH-NEUR-01"}  # "Document ref"


# ---- index -------------------------------------------------------------------------------------
def test_build_is_idempotent_and_removes_stale_chunks(chunks, index):
    assert index.count() == len(chunks)
    again = index.build(chunks)
    assert again["recreated"] is False and again["points"] == len(chunks) and again["removed"] == 0
    smaller = index.build(chunks[:-3])
    assert smaller["removed"] == 3 and index.count() == len(chunks) - 3


def test_a_different_embedder_recreates_the_collection(chunks, index):
    index.embedder = E.HashingEmbedder(dim=256)
    out = index.build(chunks[:10])
    assert out["recreated"] is True and index.count() == 10


@pytest.mark.parametrize("specialty", sorted(SPECIALTY_BY_SLUG.values()))
def test_specialty_scope_never_returns_another_handbook(index, specialty):
    hits = index.search("new-patient consultation length and list price", limit=8, specialty=specialty)
    assert hits and all(h.specialty in (None, specialty) for h in hits)


def test_unscoped_search_finds_exact_codes(index):
    hits = index.search("GAST-COLO colonoscopy pre-authorisation", limit=5)
    assert any("GAST-COLO" in h.text for h in hits)
    assert index.search("   ") == []


def test_index_persists_on_disk(tmp_path, chunks):
    ix = PolicyIndex.open(E.HashingEmbedder(), tmp_path / "q")
    ix.build(chunks)
    ix.close()
    again = PolicyIndex.open(E.HashingEmbedder(), tmp_path / "q")
    assert again.count() == len(chunks) and again.search("deflection script", limit=1)
    again.close()


# ---- LiteLLM embedder (faked) -------------------------------------------------------------------
def test_litellm_embedder_batches_caches_and_passes_dimensions(monkeypatch, tmp_path):
    import litellm

    calls = []

    def fake_embedding(model, input, dimensions, **kw):
        calls.append({"model": model, "n": len(input), "dimensions": dimensions, "api_key": kw.get("api_key")})
        return SimpleNamespace(data=[{"embedding": [float(len(t))] + [1.0] * (dimensions - 1)} for t in input])

    monkeypatch.setattr(litellm, "embedding", fake_embedding)
    settings = Settings(careflow_embedding_model="gemini/test-embed", careflow_embedding_dimensions=8,
                        gemini_api_key="k")
    emb = E.LiteLLMEmbedder(settings, cache_path=tmp_path / "c.sqlite")
    texts = [f"text {i}" for i in range(E.LiteLLMEmbedder.BATCH + 5)]
    first = emb.embed(texts)
    assert [c["n"] for c in calls] == [E.LiteLLMEmbedder.BATCH, 5]
    assert calls[0]["dimensions"] == 8 and calls[0]["model"] == "gemini/test-embed" and calls[0]["api_key"] == "k"
    assert all(abs(sum(x * x for x in v) - 1) < 1e-5 for v in first)  # normalised

    emb2 = E.LiteLLMEmbedder(settings, cache_path=tmp_path / "c.sqlite")  # cache survives a restart
    assert emb2.embed(texts[:3]) == first[:3] and len(calls) == 2
    emb.close()
    emb2.close()


def test_litellm_embedder_rejects_a_wrong_vector_size(monkeypatch, tmp_path):
    import litellm

    monkeypatch.setattr(litellm, "embedding", lambda model, input, dimensions, **kw: SimpleNamespace(
        data=[{"embedding": [1.0, 2.0]} for _ in input]))
    emb = E.LiteLLMEmbedder(Settings(careflow_embedding_dimensions=8), cache_path=":memory:")
    with pytest.raises(E.EmbeddingError):
        emb.embed(["x"])


def test_local_embedding_model_needs_no_api():
    assert isinstance(E.default_embedder(Settings(careflow_embedding_model="local")), E.HashingEmbedder)


def test_litellm_embedder_paces_by_texts_per_minute(monkeypatch, tmp_path):
    # Gemini's free tier counts every text in a batch as a request (100/min). The live build hit it.
    import litellm

    monkeypatch.setattr(litellm, "embedding", lambda model, input, dimensions, **kw: SimpleNamespace(
        data=[{"embedding": [1.0] * dimensions} for _ in input]))
    emb = E.LiteLLMEmbedder(Settings(careflow_embedding_dimensions=4, careflow_embedding_texts_per_minute=90),
                            cache_path=":memory:")
    slept = []
    monkeypatch.setattr(emb, "_sleep", slept.append)
    emb.embed([f"t{i}" for i in range(150)])  # batches of 50: the second would make 100 > 90 in one minute
    assert len(slept) == 2 and all(s > 50 for s in slept)


def test_rate_limit_waits_for_the_quota_window(monkeypatch):
    import litellm

    attempts = []

    def flaky(model, input, dimensions, **kw):
        attempts.append(1)
        if len(attempts) == 1:
            raise litellm.RateLimitError("quota", llm_provider="gemini", model=model)
        return SimpleNamespace(data=[{"embedding": [1.0] * dimensions} for _ in input])

    monkeypatch.setattr(litellm, "embedding", flaky)
    emb = E.LiteLLMEmbedder(Settings(careflow_embedding_dimensions=4), cache_path=":memory:")
    slept = []
    monkeypatch.setattr(emb, "_sleep", slept.append)
    assert len(emb.embed(["x"])[0]) == 4
    assert slept == [E.LiteLLMEmbedder.WINDOW_S + 2]
