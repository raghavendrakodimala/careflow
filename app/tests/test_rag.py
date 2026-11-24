"""Policy index: chunking, the Qdrant collection, specialty scoping, persistence, embedding cache.

Offline: the deterministic HashingEmbedder stands in for the real embedding model, and LiteLLM is faked.
Retrieval quality with the real model is measured live by scripts/run_memory_demo.py.
"""

import json
import re
from types import SimpleNamespace

import pytest

from src.config import DATA_DIR, Settings
from src.rag import embeddings
from src.rag.chunker import MAX_CHARACTERS, SPECIALTY_BY_SLUG, load_corpus_chunks
from src.rag.index import PolicyIndex

INDEX = json.loads((DATA_DIR / "corpus" / "index.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunks():
    return load_corpus_chunks()


@pytest.fixture()
def index(chunks):
    policy_index = PolicyIndex.open(embeddings.HashingEmbedder(), ":memory:")
    policy_index.build(chunks)
    yield policy_index
    policy_index.close()


# ---- chunking ----------------------------------------------------------------------------------
def test_every_document_is_chunked_with_its_exact_title(chunks):
    assert {chunk.doc_title for chunk in chunks} == {document["title"] for document in INDEX}  # eval must_cite uses these titles
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [chunk.chunk_id for chunk in load_corpus_chunks()] == [chunk.chunk_id for chunk in chunks]  # stable IDs


def test_no_text_is_lost(chunks):
    for doc in INDEX:
        source = (DATA_DIR / doc["markdown"]).read_text(encoding="utf-8")
        body = "\n".join(chunk.text for chunk in chunks if chunk.doc_slug == doc["slug"])
        for line in source.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line != "---":
                assert line in body, (doc["slug"], line[:60])


def test_each_chunk_starts_with_its_context_header(chunks):
    for chunk in chunks:
        header = chunk.text.split("\n", 1)[0]
        assert header.startswith(f"[{chunk.doc_title}") and chunk.heading in header


def test_tables_are_never_split(chunks):
    card = [chunk for chunk in chunks if chunk.doc_slug == "specialty-cardiology" and "| CARD-CONS |" in chunk.text]
    assert card and all("| CARD-CATH |" in chunk.text for chunk in card)
    for chunk in chunks:  # a chunk that opens a table row block also closes it
        rows = [line for line in chunk.text.splitlines() if line.startswith("|")]
        if rows:
            assert re.match(r"\|[-| :]+\|", rows[1]) or len(rows) == 1


def test_chunks_stay_near_the_size_target(chunks):
    long = [chunk for chunk in chunks if len(chunk.text) > MAX_CHARACTERS + 200]
    for chunk in long:  # only a single paragraph that is itself too long (kept whole) may exceed it
        body = chunk.text.split("\n", 1)[1]
        assert "\n\n" not in body, (chunk.doc_slug, chunk.heading)


def test_specialty_and_codes_are_tagged(chunks):
    for chunk in chunks:
        assert chunk.specialty == SPECIALTY_BY_SLUG.get(chunk.doc_slug)
    gast = [chunk for chunk in chunks if "GAST-COLO" in chunk.text]
    assert gast and all("GAST-COLO" in chunk.procedure_codes for chunk in gast)
    assert any("MERIDIAN-SILVER" in chunk.plan_codes for chunk in chunks if chunk.doc_slug == "copay-schedule")
    assert {chunk.doc_id for chunk in chunks if chunk.doc_slug == "clinical-escalation"} == {"OP-CS-014"}
    assert {chunk.doc_id for chunk in chunks if chunk.doc_slug == "specialty-neurology"} == {"MHP-SLH-NEUR-01"}  # "Document ref"


# ---- index -------------------------------------------------------------------------------------
def test_build_is_idempotent_and_removes_stale_chunks(chunks, index):
    assert index.count() == len(chunks)
    again = index.build(chunks)
    assert again["recreated"] is False and again["points"] == len(chunks) and again["removed"] == 0
    smaller = index.build(chunks[:-3])
    assert smaller["removed"] == 3 and index.count() == len(chunks) - 3


def test_a_different_embedder_recreates_the_collection(chunks, index):
    index.embedder = embeddings.HashingEmbedder(dimensions=256)
    out = index.build(chunks[:10])
    assert out["recreated"] is True and index.count() == 10


@pytest.mark.parametrize("specialty", sorted(SPECIALTY_BY_SLUG.values()))
def test_specialty_scope_never_returns_another_handbook(index, specialty):
    hits = index.search("new-patient consultation length and list price", limit=8, specialty=specialty)
    assert hits and all(hit.specialty in (None, specialty) for hit in hits)


def test_unscoped_search_finds_exact_codes(index):
    hits = index.search("GAST-COLO colonoscopy pre-authorisation", limit=5)
    assert any("GAST-COLO" in hit.text for hit in hits)
    assert index.search("   ") == []


def test_index_persists_on_disk(tmp_path, chunks):
    policy_index = PolicyIndex.open(embeddings.HashingEmbedder(), tmp_path / "q")
    policy_index.build(chunks)
    policy_index.close()
    again = PolicyIndex.open(embeddings.HashingEmbedder(), tmp_path / "q")
    assert again.count() == len(chunks) and again.search("deflection script", limit=1)
    again.close()


# ---- LiteLLM embedder (faked) -------------------------------------------------------------------
def test_litellm_embedder_batches_caches_and_passes_dimensions(monkeypatch, tmp_path):
    import litellm

    calls = []

    def record_embedding_call(model, input, dimensions, **options):
        calls.append({"model": model, "n": len(input), "dimensions": dimensions, "api_key": options.get("api_key")})
        return SimpleNamespace(data=[{"embedding": [float(len(text))] + [1.0] * (dimensions - 1)} for text in input])

    monkeypatch.setattr(litellm, "embedding", record_embedding_call)
    settings = Settings(careflow_embedding_model="gemini/test-embed", careflow_embedding_dimensions=8,
                        gemini_api_key="k")
    embedder = embeddings.LiteLLMEmbedder(settings, cache_path=tmp_path / "c.sqlite")
    texts = [f"text {number}" for number in range(embeddings.LiteLLMEmbedder.BATCH + 5)]
    first = embedder.embed(texts)
    assert [call["n"] for call in calls] == [embeddings.LiteLLMEmbedder.BATCH, 5]
    assert calls[0]["dimensions"] == 8 and calls[0]["model"] == "gemini/test-embed" and calls[0]["api_key"] == "k"
    assert all(abs(sum(component * component for component in vector) - 1) < 1e-5 for vector in first)  # normalised

    second_embedder = embeddings.LiteLLMEmbedder(settings, cache_path=tmp_path / "c.sqlite")  # cache survives a restart
    assert second_embedder.embed(texts[:3]) == first[:3] and len(calls) == 2
    embedder.close()
    second_embedder.close()


def test_litellm_embedder_rejects_a_wrong_vector_size(monkeypatch, tmp_path):
    import litellm

    monkeypatch.setattr(litellm, "embedding", lambda model, input, dimensions, **options: SimpleNamespace(
        data=[{"embedding": [1.0, 2.0]} for _ in input]))
    embedder = embeddings.LiteLLMEmbedder(Settings(careflow_embedding_dimensions=8), cache_path=":memory:")
    with pytest.raises(embeddings.EmbeddingError):
        embedder.embed(["x"])


def test_local_embedding_model_needs_no_api():
    assert isinstance(embeddings.build_embedder(Settings(careflow_embedding_model="local")), embeddings.HashingEmbedder)


def test_litellm_embedder_paces_by_texts_per_minute(monkeypatch, tmp_path):
    # Gemini's free tier counts every text in a batch as a request (100/min).
    import litellm

    monkeypatch.setattr(litellm, "embedding", lambda model, input, dimensions, **options: SimpleNamespace(
        data=[{"embedding": [1.0] * dimensions} for _ in input]))
    embedder = embeddings.LiteLLMEmbedder(Settings(careflow_embedding_dimensions=4, careflow_embedding_texts_per_minute=90),
                            cache_path=":memory:")
    slept = []
    monkeypatch.setattr(embedder, "_sleep", slept.append)
    embedder.embed([f"t{number}" for number in range(150)])  # batches of 50: the second would make 100 > 90 in one minute
    assert len(slept) == 2 and all(seconds > 50 for seconds in slept)


def test_rate_limit_waits_for_the_quota_window(monkeypatch):
    import litellm

    attempts = []

    def fail_once(model, input, dimensions, **options):
        attempts.append(1)
        if len(attempts) == 1:
            raise litellm.RateLimitError("quota", llm_provider="gemini", model=model)
        return SimpleNamespace(data=[{"embedding": [1.0] * dimensions} for _ in input])

    monkeypatch.setattr(litellm, "embedding", fail_once)
    embedder = embeddings.LiteLLMEmbedder(Settings(careflow_embedding_dimensions=4), cache_path=":memory:")
    slept = []
    monkeypatch.setattr(embedder, "_sleep", slept.append)
    assert len(embedder.embed(["x"])[0]) == 4
    assert slept == [embeddings.LiteLLMEmbedder.WINDOW_S + 2]
