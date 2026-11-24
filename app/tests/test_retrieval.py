"""Retrieval: BM25 sparse vectors, hybrid fusion in Qdrant, reranking and the abstain threshold.

Offline: the hashing embedder stands in for the dense model and a fake reranker for the cross-encoder.
Quality with the real models is measured live by scripts/run_retrieval_eval.py.
"""

import pytest
from qdrant_client import models as qdrant_models

from src.config import Settings
from src.rag import embeddings
from src.rag.chunker import SPECIALTY_BY_SLUG, load_corpus_chunks
from src.rag.index import MODES, PolicyIndex
from src.rag.rerank import CrossEncoderReranker, build_reranker
from src.rag.retriever import PolicyRetriever, open_policy_search
from src.rag.sparse import Bm25Encoder, extract_terms
from src.tools import CareFlowTools


@pytest.fixture(scope="module")
def chunks():
    return load_corpus_chunks(source="pdf")


@pytest.fixture(scope="module")
def index(chunks):
    policy_index = PolicyIndex.open(embeddings.HashingEmbedder(), ":memory:")
    policy_index.build(chunks)
    yield policy_index
    policy_index.close()


class KeywordReranker:
    """Fake cross-encoder: share of the query's terms found in the passage."""

    name = "fake-keyword"

    def __init__(self):
        self.calls = 0

    def score_passages(self, query, passages):
        self.calls += 1
        query_terms = set(extract_terms(query))
        return [len(query_terms & set(extract_terms(passage))) / max(1, len(query_terms)) for passage in passages]


# ---- BM25 ------------------------------------------------------------------------------------------
def test_terms_keep_codes_whole_and_split():
    found_terms = extract_terms("Is a Colonoscopy (GAST-COLO) covered under OP-CS-014 within 5 business days?")
    assert {"gast-colo", "gast", "colo", "op-cs-014", "colonoscopy", "5", "business", "days"} <= set(found_terms)
    assert "is" not in found_terms and "within" not in found_terms  # stop words


def test_bm25_weights_saturate_and_normalise_by_length():
    encoder = Bm25Encoder().fit(["alpha beta", "alpha beta gamma delta"])
    one = dict(zip(*encoder.encode_document("alpha"), strict=True))
    three = dict(zip(*encoder.encode_document("alpha alpha alpha"), strict=True))
    one_weight, three_weight = next(iter(one.values())), next(iter(three.values()))
    assert one_weight < three_weight < one_weight * 3  # more occurrences count, with diminishing returns
    ids, values = encoder.encode_query("alpha alpha beta")
    assert len(ids) == 2 and values == [1.0, 1.0]


# ---- index -----------------------------------------------------------------------------------------
def test_collection_has_dense_and_sparse_vectors_and_is_idempotent(index, chunks):
    params = index.client.get_collection(index.collection).config.params
    assert "dense" in params.vectors and "bm25" in params.sparse_vectors
    assert params.sparse_vectors["bm25"].modifier == qdrant_models.Modifier.IDF
    again = index.build(chunks)
    assert again["recreated"] is False and again["points"] == len(chunks) and index.get_source_format() == "pdf"


def test_a_single_vector_collection_is_recreated_with_named_vectors(chunks):
    policy_index = PolicyIndex.open(embeddings.HashingEmbedder(), ":memory:")
    policy_index.client.create_collection(policy_index.collection, vectors_config=qdrant_models.VectorParams(size=512, distance=qdrant_models.Distance.COSINE))
    out = policy_index.build(chunks[:5])
    assert out["recreated"] is True and policy_index.count() == 5
    policy_index.close()


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_respects_the_specialty_scope(index, mode):
    for specialty in SPECIALTY_BY_SLUG.values():
        hits = index.search("new-patient consultation length and list price", limit=8, specialty=specialty, mode=mode)
        assert hits and all(hit.specialty in (None, specialty) for hit in hits), (mode, specialty)


def test_sparse_and_hybrid_find_an_exact_policy_number(index):
    for mode in ("sparse", "hybrid"):
        hits = index.search("OP-CS-014", limit=3, mode=mode)
        assert hits and hits[0].doc_id == "OP-CS-014", mode


def test_hybrid_fuses_both_rankings(index):
    query = "pre-authorisation approval window for DERM-EXCI under CIVIC-BASE"
    dense = [hit.chunk_id for hit in index.search(query, limit=20, mode="dense")]
    sparse = [hit.chunk_id for hit in index.search(query, limit=20, mode="sparse")]
    hybrid = [hit.chunk_id for hit in index.search(query, limit=20, mode="hybrid")]
    assert hybrid and set(hybrid) <= set(dense) | set(sparse)  # fusion only re-ranks the two candidate lists
    # Reciprocal Rank Fusion: the fused winner is ranked near the top by at least one of the two.
    assert hybrid[0] in dense[:3] + sparse[:3]
    with pytest.raises(ValueError):
        index.search(query, mode="semantic")


def test_a_query_with_only_stop_words_still_searches_dense(index):
    assert index.search("what is it", limit=2, mode="hybrid")
    assert index.search("what is it", limit=2, mode="sparse") == []


# ---- retriever: rerank and abstain -------------------------------------------------------------
def test_reranker_reorders_candidates_and_reports_its_scores(index):
    reranker = KeywordReranker()
    retriever = PolicyRetriever(index, reranker=reranker, candidates=10)
    hits = retriever.search("Dermatology Consultation DERM-CONS list price", limit=3)
    assert reranker.calls == 1 and len(hits) == 3
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    assert all(0 <= hit.score <= 1 for hit in hits)


def test_retriever_abstains_below_the_threshold(index):
    retriever = PolicyRetriever(index, reranker=KeywordReranker(), min_score=0.6)
    out = retriever.retrieve("valet parking zeppelin karaoke", limit=4)
    assert out.hits == [] and out.abstained and out.top_score < 0.6
    ok = retriever.retrieve("Dermatology Consultation DERM-CONS list price", limit=4)
    assert ok.hits and not ok.abstained and ok.top_score >= 0.6
    assert retriever.search("valet parking zeppelin karaoke") == []  # search() is retrieve().hits


def _make_hit(doc, score, number):
    from src.rag.index import Hit

    return Hit(score=score, doc_title=doc, doc_id=None, heading="h", specialty=None, text=f"{doc} {number}", chunk_id=f"{doc}{number}")


class ListIndex:
    """Index stub that returns fixed hits in retrieval order."""

    def __init__(self, hits):
        self.hits = hits

    def search(self, query, limit=4, specialty=None, mode="hybrid"):
        return self.hits[:limit]


def test_documents_are_spread_one_then_two_per_document():
    hits = [_make_hit("A", 0.9, 1), _make_hit("A", 0.8, 2), _make_hit("A", 0.7, 3), _make_hit("B", 0.6, 1), _make_hit("A", 0.5, 4)]
    capped = PolicyRetriever(ListIndex(hits), max_per_document=2).rank("q")
    assert [hit.text for hit in capped] == ["A 1", "B 1", "A 2", "A 3", "A 4"]  # best stays first; nothing dropped
    assert PolicyRetriever(ListIndex(hits), max_per_document=None).rank("q") == hits


class _Scores:
    name = "fixed"

    def __init__(self, table):
        self.table = table

    def score_passages(self, query, passages):
        return [self.table[passage] for passage in passages]


def test_a_passage_that_ties_with_the_best_is_not_pushed_out_by_spreading():  # golden case EV-019
    # Three sections of one document tie at the top; other documents score near zero.
    hits = [_make_hit("S", 0, 1), _make_hit("S", 0, 2), _make_hit("S", 0, 3), _make_hit("H", 0, 1), _make_hit("M", 0, 1), _make_hit("I", 0, 1)]
    table = {"S 1": 0.9995, "S 2": 0.9990, "S 3": 0.9980, "H 1": 0.5, "M 1": 0.05, "I 1": 0.001}
    top4 = [hit.text for hit in PolicyRetriever(ListIndex(hits), reranker=_Scores(table)).rank("q")[:4]]
    assert top4 == ["S 1", "S 2", "S 3", "H 1"]
    # A passage that is merely good (not a tie) is still spread as before: one per document first.
    table["S 2"], table["S 3"] = 0.9, 0.8
    ranked = [hit.text for hit in PolicyRetriever(ListIndex(hits), reranker=_Scores(table)).rank("q")]
    assert ranked[:4] == ["S 1", "H 1", "M 1", "I 1"] and len(ranked) == 6


def test_fused_order_keeps_what_retrieval_ranked_first():
    # Retrieval ranked B first; the reranker prefers A1, A2, A3. Pure rerank order puts B last; fusion of
    # the two ranks keeps B in the top two. The shown score is the cross-encoder's either way.
    hits = [_make_hit("B", 0, 1), _make_hit("A", 0, 1), _make_hit("A", 0, 2), _make_hit("A", 0, 3)]

    class PreferringReranker:
        name = "prefers-A"

        def score_passages(self, query, passages):
            return [{"B 1": 0.1, "A 1": 0.9, "A 2": 0.8, "A 3": 0.7}[passage] for passage in passages]

    pure = PolicyRetriever(ListIndex(hits), reranker=PreferringReranker(), max_per_document=None).rank("q")
    fused = PolicyRetriever(ListIndex(hits), reranker=PreferringReranker(), max_per_document=None, fuse=True).rank("q")
    assert [hit.text for hit in pure] == ["A 1", "A 2", "A 3", "B 1"]
    assert "B 1" in [hit.text for hit in fused[:2]] and next(hit.score for hit in fused if hit.text == "B 1") == 0.1


def test_without_a_reranker_the_retriever_never_abstains(index):
    retriever = PolicyRetriever(index, reranker=None, min_score=0.99)
    out = retriever.retrieve("valet parking zeppelin karaoke", limit=4)
    assert len(out.hits) == 4 and not out.abstained and out.top_score is None


def test_search_policies_says_when_the_documents_do_not_cover_a_question(index):
    tools = CareFlowTools(policy_index=PolicyRetriever(index, reranker=KeywordReranker(), min_score=0.6))
    result = tools.call("search_policies", {"query": "valet parking zeppelin karaoke"})
    assert result["ok"] and result["count"] == 0 and result["results"] == []
    assert "do not cover" in result["note"] and "flag_for_human" in result["note"]
    hit = tools.call("search_policies", {"query": "Dermatology Consultation DERM-CONS list price"})
    assert hit["count"] and "exactly" in hit["note"]


def test_retriever_is_configured_from_settings(tmp_path):
    retriever = open_policy_search(Settings(careflow_embedding_model="local", careflow_reranker_model="none"), ":memory:")
    assert retriever.reranker is None and retriever.min_score is None and retriever.count() == 0
    retriever.close()
    retriever = open_policy_search(Settings(careflow_embedding_model="local", careflow_rerank_min_score=0.3,
                                      careflow_retrieval_candidates=12), ":memory:")
    assert isinstance(retriever.reranker, CrossEncoderReranker) and retriever.min_score == 0.3 and retriever.candidates == 12
    assert retriever.reranker._model is None  # loaded lazily: building a retriever downloads nothing
    retriever.close()
    assert build_reranker("NONE") is None and build_reranker("") is None


def test_cross_encoder_scores_are_probabilities(monkeypatch):
    reranker = CrossEncoderReranker("fake/model")

    class FakeModel:
        def predict(self, pairs, **options):
            assert options["activation_fn"] is not None  # raw logits requested, our sigmoid applied
            return [5.0, 0.0, -5.0][: len(pairs)]

    monkeypatch.setattr(reranker, "_load", lambda: FakeModel())
    scores = reranker.score_passages("q", ["a", "b", "c"])
    assert scores[0] > 0.99 and scores[1] == 0.5 and scores[2] < 0.01
    assert reranker.score_passages("q", []) == []


# ---- specialty inference, input limits and housekeeping ------------------------------------------------------------------------
from src.rag.index import IndexOutOfDateError  # noqa: E402
from src.rag.retriever import infer_specialty  # noqa: E402


@pytest.mark.parametrize("query, expected", [
    ("Pre-authorisation window for an echocardiogram (CARD-ECHO) under GOLD", "Cardiology"),
    ("What will a colonoscopy cost on MERIDIAN-SILVER?", "Gastroenterology"),
    ("MRI Brain (NEUR-MRI) co-pay", "Neurology"),
    ("knee arthroscopy price", "Orthopaedics"),
    ("How long is an ENT consultation?", "ENT"),
    ("Does an MRI need pre-authorisation?", None),  # knee or brain: ambiguous
    ("How quickly must a Priority referral be actioned?", None),
    ("cardiology or neurology, which is faster?", None),  # two specialties
    ("I went there every year", None),  # "ear" only as a whole word
])
def test_specialty_is_inferred_only_when_unambiguous(query, expected):
    assert infer_specialty(query) == expected


def test_unscoped_questions_get_at_most_one_handbook_up_front():
    make_handbook_hit = lambda specialty, number: Hit(score=0, doc_title=f"{specialty} Service Line Handbook", doc_id=None, heading="h",  # noqa: E731
                          specialty=specialty, text=f"{specialty} {number}", chunk_id=f"{specialty}{number}")
    from src.rag.index import Hit

    hits = [make_handbook_hit("Neurology", 1), make_handbook_hit("Orthopaedics", 1), make_handbook_hit("Cardiology", 1), _make_hit("Co-pay", 0, 1), _make_hit("Preauth", 0, 1)]
    unscoped = PolicyRetriever(ListIndex(hits), infer=False).rank("Does an MRI need pre-authorisation?")
    assert [hit.text for hit in unscoped[:3]] == ["Neurology 1", "Co-pay 1", "Preauth 1"]
    scoped = PolicyRetriever(ListIndex(hits)).rank("x", specialty="Neurology")
    assert [hit.text for hit in scoped[:2]] == ["Neurology 1", "Orthopaedics 1"]  # scoped: the family limit is off


def test_inferred_scope_reaches_the_index(index):
    retriever = PolicyRetriever(index)
    out = retriever.retrieve("What will a colonoscopy cost on MERIDIAN-SILVER?", limit=8)
    assert out.specialty == "Gastroenterology"
    assert all(hit.specialty in (None, "Gastroenterology") for hit in out.hits)


def test_an_out_of_date_index_on_disk_is_reported_not_crashed(tmp_path):
    policy_index = PolicyIndex.open(embeddings.HashingEmbedder(), tmp_path / "q")
    policy_index.client.create_collection(policy_index.collection, vectors_config=qdrant_models.VectorParams(size=512, distance=qdrant_models.Distance.COSINE))
    policy_index.client.upsert(policy_index.collection, [qdrant_models.PointStruct(id=1, vector=[0.1] * 512, payload={"doc_title": "x"})])
    with pytest.raises(IndexOutOfDateError):
        policy_index.search("referral")
    result = CareFlowTools(policy_index=PolicyRetriever(policy_index)).call("search_policies", {"query": "referral"})
    assert result["ok"] is False and "build_policy_index" in result["error"]
    policy_index.close()


def test_patient_identifiers_never_reach_the_search(index):
    seen = []

    class Spy:
        def retrieve(self, query, limit=4, specialty=None):
            seen.append(query)
            return PolicyRetriever(index).retrieve(query, limit, specialty)

    tools = CareFlowTools(policy_index=Spy())
    result = tools.call("search_policies", {"query": "Does MHP-P-00001 (member MER90272292, born 26/12/1981, "
                                                "referral MHP-R-00012) need pre-authorisation?"})
    assert result["ok"] and result["redacted_identifiers"] == 4
    assert seen == ["Does (member , born , referral ) need pre-authorisation?"]
    only_ids = tools.call("search_policies", {"query": "MHP-P-00001"})
    assert only_ids["ok"] is False and len(seen) == 1


def test_search_inputs_are_bounded():
    tools = CareFlowTools(policy_index=PolicyRetriever(ListIndex([])))
    for bad in ({"query": "x" * 501}, {"query": ""}, {"query": "x", "limit": 0}, {"query": "x", "limit": 9}):
        assert tools.call("search_policies", bad)["ok"] is False, bad
    assert tools.call("find_appointment_slots", {"specialty": "ENT", "limit": 21})["ok"] is False


def test_results_report_abstention_per_call(index):
    tools = CareFlowTools(policy_index=PolicyRetriever(index, reranker=KeywordReranker(), min_score=0.6))
    assert tools.call("search_policies", {"query": "valet parking zeppelin karaoke"})["abstained"] is True
    assert tools.call("search_policies", {"query": "Dermatology Consultation DERM-CONS list price"})["abstained"] is False


def test_closing_the_index_closes_the_embedder():
    closed = []
    embedder = embeddings.HashingEmbedder()
    embedder.close = lambda: closed.append(True)
    PolicyIndex.open(embedder, ":memory:").close()
    assert closed == [True]


def test_reranker_prefers_the_local_cache(monkeypatch):
    import sys
    import types

    calls = []

    class FakeCrossEncoder:
        def __init__(self, name, **options):
            calls.append(options.get("local_files_only", False))
            if options.get("local_files_only"):
                raise OSError("not cached")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    CrossEncoderReranker("fake/model")._load()
    assert calls == [True, False]  # cache first, then one download
