"""M4 retrieval: BM25 sparse vectors, hybrid fusion in Qdrant, reranking and the abstain threshold.

Offline: the hashing embedder stands in for the dense model and a fake reranker for the cross-encoder.
Quality with the real models is measured live by scripts/run_m4_eval.py.
"""

import pytest
from qdrant_client import models as qm

from careflow.config import Settings
from careflow.rag import embeddings as E
from careflow.rag.chunker import SPECIALTY_BY_SLUG, load_corpus_chunks
from careflow.rag.index import MODES, PolicyIndex
from careflow.rag.rerank import CrossEncoderReranker, default_reranker
from careflow.rag.retriever import PolicyRetriever, open_policy_search
from careflow.rag.sparse import Bm25Encoder, terms
from careflow.tools import CareFlowTools


@pytest.fixture(scope="module")
def chunks():
    return load_corpus_chunks(source="pdf")


@pytest.fixture(scope="module")
def index(chunks):
    ix = PolicyIndex.open(E.HashingEmbedder(), ":memory:")
    ix.build(chunks)
    yield ix
    ix.close()


class KeywordReranker:
    """Fake cross-encoder: share of the query's terms found in the passage."""

    name = "fake-keyword"

    def __init__(self):
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        q = set(terms(query))
        return [len(q & set(terms(p))) / max(1, len(q)) for p in passages]


# ---- BM25 ------------------------------------------------------------------------------------------
def test_terms_keep_codes_whole_and_split():
    t = terms("Is a Colonoscopy (GAST-COLO) covered under OP-CS-014 within 5 business days?")
    assert {"gast-colo", "gast", "colo", "op-cs-014", "colonoscopy", "5", "business", "days"} <= set(t)
    assert "is" not in t and "within" not in t  # stop words


def test_bm25_weights_saturate_and_normalise_by_length():
    enc = Bm25Encoder().fit(["alpha beta", "alpha beta gamma delta"])
    one = dict(zip(*enc.encode_document("alpha")))
    three = dict(zip(*enc.encode_document("alpha alpha alpha")))
    w1, w3 = next(iter(one.values())), next(iter(three.values()))
    assert w1 < w3 < w1 * 3  # more occurrences count, with diminishing returns
    ids, values = enc.encode_query("alpha alpha beta")
    assert len(ids) == 2 and values == [1.0, 1.0]


# ---- index -----------------------------------------------------------------------------------------
def test_collection_has_dense_and_sparse_vectors_and_is_idempotent(index, chunks):
    params = index.client.get_collection(index.collection).config.params
    assert "dense" in params.vectors and "bm25" in params.sparse_vectors
    assert params.sparse_vectors["bm25"].modifier == qm.Modifier.IDF
    again = index.build(chunks)
    assert again["recreated"] is False and again["points"] == len(chunks) and index.source_format() == "pdf"


def test_an_m3_collection_is_recreated_with_named_vectors(chunks):
    ix = PolicyIndex.open(E.HashingEmbedder(), ":memory:")
    ix.client.create_collection(ix.collection, vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE))
    out = ix.build(chunks[:5])
    assert out["recreated"] is True and ix.count() == 5
    ix.close()


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_respects_the_specialty_scope(index, mode):
    for s in SPECIALTY_BY_SLUG.values():
        hits = index.search("new-patient consultation length and list price", limit=8, specialty=s, mode=mode)
        assert hits and all(h.specialty in (None, s) for h in hits), (mode, s)


def test_sparse_and_hybrid_find_an_exact_policy_number(index):
    for mode in ("sparse", "hybrid"):
        hits = index.search("OP-CS-014", limit=3, mode=mode)
        assert hits and hits[0].doc_id == "OP-CS-014", mode


def test_hybrid_fuses_both_rankings(index):
    q = "pre-authorisation approval window for DERM-EXCI under CIVIC-BASE"
    dense = [h.chunk_id for h in index.search(q, limit=20, mode="dense")]
    sparse = [h.chunk_id for h in index.search(q, limit=20, mode="sparse")]
    hybrid = [h.chunk_id for h in index.search(q, limit=20, mode="hybrid")]
    assert hybrid and set(hybrid) <= set(dense) | set(sparse)  # fusion only re-ranks the two candidate lists
    # Reciprocal Rank Fusion: the fused winner is ranked near the top by at least one of the two.
    assert hybrid[0] in dense[:3] + sparse[:3]
    with pytest.raises(ValueError):
        index.search(q, mode="semantic")


def test_a_query_with_only_stop_words_still_searches_dense(index):
    assert index.search("what is it", limit=2, mode="hybrid")
    assert index.search("what is it", limit=2, mode="sparse") == []


# ---- retriever: rerank and abstain -------------------------------------------------------------
def test_reranker_reorders_candidates_and_reports_its_scores(index):
    rr = KeywordReranker()
    ret = PolicyRetriever(index, reranker=rr, candidates=10)
    hits = ret.search("Dermatology Consultation DERM-CONS list price", limit=3)
    assert rr.calls == 1 and len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(0 <= h.score <= 1 for h in hits)


def test_retriever_abstains_below_the_threshold(index):
    ret = PolicyRetriever(index, reranker=KeywordReranker(), min_score=0.6)
    out = ret.retrieve("valet parking zeppelin karaoke", limit=4)
    assert out.hits == [] and out.abstained and out.top_score < 0.6
    ok = ret.retrieve("Dermatology Consultation DERM-CONS list price", limit=4)
    assert ok.hits and not ok.abstained and ok.top_score >= 0.6
    assert ret.search("valet parking zeppelin karaoke") == []  # search() is retrieve().hits


def _hit(doc, score, n):
    from careflow.rag.index import Hit

    return Hit(score=score, doc_title=doc, doc_id=None, heading="h", specialty=None, text=f"{doc} {n}", chunk_id=f"{doc}{n}")


class ListIndex:
    """Index stub that returns fixed hits in retrieval order."""

    def __init__(self, hits):
        self.hits = hits

    def search(self, query, limit=4, specialty=None, mode="hybrid"):
        return self.hits[:limit]


def test_documents_are_spread_one_then_two_per_document():
    hits = [_hit("A", 0.9, 1), _hit("A", 0.8, 2), _hit("A", 0.7, 3), _hit("B", 0.6, 1), _hit("A", 0.5, 4)]
    capped = PolicyRetriever(ListIndex(hits), max_per_document=2).rank("q")
    assert [h.text for h in capped] == ["A 1", "B 1", "A 2", "A 3", "A 4"]  # best stays first; nothing dropped
    assert PolicyRetriever(ListIndex(hits), max_per_document=None).rank("q") == hits


class _Scores:
    name = "fixed"

    def __init__(self, table):
        self.table = table

    def score(self, query, passages):
        return [self.table[p] for p in passages]


def test_a_passage_that_ties_with_the_best_is_not_pushed_out_by_spreading():  # golden case EV-019
    # Three sections of one document tie at the top; other documents score near zero.
    hits = [_hit("S", 0, 1), _hit("S", 0, 2), _hit("S", 0, 3), _hit("H", 0, 1), _hit("M", 0, 1), _hit("I", 0, 1)]
    table = {"S 1": 0.9995, "S 2": 0.9990, "S 3": 0.9980, "H 1": 0.5, "M 1": 0.05, "I 1": 0.001}
    top4 = [h.text for h in PolicyRetriever(ListIndex(hits), reranker=_Scores(table)).rank("q")[:4]]
    assert top4 == ["S 1", "S 2", "S 3", "H 1"]
    # A passage that is merely good (not a tie) is still spread as before: one per document first.
    table["S 2"], table["S 3"] = 0.9, 0.8
    ranked = [h.text for h in PolicyRetriever(ListIndex(hits), reranker=_Scores(table)).rank("q")]
    assert ranked[:4] == ["S 1", "H 1", "M 1", "I 1"] and len(ranked) == 6


def test_fused_order_keeps_what_retrieval_ranked_first():
    # Retrieval ranked B first; the reranker prefers A1, A2, A3. Pure rerank order puts B last; fusion of
    # the two ranks keeps B in the top two. The shown score is the cross-encoder's either way.
    hits = [_hit("B", 0, 1), _hit("A", 0, 1), _hit("A", 0, 2), _hit("A", 0, 3)]

    class Prefers:
        name = "prefers-A"

        def score(self, query, passages):
            return [{"B 1": 0.1, "A 1": 0.9, "A 2": 0.8, "A 3": 0.7}[p] for p in passages]

    pure = PolicyRetriever(ListIndex(hits), reranker=Prefers(), max_per_document=None).rank("q")
    fused = PolicyRetriever(ListIndex(hits), reranker=Prefers(), max_per_document=None, fuse=True).rank("q")
    assert [h.text for h in pure] == ["A 1", "A 2", "A 3", "B 1"]
    assert "B 1" in [h.text for h in fused[:2]] and next(h.score for h in fused if h.text == "B 1") == 0.1


def test_without_a_reranker_the_retriever_never_abstains(index):
    ret = PolicyRetriever(index, reranker=None, min_score=0.99)
    out = ret.retrieve("valet parking zeppelin karaoke", limit=4)
    assert len(out.hits) == 4 and not out.abstained and out.top_score is None


def test_search_policies_says_when_the_documents_do_not_cover_a_question(index):
    tools = CareFlowTools(policy_index=PolicyRetriever(index, reranker=KeywordReranker(), min_score=0.6))
    r = tools.call("search_policies", {"query": "valet parking zeppelin karaoke"})
    assert r["ok"] and r["count"] == 0 and r["results"] == []
    assert "do not cover" in r["note"] and "flag_for_human" in r["note"]
    hit = tools.call("search_policies", {"query": "Dermatology Consultation DERM-CONS list price"})
    assert hit["count"] and "exactly" in hit["note"]


def test_retriever_is_configured_from_settings(tmp_path):
    ret = open_policy_search(Settings(careflow_embedding_model="local", careflow_reranker_model="none"), ":memory:")
    assert ret.reranker is None and ret.min_score is None and ret.count() == 0
    ret.close()
    ret = open_policy_search(Settings(careflow_embedding_model="local", careflow_rerank_min_score=0.3,
                                      careflow_retrieval_candidates=12), ":memory:")
    assert isinstance(ret.reranker, CrossEncoderReranker) and ret.min_score == 0.3 and ret.candidates == 12
    assert ret.reranker._model is None  # loaded lazily: building a retriever downloads nothing
    ret.close()
    assert default_reranker("NONE") is None and default_reranker("") is None


def test_cross_encoder_scores_are_probabilities(monkeypatch):
    rr = CrossEncoderReranker("fake/model")

    class FakeModel:
        def predict(self, pairs, **kw):
            assert kw["activation_fn"] is not None  # raw logits requested, our sigmoid applied
            return [5.0, 0.0, -5.0][: len(pairs)]

    monkeypatch.setattr(rr, "_load", lambda: FakeModel())
    s = rr.score("q", ["a", "b", "c"])
    assert s[0] > 0.99 and s[1] == 0.5 and s[2] < 0.01
    assert rr.score("q", []) == []


# ---- architect review (notes/feedback.md, M4) ---------------------------------------------------------
from careflow.rag.index import IndexOutOfDate  # noqa: E402
from careflow.rag.retriever import infer_specialty  # noqa: E402


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
def test_specialty_is_inferred_only_when_unambiguous(query, expected):  # M4-3
    assert infer_specialty(query) == expected


def test_unscoped_questions_get_at_most_one_handbook_up_front():  # M4-3
    hb = lambda s, n: Hit(score=0, doc_title=f"{s} Service Line Handbook", doc_id=None, heading="h",  # noqa: E731
                          specialty=s, text=f"{s} {n}", chunk_id=f"{s}{n}")
    from careflow.rag.index import Hit

    hits = [hb("Neurology", 1), hb("Orthopaedics", 1), hb("Cardiology", 1), _hit("Co-pay", 0, 1), _hit("Preauth", 0, 1)]
    unscoped = PolicyRetriever(ListIndex(hits), infer=False).rank("Does an MRI need pre-authorisation?")
    assert [h.text for h in unscoped[:3]] == ["Neurology 1", "Co-pay 1", "Preauth 1"]
    scoped = PolicyRetriever(ListIndex(hits)).rank("x", specialty="Neurology")
    assert [h.text for h in scoped[:2]] == ["Neurology 1", "Orthopaedics 1"]  # scoped: the family limit is off


def test_inferred_scope_reaches_the_index(index):  # M4-3
    ret = PolicyRetriever(index)
    out = ret.retrieve("What will a colonoscopy cost on MERIDIAN-SILVER?", limit=8)
    assert out.specialty == "Gastroenterology"
    assert all(h.specialty in (None, "Gastroenterology") for h in out.hits)


def test_an_m3_index_on_disk_is_reported_not_crashed(tmp_path):  # M4-4a
    ix = PolicyIndex.open(E.HashingEmbedder(), tmp_path / "q")
    ix.client.create_collection(ix.collection, vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE))
    ix.client.upsert(ix.collection, [qm.PointStruct(id=1, vector=[0.1] * 512, payload={"doc_title": "x"})])
    with pytest.raises(IndexOutOfDate):
        ix.search("referral")
    r = CareFlowTools(policy_index=PolicyRetriever(ix)).call("search_policies", {"query": "referral"})
    assert r["ok"] is False and "build_policy_index" in r["error"]
    ix.close()


def test_patient_identifiers_never_reach_the_search(index):  # M4-7b
    seen = []

    class Spy:
        def retrieve(self, query, limit=4, specialty=None):
            seen.append(query)
            return PolicyRetriever(index).retrieve(query, limit, specialty)

    tools = CareFlowTools(policy_index=Spy())
    r = tools.call("search_policies", {"query": "Does MHP-P-00001 (member MER90272292, born 26/12/1981, "
                                                "referral MHP-R-00012) need pre-authorisation?"})
    assert r["ok"] and r["redacted_identifiers"] == 4
    assert seen == ["Does (member , born , referral ) need pre-authorisation?"]
    only_ids = tools.call("search_policies", {"query": "MHP-P-00001"})
    assert only_ids["ok"] is False and len(seen) == 1


def test_search_inputs_are_bounded():  # M4-7b
    tools = CareFlowTools(policy_index=PolicyRetriever(ListIndex([])))
    for bad in ({"query": "x" * 501}, {"query": ""}, {"query": "x", "limit": 0}, {"query": "x", "limit": 9}):
        assert tools.call("search_policies", bad)["ok"] is False, bad
    assert tools.call("find_appointment_slots", {"specialty": "ENT", "limit": 21})["ok"] is False


def test_results_report_abstention_per_call(index):  # M4-8a
    tools = CareFlowTools(policy_index=PolicyRetriever(index, reranker=KeywordReranker(), min_score=0.6))
    assert tools.call("search_policies", {"query": "valet parking zeppelin karaoke"})["abstained"] is True
    assert tools.call("search_policies", {"query": "Dermatology Consultation DERM-CONS list price"})["abstained"] is False


def test_closing_the_index_closes_the_embedder():  # M4-6b
    closed = []
    emb = E.HashingEmbedder()
    emb.close = lambda: closed.append(True)
    PolicyIndex.open(emb, ":memory:").close()
    assert closed == [True]


def test_reranker_prefers_the_local_cache(monkeypatch):  # M4-7d
    import sys
    import types

    calls = []

    class FakeCE:
        def __init__(self, name, **kw):
            calls.append(kw.get("local_files_only", False))
            if kw.get("local_files_only"):
                raise OSError("not cached")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCE))
    CrossEncoderReranker("fake/model")._load()
    assert calls == [True, False]  # cache first, then one download
