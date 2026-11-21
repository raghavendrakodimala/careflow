"""M3 tools: find_patient (privacy-safe identity check, audit A4) and search_policies."""

import pytest

from careflow.rag.chunker import load_corpus_chunks
from careflow.rag.embeddings import HashingEmbedder
from careflow.rag.index import PolicyIndex
from careflow.tools import CareFlowTools


@pytest.fixture(scope="module")
def t():
    return CareFlowTools()


@pytest.fixture(scope="module")
def p(t):
    return t.patients["MHP-P-00000"]


def test_name_and_birth_date_give_only_the_patient_ref(t, p):
    r = t.find_patient(p["given_name"], p["family_name"], birth_date=p["birth_date"])
    assert r == {"ok": True, "found": True, "patient_ref": "MHP-P-00000"}


def test_formats_case_and_spacing_are_tolerated(t, p):
    y, m, d = p["birth_date"].split("-")
    r = t.find_patient(f"  {p['given_name'].upper()} ", p["family_name"].lower(), birth_date=f"{d}/{m}/{y}")
    assert r["found"] and r["patient_ref"] == p["patient_ref"]


def test_name_and_member_number(t, p):
    r = t.find_patient(p["given_name"], p["family_name"], member_number=p["member_number"].lower())
    assert r["found"] and r["patient_ref"] == p["patient_ref"]


def test_name_alone_is_not_enough(t, p):
    r = t.find_patient(p["given_name"], p["family_name"])
    assert r["ok"] is False and "patient_ref" not in r


def test_a_mismatch_never_says_which_detail_was_wrong(t, p):
    other = t.patients["MHP-P-00001"]
    wrong_dob = t.find_patient(p["given_name"], p["family_name"], birth_date="1900-01-01")
    wrong_name = t.find_patient("Nobody", "Atall", birth_date=p["birth_date"])
    wrong_member = t.find_patient(p["given_name"], p["family_name"], birth_date=p["birth_date"],
                                  member_number=other["member_number"])
    assert wrong_dob == wrong_name == wrong_member  # identical replies: nothing to probe
    assert wrong_dob["found"] is False and "patient_ref" not in wrong_dob


def test_unreadable_birth_date_is_an_error(t, p):
    assert t.find_patient(p["given_name"], p["family_name"], birth_date="last tuesday")["ok"] is False


def test_find_patient_through_the_contract(t, p):
    r = t.call("find_patient", {"given_name": p["given_name"], "family_name": p["family_name"],
                                "birth_date": p["birth_date"], "member_number": None})
    assert r["found"]
    assert "given_name" in t.call("find_patient", {"family_name": "x", "birth_date": "2000-01-01"})["error"]


def test_policy_search_is_offered_only_with_an_index(t):
    assert "search_policies" not in t.available_tools()
    assert t.search_policies("referral SLA")["ok"] is False


def test_policy_search_returns_titled_passages():
    ix = PolicyIndex.open(HashingEmbedder(), ":memory:")
    ix.build(load_corpus_chunks())
    tools = CareFlowTools(policy_index=ix)
    assert "search_policies" in tools.available_tools()
    r = tools.call("search_policies", {"query": "consultation length", "specialty": "Neurology", "limit": 3})
    assert r["ok"] and r["count"] == 3
    assert all(h["document"] for h in r["results"]) and "Answer only from" in r["note"]
    assert all("Cardiology" not in h["document"] for h in r["results"])
    assert tools.search_policies("x", specialty="Podiatry")["ok"] is False
    assert "specialty" in tools.call("search_policies", {"query": "x", "specialty": "Podiatry"})["error"]
    ix.close()
