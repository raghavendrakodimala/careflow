"""Patient lookup and policy search tools: find_patient (privacy-safe identity check) and search_policies."""

import pytest

from src.rag.chunker import load_corpus_chunks
from src.rag.embeddings import HashingEmbedder
from src.rag.index import PolicyIndex
from src.tools import CareFlowTools


@pytest.fixture(scope="module")
def tools():
    return CareFlowTools()


@pytest.fixture(scope="module")
def patient(tools):
    return tools.patients["MHP-P-00000"]


def test_name_and_birth_date_give_only_the_patient_ref(tools, patient):
    result = tools.find_patient(patient["given_name"], patient["family_name"], birth_date=patient["birth_date"])
    assert result == {"ok": True, "found": True, "patient_ref": "MHP-P-00000"}


def test_formats_case_and_spacing_are_tolerated(tools, patient):
    year, month, day = patient["birth_date"].split("-")
    result = tools.find_patient(f"  {patient['given_name'].upper()} ", patient["family_name"].lower(), birth_date=f"{day}/{month}/{year}")
    assert result["found"] and result["patient_ref"] == patient["patient_ref"]


def test_name_and_member_number(tools, patient):
    result = tools.find_patient(patient["given_name"], patient["family_name"], member_number=patient["member_number"].lower())
    assert result["found"] and result["patient_ref"] == patient["patient_ref"]


def test_name_alone_is_not_enough(tools, patient):
    result = tools.find_patient(patient["given_name"], patient["family_name"])
    assert result["ok"] is False and "patient_ref" not in result


def test_a_mismatch_never_says_which_detail_was_wrong(tools, patient):
    other = tools.patients["MHP-P-00001"]
    wrong_dob = tools.find_patient(patient["given_name"], patient["family_name"], birth_date="1900-01-01")
    wrong_name = tools.find_patient("Nobody", "Atall", birth_date=patient["birth_date"])
    wrong_member = tools.find_patient(patient["given_name"], patient["family_name"], birth_date=patient["birth_date"],
                                  member_number=other["member_number"])
    assert wrong_dob == wrong_name == wrong_member  # identical replies: nothing to probe
    assert wrong_dob["found"] is False and "patient_ref" not in wrong_dob


def test_unreadable_birth_date_is_an_error(tools, patient):
    assert tools.find_patient(patient["given_name"], patient["family_name"], birth_date="last tuesday")["ok"] is False


def test_find_patient_through_the_contract(tools, patient):
    result = tools.call("find_patient", {"given_name": patient["given_name"], "family_name": patient["family_name"],
                                "birth_date": patient["birth_date"], "member_number": None})
    assert result["found"]
    assert "given_name" in tools.call("find_patient", {"family_name": "x", "birth_date": "2000-01-01"})["error"]


def test_policy_search_is_offered_only_with_an_index(tools):
    assert "search_policies" not in tools.list_available_tools()
    assert tools.search_policies("referral SLA")["ok"] is False


def test_policy_search_returns_titled_passages():
    index = PolicyIndex.open(HashingEmbedder(), ":memory:")
    index.build(load_corpus_chunks())
    indexed_tools = CareFlowTools(policy_index=index)
    assert "search_policies" in indexed_tools.list_available_tools()
    result = indexed_tools.call("search_policies", {"query": "consultation length", "specialty": "Neurology", "limit": 3})
    assert result["ok"] and result["count"] == 3
    assert all(passage["document"] for passage in result["results"]) and "Answer only from" in result["note"]
    assert all("Cardiology" not in passage["document"] for passage in result["results"])
    assert indexed_tools.search_policies("x", specialty="Podiatry")["ok"] is False
    assert "specialty" in indexed_tools.call("search_policies", {"query": "x", "specialty": "Podiatry"})["error"]
    index.close()
