"""The HTTP service, through FastAPI's TestClient. The intake parser, the model, the reviewer and the
classifier are fakes; the store is in memory, so no key, index or child process is needed."""

import pytest
from fastapi.testclient import TestClient

from src.agents import SafetyVerdict
from src.api.app import Resources, create_app
from src.config import Settings
from src.guardrails import ClinicalRisk
from src.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT

from .test_workflow import Harness, _find_patient, _make_call, _make_message


def _open_client(harness: Harness, **settings) -> TestClient:
    harness.dependencies.review = lambda message, reply: SafetyVerdict(safe=True)
    resources = Resources(harness.dependencies, harness.checkpointer, classifier=lambda text: ClinicalRisk(score=0.0),
                          settings=Settings(careflow_tracing="off", **settings))
    return TestClient(create_app(resources))


@pytest.fixture()
def harness():
    return Harness()


def test_health(harness):
    with _open_client(harness) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "patients": 400, "pending_reviews": 0, "policy_search": False, "tracing": False,
                    "guard_classifier": "hosted",  # the test's fake is a plain callable, like the hosted one
                    "mcp": "off", "breakers": {}}


def test_triage_handles_a_routine_request_automatically(harness):
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        response = client.post("/v1/triage", json={"raw_text": "I'd like a dermatology appointment.", "patient_ref": ref,
                                             "channel": "sms", "thread_id": "case-1"})
    body = response.json()
    assert response.status_code == 200 and (body["route"], body["status"], body["thread_id"]) == ("auto", "completed", "case-1")
    assert body["steps"][0] == "guard" and body["steps"][-1] == "finalize" and "audit" not in body
    assert body["response"] == "Here is what I found." and body["patient_ref"] == ref


def test_malformed_bodies_are_rejected_with_422(harness):
    with _open_client(harness) as client:
        for bad in ({}, {"raw_text": ""}, {"raw_text": "x", "channel": "carrier pigeon"}, {"raw_text": "x", "extra": 1},
                    {"raw_text": "x", "thread_id": "has spaces"}, {"raw_text": "x" * 8001}):
            assert client.post("/v1/triage", json=bad).status_code == 422, bad
        assert client.post("/v1/chat", json={"message": ""}).status_code == 422
        assert client.post("/v1/triage", content="not json", headers={"content-type": "application/json"}).status_code == 422
    assert harness.tools.flags == []


def test_a_clinical_message_gets_the_refuse_payload(harness):
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        body = client.post("/v1/triage", json={"raw_text": "Should I take aspirin for this headache?",
                                               "patient_ref": ref}).json()
    assert (body["route"], body["status"]) == ("refuse", "awaiting_human_review")
    assert body["response"] == DEFLECTION_SCRIPT and body["guard"]["blocked"] is True
    assert body["flag_id"] and body["review_id"] and harness.offered == []


def test_review_queue_fetch_decide_and_complete_over_http():
    harness = Harness(parsed={"urgency": "Urgent"})
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        started = client.post("/v1/triage", json={"raw_text": "I need to be seen soon.", "patient_ref": ref,
                                                  "thread_id": "case-9"}).json()
        assert (started["route"], started["status"]) == ("human_review", "awaiting_human_review")

        queue = client.get("/v1/reviews").json()
        assert queue["count"] == 1 and queue["reviews"][0]["thread_id"] == "case-9"
        payload = client.get("/v1/reviews/case-9").json()
        assert payload["review_id"] == started["review_id"] and payload["band"] == "Urgent"

        url = "/v1/reviews/case-9/decision"
        assert client.post(url, json={"action": "approve"}).status_code == 422  # no reviewer
        assert client.post(url, json={"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine"}).status_code == 422
        assert client.get("/v1/reviews").json()["count"] == 1  # still waiting

        done = client.post(url, json={"reviewer_id": "RN-0042", "action": "approve", "note": "Called."})
        assert done.status_code == 200 and done.json()["status"] == "completed"
        assert done.json()["decision"]["reviewer_id"] == "RN-0042"
        assert client.get("/v1/reviews").json()["count"] == 0
        assert client.get("/v1/reviews/case-9").status_code == 404
        assert client.post(url, json={"reviewer_id": "RN-0042", "action": "approve"}).status_code == 409
    assert harness.tools.flags[0]["status"] == "reviewed_approve"


def test_a_thread_id_cannot_be_reused(harness):
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        body = {"raw_text": "I'd like a dermatology appointment.", "patient_ref": ref, "thread_id": "case-2"}
        assert client.post("/v1/triage", json=body).status_code == 200
        again = client.post("/v1/triage", json=body)
    assert again.status_code == 409 and "case-2" in again.json()["detail"]


def test_decision_on_an_unknown_thread_is_409(harness):
    with _open_client(harness) as client:
        response = client.post("/v1/reviews/nobody/decision", json={"reviewer_id": "RN-1", "action": "approve"})
    assert response.status_code == 409


def test_chat_answers_through_the_team_and_reports_the_route():
    ref_holder = Harness(steps=[_make_message(calls=[_make_call("c1", "check_eligibility", patient_ref="MHP-P-00001")]),
                                _make_message("Your plan is active.")])
    with _open_client(ref_holder) as client:
        body = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": "MHP-P-00001"}).json()
    assert body["answer"] == "Your plan is active." and body["route"] == "auto"
    assert body["agents"] == ["insurance"] and body["tools_used"] == ["check_eligibility"]
    assert body["guard"]["blocked"] is False and body["thread_id"]


def test_chat_blocks_an_emergency_before_any_agent(harness):
    with _open_client(harness) as client:
        body = client.post("/v1/chat", json={"message": "I have chest pain, what should I do?"}).json()
    assert body["route"] == "refuse" and body["answer"].startswith(EMERGENCY_STATEMENT) and body["agents"] == []
    assert harness.offered == [] and harness.tools.flags[0]["urgency"] == "Emergent"


def test_chat_fails_closed_when_the_classifier_does_not_answer(harness):
    def raise_timeout(text):
        raise TimeoutError

    resources = Resources(harness.dependencies, harness.checkpointer, classifier=raise_timeout, settings=Settings(careflow_tracing="off"))
    with TestClient(create_app(resources)) as client:
        body = client.post("/v1/chat", json={"message": "Is my plan active?"}).json()
    assert body["route"] == "refuse" and body["answer"] == DEFLECTION_SCRIPT and body["guard"]["layer"] == "fail_closed"


def test_chat_says_so_when_the_model_is_down(harness):
    harness.dependencies.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
    with _open_client(harness) as client:
        response = client.post("/v1/chat", json={"message": "Is my plan active?"})
    assert response.status_code == 503 and "cannot answer right now" in response.json()["detail"]


def test_api_key_is_required_when_configured(harness):
    with _open_client(harness, careflow_api_key="s3cret") as client:
        assert client.get("/health").status_code == 200  # health stays open for probes
        assert client.get("/v1/reviews").status_code == 401
        assert client.post("/v1/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/reviews", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_the_app_does_not_close_resources_it_was_given(harness):
    with _open_client(harness):
        pass
    assert harness.tools.store.list_rows("patients")  # still open
