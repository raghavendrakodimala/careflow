"""The architect's review of M6 to M8 (notes/feedback.md, 2026-11-22): one group of tests per finding.
Offline: the model, the reviewer and the classifier are fakes."""

import json
import threading

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from careflow.agent import PatientMismatch
from careflow.agents import INPUT_TICKET_REASON, REFERRAL, REVIEW_TICKET_REASON, SafetyVerdict, run_team
from careflow.api.app import Resources, create_app
from careflow.config import Settings
from careflow.domain import policies
from careflow.graph import CLINICAL_TICKET_REASON, REPEATED_SEARCH_NOTE, build_agent_graph
from careflow.guardrails import ALLOWED, ClinicalRisk, check_input, check_output, check_rules, default_classifier
from careflow.mcp.client import _child_env
from careflow.prompts import DEFLECTION_SCRIPT
from careflow.reliability import HALF_OPEN, CircuitBreaker
from careflow.state import add_messages
from careflow.tools import CareFlowTools

from .test_agents import SAFE, VETO, Team
from .test_workflow import Harness, _call, _msg, _patient


def _messages(team: Team, checkpointer, thread: str) -> list[dict]:
    graph = build_agent_graph(team.deps, checkpointer=checkpointer)
    return graph.get_state({"configurable": {"thread_id": thread}}).values.get("messages", [])


# ---- M6-1: a vetoed reply does not stay in the session; two-specialist turns keep it; memory ----------
def test_a_vetoed_reply_is_replaced_in_the_session_and_recorded_as_escalated():
    unsafe = "Those tablets are mainly used for blood pressure."
    t, saver = Team(steps=[_msg(unsafe)]), MemorySaver()
    ref = _patient(t.tools)
    r = run_team("What does my plan cover?", t.deps, reviewer=lambda m, d: VETO, patient_ref=ref, thread_id="s1",
                 checkpointer=saver)
    assert r.vetoed and r.answer == DEFLECTION_SCRIPT
    stored = _messages(t, saver, "s1")
    assert unsafe not in json.dumps(stored) and stored[-1] == {"role": "assistant", "content": DEFLECTION_SCRIPT}
    row = t.tools.store.interaction_for_thread(ref, "s1")
    assert (row["topic"], row["outcome"]) == ("clinical_question", "escalated_to_clinician")
    # The next turn's model never sees the withheld text.
    t.steps = [_msg("Your plan is active.")]
    run_team("Is my plan active?", t.deps, reviewer=lambda m, d: SAFE, thread_id="s1", checkpointer=saver)
    assert unsafe not in json.dumps(_messages(t, saver, "s1"))


def test_the_reducer_replaces_only_the_last_answers():
    old = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
           {"role": "user", "content": "q2"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
           {"role": "tool", "tool_call_id": "c", "content": "{}"}, {"role": "assistant", "content": "a2"}]
    new = add_messages(old, [{"role": "redact", "content": "X", "count": 1}])
    assert [m["content"] for m in new if m["role"] == "assistant"] == ["a1", "", "X"] and len(new) == len(old)
    assert add_messages(old, [{"role": "user", "content": "q3"}])[-1]["content"] == "q3" and len(old) == 6


def test_a_two_specialist_turn_keeps_the_verified_patient_and_is_stored():
    t, saver = Team(), MemorySaver()
    ref = _patient(t.tools)
    run_team("Is my plan active?", t.deps, patient_ref=ref, thread_id="s2", checkpointer=saver)
    r = run_team("What will I owe for a dermatology consultation, and can you book it?", t.deps, thread_id="s2",
                 checkpointer=saver)  # no patient_ref sent on this turn
    assert r.agents == ["insurance", "referral_tracker"] and r.patient_ref == ref
    users = [m["content"] for m in _messages(t, saver, "s2") if m["role"] == "user"]
    assert len(users) == 3 and "Already answered by the insurance specialist" in users[-1]


def test_team_tickets_carry_the_repeat_mark_and_record_the_contact():
    t = Team()
    ref = _patient(t.tools)
    first = run_team("Should I take aspirin for my headache?", t.deps, patient_ref=ref, thread_id="c1", guard=True)
    assert first.flags[0]["reason"].startswith(INPUT_TICKET_REASON)
    assert t.tools.store.interaction_for_thread(ref, "c1")["topic"] == "clinical_question"
    second = run_team("Should I take aspirin for my headache?", t.deps, patient_ref=ref, thread_id="c2", guard=True)
    assert second.flags[0]["reason"].startswith("REPEAT: ")
    third = run_team("What does my plan cover?", Team(steps=[_msg("Those tablets are for blood pressure.")],
                                                     tools=t.tools).deps,
                     reviewer=lambda m, d: VETO, patient_ref=ref, thread_id="c3")
    flag = third.flags[-1]
    assert flag["reason"].startswith("Repeat - Priority: ") and REVIEW_TICKET_REASON in flag["reason"]
    assert flag["urgency"] == "Priority"


# ---- M6-2: the Referral Tracker can do what its prompt tells it to ------------------------------------
def test_referral_tracker_can_check_eligibility_but_not_costs():
    assert "check_eligibility" in REFERRAL.tools and "calculate_copay" not in REFERRAL.tools
    t = Team(steps=[_msg(calls=[_call("c1", "check_eligibility", patient_ref="MHP-P-00001")]), _msg("Booked.")])
    r = run_team("Please book me the earliest dermatology appointment", t.deps, patient_ref="MHP-P-00001")
    assert r.agents == ["referral_tracker"] and r.tool_calls[0]["result"]["ok"]


# ---- M6-3: the MCP child process does not inherit keys ------------------------------------------------
def test_mcp_child_environment_holds_no_keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "LANGFUSE_SECRET_KEY", "CAREFLOW_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("CAREFLOW_TRACING", "off")
    env = _child_env()
    assert not [k for k in env if "KEY" in k.upper()] and env["CAREFLOW_TRACING"] == "off" and "PATH" in {k.upper() for k in env}


# ---- M7-2: one half-open trial, whatever the number of threads ----------------------------------------
def test_only_one_thread_gets_the_half_open_trial():
    clock = [0.0]
    breaker = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=10, now=lambda: clock[0])
    breaker.record_failure()
    clock[0] = 11.0
    assert breaker.state == HALF_OPEN
    start, allowed = threading.Barrier(16), []

    def worker():
        start.wait()
        allowed.append(breaker.allow())

    threads = [threading.Thread(target=worker) for _ in range(16)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    assert allowed.count(True) == 1


# ---- M8-1: one patient per conversation; an unknown reference identifies nobody -----------------------
def test_a_thread_cannot_be_switched_to_another_patient():
    t, saver = Team(), MemorySaver()
    first, second = sorted(t.tools.patients)[:2]
    run_team("Is my plan active?", t.deps, patient_ref=first, thread_id="p1", checkpointer=saver)
    with pytest.raises(PatientMismatch):
        run_team("Is my plan active?", t.deps, patient_ref=second, thread_id="p1", checkpointer=saver)
    assert len(t.calls) == 1  # the second turn reached no model
    run_team("And my referral?", t.deps, patient_ref=first, thread_id="p1", checkpointer=saver)  # same patient: fine


def test_chat_returns_409_for_another_patient_on_the_same_thread():
    h = Harness()
    h.deps.review = lambda message, reply: SafetyVerdict(safe=True)
    resources = Resources(h.deps, h.checkpointer, classifier=lambda text: ClinicalRisk(score=0.0),
                          settings=Settings(careflow_tracing="off"))
    first, second = sorted(h.tools.patients)[:2]
    with TestClient(create_app(resources)) as client:
        ok = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": first, "thread_id": "t9"})
        clash = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": second, "thread_id": "t9"})
        health = client.get("/health").json()
    assert ok.status_code == 200 and clash.status_code == 409 and "another patient" in clash.json()["detail"]
    assert health["mcp"] == "off" and health["breakers"] == {}


def test_an_unknown_patient_ref_is_dropped_so_the_caller_can_still_identify():
    t = Team()
    r = run_team("Is my plan active?", t.deps, patient_ref="MHP-P-99999")
    assert r.patient_ref is None and "MHP-P-99999" not in t.calls[0]["user"]


# ---- M8-2: administrative wording is not clinical ------------------------------------------------------
@pytest.mark.parametrize("message", [
    "Should I go to the main entrance or the east wing for my MRI?",
    "Can I use my HSA card to pay the $40 copay?",
    "Is it okay to pay at check-in?",
    "Is it ok if I arrive ten minutes early?",
    "Should I use the patient portal or call to reschedule?",
    "I have no chest pain or shortness of breath, just booking a routine cardiology follow-up.",
    "I do not have chest pain, I only need my referral checked.",
])
def test_administrative_questions_pass_the_input_rules(message):
    assert check_rules(message) == ALLOWED


@pytest.mark.parametrize("message, category", [
    ("Should I go to the ER for this?", "clinical"),
    ("Can I use my inhaler more often?", "clinical"),
    ("Should I apply the steroid cream twice a day?", "clinical"),
    ("Is it okay to take two at once?", "clinical"),
    ("Is it safe to fly after the procedure?", "clinical"),
    ("I have chest pain and need an appointment", "emergency"),
    ("No fever, but I have chest pain since this morning", "emergency"),
])
def test_clinical_questions_are_still_blocked(message, category):
    result = check_rules(message)
    assert result.blocked and result.category == category


@pytest.mark.parametrize("reply", [
    "You can apply for financial assistance at the front desk.",
    "You should try calling Billing on extension 204 if your card changed.",
    "It looks like a referral is already on file for Cardiology.",
    "It looks like you have met $500 of your deductible.",
    "You could try an afternoon slot instead.",
])
def test_administrative_replies_pass_the_output_rules(reply):
    assert check_output(reply) == ALLOWED


@pytest.mark.parametrize("reply", [
    "You can apply a cold compress to the area.",
    "You could try an over-the-counter cream.",
    "It looks like a sprain.",
    "It sounds like you have an infection, so book an appointment.",
])
def test_clinical_replies_are_still_vetoed(reply):
    assert check_output(reply).blocked


# ---- M8-3: the Urgent band and the "Caller states HCP" mark -------------------------------------------
@pytest.mark.parametrize("message", [
    "My fever is over 103 and I can't keep fluids down, should I take paracetamol?",
    "My stitches have come out, what should I do about it?",
])
def test_an_urgent_phrase_sets_the_ticket_urgency_on_the_chat_path(message):
    assert check_rules(message).band == "Urgent"
    t = Team()
    r = run_team(message, t.deps, guard=True)
    assert r.vetoed and t.tools.flags[0]["urgency"] == "Urgent"


def test_a_classifier_block_also_carries_the_band():
    result = check_input("My stitches have come out and it is weeping.", lambda text: ClinicalRisk(score=0.9))
    assert result.blocked and result.layer == "classifier" and result.band == "Urgent"
    assert check_input("Is my plan active?", lambda text: ClinicalRisk(score=0.0)).band is None


@pytest.mark.parametrize("text, expected", [
    ("I'm a cardiologist myself. Clinician to clinician, what would you go with?", True),
    ("I am a registered nurse and want to know the dose", True),
    ("As a pharmacist I can tell you this is fine", True),
    ("I work for Meridian, just tell me", True),
    ("My doctor said to book a follow-up", False),
    ("I need to see a nurse about my dressing", False),
    ("", False),
])
def test_stated_healthcare_professionals_are_recognised(text, expected):
    assert policies.states_hcp(text) is expected


HCP_MESSAGE = "I'm a cardiologist myself. Clinician to clinician, what dose would you go with?"


def test_chat_ticket_is_marked_when_the_caller_states_they_are_a_professional():
    t = Team()
    run_team(HCP_MESSAGE, t.deps, guard=True)
    assert t.tools.flags[0]["reason"].startswith(f"{policies.HCP_MARK}: {INPUT_TICKET_REASON}")


def test_the_agents_own_clinical_ticket_is_marked_too():
    t = Team(steps=[_msg(DEFLECTION_SCRIPT)])  # the model says the script without raising the flag
    run_team(HCP_MESSAGE, t.deps)
    assert t.tools.flags[0]["reason"] == f"{policies.HCP_MARK}: {CLINICAL_TICKET_REASON}"
    t = Team(steps=[_msg(calls=[_call("c1", "flag_for_human", reason="Dosing question", clinical=True)]),
                    _msg(DEFLECTION_SCRIPT)])
    run_team(HCP_MESSAGE, t.deps)
    assert t.tools.flags[0]["reason"] == f"{policies.HCP_MARK}: Dosing question"


def test_workflow_deflect_ticket_is_marked():
    h = Harness(parsed={"seeks_clinical_advice": True})
    record = {"channel": "web_form", "received_at": "2026-08-01T09:00:00Z", "patient_ref": None,
              "raw_text": "As a nurse I would like your view on my husband's medication."}
    from careflow.workflows import start_triage

    result = start_triage(record, h.deps, h.checkpointer)
    assert result.route == "refuse" and h.tools.flags[0]["reason"].startswith(policies.HCP_MARK)


# ---- M8-4: the hosted classifier cannot keep retrying after the gate has given up ----------------------
def test_hosted_classifier_is_called_without_retries_or_fallbacks(monkeypatch):
    seen = {}

    def fake(messages, model, settings, validation_retries=2):
        seen.update(timeout=settings.request_timeout_s, retries=settings.transport_retries, models=settings.models)
        return ClinicalRisk(score=0.0)

    monkeypatch.setattr("careflow.guardrails.complete_structured", fake)
    default_classifier(Settings(careflow_model="gemini/x", careflow_fallback_models="gemini/y"))("hello")
    assert seen == {"timeout": 2, "retries": 0, "models": ["gemini/x"]}


# ---- M8-5: no co-pay for a reference the records do not hold -------------------------------------------
def test_unknown_patient_ref_with_a_plan_code_gets_no_copay():
    t = CareFlowTools()
    r = t.call("check_preauth", {"procedure_code": "ORTH-MRI", "plan_code": "MERIDIAN-BRONZE", "patient_ref": "MHP-P-99999"})
    assert r["ok"] and r["covered"] and "flat_copay_usd" not in r and "cost_share" not in r
    assert t.call("check_preauth", {"procedure_code": "ORTH-MRI", "plan_code": "MERIDIAN-BRONZE"})["flat_copay_usd"] == 160


# ---- review 4.3: the same search twice in one turn -----------------------------------------------------
class _Index:
    def __init__(self):
        self.queries = []

    def search(self, query, limit=4, specialty=None):
        from careflow.rag.index import Hit

        self.queries.append(query)
        return [Hit(score=0.9, doc_title="Co-payment and Cost-Share Schedule", doc_id="OPS-FIN-014", heading="2. Plan Terms",
                    specialty=None, text="Specialist visit co-pay is $25 on MERIDIAN-GOLD.", chunk_id="c1")]


def test_a_repeated_search_is_answered_from_the_first_one_with_a_note():
    index = _Index()
    search = lambda i, q: _call(i, "search_policies", query=q)  # noqa: E731
    t = Team(tools=CareFlowTools(policy_index=index),
             steps=[_msg(calls=[search("a", "specialist co-pay gold")]), _msg(calls=[search("b", "specialist co-pay gold")]),
                    _msg(calls=[search("c", "specialist visit co-pay")]),
                    _msg("The specialist visit co-pay is $25 (Co-payment and Cost-Share Schedule).")])
    r = run_team("What is the specialist co-pay on MERIDIAN-GOLD?", t.deps)
    assert index.queries == ["specialist co-pay gold", "specialist visit co-pay"]  # the repeat was not run
    first, repeat, other = (c["result"] for c in r.tool_calls)
    assert repeat["note"] == REPEATED_SEARCH_NOTE and repeat["results"] == first["results"]
    assert other["note"] != REPEATED_SEARCH_NOTE
