"""Offline tests for the triage workflow: fixed step order, routes, the human-review pause and
resuming from a SQLite checkpoint. The intake parser and the model are fakes; stores are in memory or tmp_path."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from src.agent import close_checkpointer, open_session_checkpointer
from src.graph import AgentDependencies
from src.memory import PatientMemory
from src.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from src.schemas import PatientRequest
from src.store import RuntimeStore
from src.tools import CareFlowTools
from src.workflows import (
    HANDOFF_ANSWER,
    NODES,
    NotAwaitingReviewError,
    TriageDependencies,
    build_triage_graph,
    get_pending_review,
    resume_triage,
    start_triage,
)

APPROVE = {"reviewer_id": "RN-0042", "action": "approve"}


def _make_call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _make_message(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls or [])


class Harness:
    """A workflow with a scripted intake parse and a scripted model, recording what the model was offered."""

    def __init__(self, tools=None, parsed=None, steps=None, parse_error=None):
        self.tools = tools or CareFlowTools()
        self.parsed = {"urgency": "Routine", "seeks_clinical_advice": False, "specialty": "Dermatology", **(parsed or {})}
        self.steps = list(steps or [_make_message("Here is what I found.")])
        self.parse_error = parse_error
        self.offered: list[set[str]] = []
        self.checkpointer = MemorySaver()
        self.dependencies = TriageDependencies(
            agent=AgentDependencies(tools=self.tools, model=self._call_model, memory=PatientMemory(self.tools.store, self.tools.clock)),
            parse=self._parse,
        )

    def _parse(self, record):
        if self.parse_error:
            raise self.parse_error
        return PatientRequest(**self.parsed)

    def _call_model(self, messages, specs):
        self.offered.append({spec["function"]["name"] for spec in specs})
        return self.steps[min(len(self.offered), len(self.steps)) - 1]

    def start(self, patient_ref=None, **kw):
        record = {"channel": "web_form", "received_at": "2026-08-01T09:00:00Z", "raw_text": "I need an appointment.",
                  "patient_ref": patient_ref, "ground_truth": {"urgency": "Emergent"}}
        return start_triage(record, self.dependencies, self.checkpointer, **kw)

    def resume(self, thread_id, decision=APPROVE):
        return resume_triage(thread_id, decision, self.dependencies, self.checkpointer)


def _find_patient(tools, status="active", specialty=None):
    """A patient with the given coverage and no live referral (optionally: none to `specialty` only)."""
    for ref, patient in tools.patients.items():
        live = [referral for referral in tools.store.list_rows_for_patient("referrals", ref) if referral["status"] in ("open", "accepted", "scheduled")]
        if patient["eligibility_status"] == status and not [referral for referral in live if specialty in (None, referral["to_specialty"])]:
            return ref
    raise AssertionError("no such patient in the seed data")


def _find_referral(tools, **want):
    """(patient_ref, referral row) for a live referral whose details match `want`."""
    details = tools.referral_details
    for referral in tools.store.list_rows("referrals"):
        row = {**referral, **details[referral["referral_id"]]}
        if all(row[key] == value for key, value in want.items()):
            return referral["patient_ref"], row
    raise AssertionError(f"no referral with {want}")


# ---- shape and order ---------------------------------------------------------------------------------
def test_graph_has_the_planned_nodes():
    harness = Harness()
    graph = build_triage_graph(harness.dependencies, harness.checkpointer).get_graph()
    assert set(NODES) <= set(graph.nodes)
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert {("guard", "intake"), ("intake", "hydrate"), ("hydrate", "deflect"), ("hydrate", "eligibility"), ("eligibility", "routing"),
            ("routing", "team"), ("team", "safety_review"), ("safety_review", "decide"), ("deflect", "decide"), ("decide", "review"),
            ("decide", "finalize"), ("review", "finalize")} <= edges


def test_auto_request_runs_the_steps_in_order_and_completes():
    harness = Harness()
    result = harness.start(_find_patient(harness.tools))
    assert result.steps == ["guard", "intake", "hydrate", "eligibility", "routing", "team", "safety_review", "decide", "finalize"]
    assert (result.status, result.route, result.reasons) == ("completed", "auto", [])
    assert result.response == "Here is what I found."
    assert result.review_id is None and harness.tools.flags == [] and harness.tools.store.list_pending_reviews() == []
    assert "book_appointment" in harness.offered[0]  # nothing to wait for, so the assistant may book


def test_labels_never_enter_the_workflow_state():
    harness = Harness()
    result = harness.start(_find_patient(harness.tools))
    state = build_triage_graph(harness.dependencies, harness.checkpointer).get_state({"configurable": {"thread_id": result.thread_id}})
    assert "ground_truth" not in state.values["record"]


# ---- refuse: act, then pause -------------------------------------------------------------------------
def test_clinical_question_gets_the_script_and_a_ticket_at_once_then_waits_for_staff():
    harness = Harness(parsed={"seeks_clinical_advice": True})
    ref = _find_patient(harness.tools)
    result = harness.start(ref)
    assert result.steps == ["guard", "intake", "hydrate", "deflect", "decide"]
    assert (result.status, result.route) == ("awaiting_human_review", "refuse")
    assert result.response == DEFLECTION_SCRIPT
    assert harness.offered == []  # the model never saw the message
    (flag,) = harness.tools.flags
    assert (flag["category"], flag["patient_ref"], flag["status"]) == ("clinical", ref, "awaiting_staff")
    assert result.eligibility == {}  # insurance is not raised before the clinical response (SOP 5.1)
    (contact,) = [interaction for interaction in harness.tools.store.list_rows_for_patient("patient_interactions", ref) if interaction["thread_id"] == result.thread_id]
    assert (contact["topic"], contact["outcome"]) == ("clinical_question", "escalated_to_clinician")


def test_emergent_request_starts_with_the_emergency_statement_and_is_never_booked():
    harness = Harness(parsed={"urgency": "Emergent", "specialty": "Cardiology"})
    result = harness.start(_find_patient(harness.tools))
    assert result.route == "refuse" and result.response.startswith(EMERGENCY_STATEMENT) and DEFLECTION_SCRIPT in result.response
    assert harness.tools.flags[0]["urgency"] == "Emergent" and harness.tools.flags[0]["category"] == "clinical"
    assert harness.offered == [] and harness.tools.booked == {}
    assert "referral policy 3.3" in result.reasons[0]


def test_unreadable_message_fails_closed():
    harness = Harness(parse_error=RuntimeError("model down"))
    result = harness.start(_find_patient(harness.tools))
    assert (result.status, result.route) == ("awaiting_human_review", "refuse")
    assert result.response == DEFLECTION_SCRIPT and "9.3" in result.reasons[0]
    assert harness.tools.flags[0]["category"] == "clinical"


def test_a_clinical_question_the_parser_missed_is_still_a_refusal():
    # The assistant's own rule catches it: it flags a clinician and gives the script.
    steps = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Asked about a dose", urgency="Routine", clinical=True)]),
             _make_message(DEFLECTION_SCRIPT)]
    harness = Harness(steps=steps)
    result = harness.start(_find_patient(harness.tools))
    assert (result.status, result.route) == ("awaiting_human_review", "refuse")
    assert len(harness.tools.flags) == 1 and result.flag_id == harness.tools.flags[0]["flag_id"]


# ---- human review ------------------------------------------------------------------------------------
def test_review_needed_withholds_booking_and_pauses():
    harness = Harness(parsed={"specialty": "Cardiology"})
    ref = _find_patient(harness.tools, specialty="Cardiology")
    result = harness.start(ref)
    assert (result.status, result.route) == ("awaiting_human_review", "human_review")
    assert "sign-off" in result.reasons[0] and result.plan["sign_off_required"] is True and result.plan["referral"] is None
    assert harness.offered and all("book_appointment" not in names for names in harness.offered)
    (flag,) = harness.tools.flags
    assert (flag["category"], flag["status"]) == ("administrative", "awaiting_staff")
    assert harness.tools.booked == {}
    (pending,) = harness.tools.store.list_pending_reviews()
    assert pending["review_id"] == result.review_id and pending["flag_id"] == flag["flag_id"]
    payload = get_pending_review(result.thread_id, harness.dependencies, harness.checkpointer)
    assert payload["reasons"] == result.reasons and payload["response_sent"] == result.response


def test_staff_decision_completes_the_case_and_is_recorded():
    harness = Harness(parsed={"urgency": "Urgent"})
    result = harness.start(_find_patient(harness.tools))
    assert result.route == "human_review" and "SOP 4.8" in result.reasons[0]
    done = harness.resume(result.thread_id, {"reviewer_id": "RN-0042", "action": "approve", "note": "Called the patient."})
    assert done.status == "completed" and done.steps[-1] == "finalize"
    assert done.decision["reviewer_id"] == "RN-0042"
    row = harness.tools.store.get_review(result.review_id)
    assert (row["status"], row["reviewer_id"], row["action"], row["note"]) == ("decided", "RN-0042", "approve", "Called the patient.")
    assert row["decided_at"]
    assert harness.tools.flags[0]["status"] == "reviewed_approve" and harness.tools.store.list_pending_reviews() == []
    assert get_pending_review(result.thread_id, harness.dependencies, harness.checkpointer) is None
    with pytest.raises(NotAwaitingReviewError):
        harness.resume(result.thread_id)


def test_invalid_decisions_are_rejected_and_the_case_stays_paused():
    harness = Harness(parsed={"urgency": "Urgent"})
    result = harness.start(_find_patient(harness.tools))
    for bad in ({"action": "approve"}, {"reviewer_id": " ", "action": "approve"},
                {"reviewer_id": "RN-1", "action": "maybe"}, {"reviewer_id": "RN-1", "action": "approve", "x": 1}):
        with pytest.raises(ValidationError):
            harness.resume(result.thread_id, bad)
    with pytest.raises(ValueError, match=r"SOP 4\.5"):  # intake never lowers a band; staff must say why
        harness.resume(result.thread_id, {"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine"})
    assert get_pending_review(result.thread_id, harness.dependencies, harness.checkpointer) is not None
    assert harness.tools.flags[0]["status"] == "awaiting_staff" and len(harness.tools.store.list_pending_reviews()) == 1
    done = harness.resume(result.thread_id, {"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine", "note": "Nurse triage."})
    assert done.band == "Routine" and harness.tools.flags[0]["urgency"] == "Routine"


def test_assistant_asking_for_staff_or_finding_nothing_becomes_human_review():
    steps = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Needs a coverage letter", urgency="Routine")]),
             _make_message("I have passed this to our staff.")]
    harness = Harness(steps=steps)
    result = harness.start(_find_patient(harness.tools))
    assert (result.status, result.route) == ("awaiting_human_review", "human_review")
    assert result.reasons == ["The assistant asked for staff help"] and len(harness.tools.flags) == 1  # no second flag


def test_model_failure_hands_the_case_to_staff():
    harness = Harness()
    harness.dependencies.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
    result = harness.start(_find_patient(harness.tools))
    assert (result.status, result.route) == ("awaiting_human_review", "human_review")
    assert "unavailable" in result.reasons[0] and "staff" in result.response


def test_unknown_caller_and_missing_specialty_go_to_staff_without_running_the_assistant_when_asked():
    harness = Harness(parsed={"specialty": None})
    result = harness.start(None, answer=False)
    assert result.route == "human_review" and len(result.reasons) == 2
    assert result.response == HANDOFF_ANSWER and harness.offered == []
    assert harness.tools.flags[0]["patient_ref"] is None


# ---- eligibility and referrals -----------------------------------------------------------------------
def test_lapsed_coverage_is_never_a_refusal():
    harness = Harness()
    result = harness.start(_find_patient(harness.tools, status="lapsed"))
    assert (result.status, result.route) == ("completed", "auto")  # SOP 5.4
    assert result.eligibility["status"] == "lapsed" and result.eligibility["payment"] == "self_pay_offer"


def test_pending_verification_can_still_be_booked():
    harness = Harness()
    result = harness.start(_find_patient(harness.tools, status="pending_verification"))
    assert result.route == "auto" and result.eligibility["payment"] == "insurance_pending_verification"


def test_referral_on_file_gives_the_plan_and_pending_preauth_needs_staff():
    tools = CareFlowTools()
    ref, row = _find_referral(tools, status="open", preauth_status="pending")
    harness = Harness(tools=tools, parsed={"specialty": row["to_specialty"]})
    result = harness.start(ref, answer=False)
    assert result.plan["referral"]["referral_id"] == row["referral_id"]
    assert result.plan["referral"]["sla_due_on"] == row["sla_due_on"]
    assert result.plan["preauth"]["status"] == "pending" and result.plan["preauth"]["procedure_code"] == row["procedure_code"]
    assert result.route == "human_review" and any("pending" in reason for reason in result.reasons)


def test_accepted_referral_to_a_sign_off_specialty_is_auto():
    tools = CareFlowTools()
    ref, _ = _find_referral(tools, status="accepted", to_specialty="Cardiology", urgency="Routine", preauth_status="not_required")
    harness = Harness(tools=tools, parsed={"specialty": "Cardiology"})
    result = harness.start(ref, answer=False)
    assert result.plan["referral"]["overdue"] is False  # accepted = already actioned
    assert (result.route, result.status) == ("auto", "completed")


def test_referral_urgency_raises_the_band():
    tools = CareFlowTools()
    ref, row = _find_referral(tools, status="accepted", urgency="Urgent")
    harness = Harness(tools=tools, parsed={"specialty": row["to_specialty"]})
    result = harness.start(ref, answer=False)
    assert result.band == "Urgent" and result.route == "human_review"


# ---- repeat contacts ---------------------------------------------------------------------------------
def test_repeat_contact_keeps_at_least_the_earlier_band():
    harness = Harness()
    ref = _find_patient(harness.tools)
    earlier = (harness.tools.clock.now() - timedelta(hours=2)).isoformat(timespec="minutes")
    harness.tools.store.add_interaction(ref, earlier, "sms", "appointment_booking", "callback_scheduled", "Asked to book.", thread_id="old")
    harness.tools.store.add_flag(ref, "Earlier contact", "Urgent", earlier, contact_id="old")
    result = harness.start(ref, answer=False)
    assert result.band == "Urgent" and result.route == "human_review"  # SOP 7.3: parsed as Routine, kept at Urgent


def test_third_clinical_contact_in_72_hours_is_repeat_priority():
    harness = Harness(parsed={"seeks_clinical_advice": True})
    ref = _find_patient(harness.tools)
    for hours in (30, 50):
        contacted_at = (harness.tools.clock.now() - timedelta(hours=hours)).isoformat(timespec="minutes")
        harness.tools.store.add_interaction(ref, contacted_at, "sms", "clinical_question", "escalated_to_clinician", "Asked.", thread_id=f"t{hours}")
    result = harness.start(ref)
    flag = harness.tools.flags[-1]
    assert flag["reason"].startswith("Repeat - Priority") and flag["urgency"] == "Priority"  # OP-CS-014 8.2
    assert result.response == DEFLECTION_SCRIPT  # the flag changes the callback priority, not what is said


# ---- checkpoint: pause, restart, resume --------------------------------------------------------------
def test_paused_case_survives_a_restart_and_resumes_without_repeating_side_effects(tmp_path):
    def open_all():
        store = RuntimeStore(tmp_path / "runtime.db")
        harness = Harness(tools=CareFlowTools(store=store), parsed={"specialty": "Cardiology"})
        harness.checkpointer = open_session_checkpointer(tmp_path / "sessions.db")
        return harness

    harness = open_all()
    ref = _find_patient(harness.tools, specialty="Cardiology")
    result = harness.start(ref, thread_id="case-1")
    assert result.status == "awaiting_human_review"
    calls_before = len(harness.offered)
    close_checkpointer(harness.checkpointer)
    harness.tools.store.close()

    restarted = open_all()  # a new process: new store connection, new checkpointer, new graph
    assert get_pending_review("case-1", restarted.dependencies, restarted.checkpointer)["review_id"] == result.review_id
    assert restarted.tools.booked == {} and restarted.tools.flags[0]["status"] == "awaiting_staff"
    done = restarted.resume("case-1")
    assert done.status == "completed" and done.patient_ref == ref
    assert done.response == result.response and done.reasons == result.reasons and done.plan == result.plan
    assert done.steps == [*result.steps, "finalize"]
    assert len(restarted.tools.flags) == 1 and len(restarted.tools.store.list_rows("reviews")) == 1  # nothing raised twice
    assert restarted.offered == [] and calls_before > 0  # the model was not called again on resume (a sign-off case)
    row = restarted.tools.store.get_review(result.review_id)
    assert (row["status"], row["reviewer_id"]) == ("decided", "RN-0042")
    close_checkpointer(restarted.checkpointer)
    restarted.tools.store.close()


def test_threads_are_independent():
    harness = Harness(parsed={"urgency": "Urgent"})
    first, second = harness.start(_find_patient(harness.tools)), harness.start(_find_patient(harness.tools))
    harness.resume(first.thread_id)
    assert get_pending_review(second.thread_id, harness.dependencies, harness.checkpointer) is not None
    assert [review["review_id"] for review in harness.tools.store.list_pending_reviews()] == [second.review_id]
