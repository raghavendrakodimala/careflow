"""Offline tests for the M5 triage workflow: fixed step order, routes, the human-review pause and
resuming from a SQLite checkpoint. The intake parser and the model are fakes; stores are in memory or tmp_path."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from careflow.agent import close_checkpointer, session_checkpointer
from careflow.graph import Deps
from careflow.memory import PatientMemory
from careflow.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from careflow.schemas import PatientRequest
from careflow.store import RuntimeStore
from careflow.tools import CareFlowTools
from careflow.workflows import (HANDOFF_ANSWER, NODES, NotAwaitingReview, TriageDeps, build_triage_graph,
                                pending_review, resume_triage, start_triage)

APPROVE = {"reviewer_id": "RN-0042", "action": "approve"}


def _call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _msg(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls or [])


class Harness:
    """A workflow with a scripted intake parse and a scripted model, recording what the model was offered."""

    def __init__(self, tools=None, parsed=None, steps=None, parse_error=None):
        self.tools = tools or CareFlowTools()
        self.parsed = {"urgency": "Routine", "seeks_clinical_advice": False, "specialty": "Dermatology", **(parsed or {})}
        self.steps = list(steps or [_msg("Here is what I found.")])
        self.parse_error = parse_error
        self.offered: list[set[str]] = []
        self.checkpointer = MemorySaver()
        self.deps = TriageDeps(
            agent=Deps(tools=self.tools, model=self._model, memory=PatientMemory(self.tools.store, self.tools.clock)),
            parse=self._parse,
        )

    def _parse(self, record):
        if self.parse_error:
            raise self.parse_error
        return PatientRequest(**self.parsed)

    def _model(self, messages, specs):
        self.offered.append({s["function"]["name"] for s in specs})
        return self.steps[min(len(self.offered), len(self.steps)) - 1]

    def start(self, patient_ref=None, **kw):
        record = {"channel": "web_form", "received_at": "2026-08-01T09:00:00Z", "raw_text": "I need an appointment.",
                  "patient_ref": patient_ref, "ground_truth": {"urgency": "Emergent"}}
        return start_triage(record, self.deps, self.checkpointer, **kw)

    def resume(self, thread_id, decision=APPROVE):
        return resume_triage(thread_id, decision, self.deps, self.checkpointer)


def _patient(tools, status="active", specialty=None):
    """A patient with the given coverage and no live referral (optionally: none to `specialty` only)."""
    for ref, p in tools.patients.items():
        live = [r for r in tools.store.rows_for_patient("referrals", ref) if r["status"] in ("open", "accepted", "scheduled")]
        if p["eligibility_status"] == status and not [r for r in live if specialty in (None, r["to_specialty"])]:
            return ref
    raise AssertionError("no such patient in the seed data")


def _referral(tools, **want):
    """(patient_ref, referral row) for a live referral whose details match `want`."""
    details = tools.referral_details
    for r in tools.store.rows("referrals"):
        row = {**r, **details[r["referral_id"]]}
        if all(row[k] == v for k, v in want.items()):
            return r["patient_ref"], row
    raise AssertionError(f"no referral with {want}")


# ---- shape and order ---------------------------------------------------------------------------------
def test_graph_has_the_planned_nodes():
    h = Harness()
    graph = build_triage_graph(h.deps, h.checkpointer).get_graph()
    assert set(NODES) <= set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("guard", "intake"), ("intake", "hydrate"), ("hydrate", "deflect"), ("hydrate", "eligibility"), ("eligibility", "routing"),
            ("routing", "team"), ("team", "safety_review"), ("safety_review", "decide"), ("deflect", "decide"), ("decide", "review"),
            ("decide", "finalize"), ("review", "finalize")} <= edges


def test_auto_request_runs_the_steps_in_order_and_completes():
    h = Harness()
    r = h.start(_patient(h.tools))
    assert r.steps == ["guard", "intake", "hydrate", "eligibility", "routing", "team", "safety_review", "decide", "finalize"]
    assert (r.status, r.route, r.reasons) == ("completed", "auto", [])
    assert r.response == "Here is what I found."
    assert r.review_id is None and h.tools.flags == [] and h.tools.store.pending_reviews() == []
    assert "book_appointment" in h.offered[0]  # nothing to wait for, so the assistant may book


def test_labels_never_enter_the_workflow_state():
    h = Harness()
    r = h.start(_patient(h.tools))
    state = build_triage_graph(h.deps, h.checkpointer).get_state({"configurable": {"thread_id": r.thread_id}})
    assert "ground_truth" not in state.values["record"]


# ---- refuse: act, then pause -------------------------------------------------------------------------
def test_clinical_question_gets_the_script_and_a_ticket_at_once_then_waits_for_staff():
    h = Harness(parsed={"seeks_clinical_advice": True})
    ref = _patient(h.tools)
    r = h.start(ref)
    assert r.steps == ["guard", "intake", "hydrate", "deflect", "decide"]
    assert (r.status, r.route) == ("awaiting_human_review", "refuse")
    assert r.response == DEFLECTION_SCRIPT
    assert h.offered == []  # the model never saw the message
    (flag,) = h.tools.flags
    assert (flag["category"], flag["patient_ref"], flag["status"]) == ("clinical", ref, "awaiting_staff")
    assert r.eligibility == {}  # insurance is not raised before the clinical response (SOP 5.1)
    (contact,) = [c for c in h.tools.store.rows_for_patient("patient_interactions", ref) if c["thread_id"] == r.thread_id]
    assert (contact["topic"], contact["outcome"]) == ("clinical_question", "escalated_to_clinician")


def test_emergent_request_starts_with_the_emergency_statement_and_is_never_booked():
    h = Harness(parsed={"urgency": "Emergent", "specialty": "Cardiology"})
    r = h.start(_patient(h.tools))
    assert r.route == "refuse" and r.response.startswith(EMERGENCY_STATEMENT) and DEFLECTION_SCRIPT in r.response
    assert h.tools.flags[0]["urgency"] == "Emergent" and h.tools.flags[0]["category"] == "clinical"
    assert h.offered == [] and h.tools.booked == {}
    assert "referral policy 3.3" in r.reasons[0]


def test_unreadable_message_fails_closed():
    h = Harness(parse_error=RuntimeError("model down"))
    r = h.start(_patient(h.tools))
    assert (r.status, r.route) == ("awaiting_human_review", "refuse")
    assert r.response == DEFLECTION_SCRIPT and "9.3" in r.reasons[0]
    assert h.tools.flags[0]["category"] == "clinical"


def test_a_clinical_question_the_parser_missed_is_still_a_refusal():
    # The assistant's own rule catches it: it flags a clinician and gives the script.
    steps = [_msg(calls=[_call("c1", "flag_for_human", reason="Asked about a dose", urgency="Routine", clinical=True)]),
             _msg(DEFLECTION_SCRIPT)]
    h = Harness(steps=steps)
    r = h.start(_patient(h.tools))
    assert (r.status, r.route) == ("awaiting_human_review", "refuse")
    assert len(h.tools.flags) == 1 and r.flag_id == h.tools.flags[0]["flag_id"]


# ---- human review ------------------------------------------------------------------------------------
def test_review_needed_withholds_booking_and_pauses():
    h = Harness(parsed={"specialty": "Cardiology"})
    ref = _patient(h.tools, specialty="Cardiology")
    r = h.start(ref)
    assert (r.status, r.route) == ("awaiting_human_review", "human_review")
    assert "sign-off" in r.reasons[0] and r.plan["sign_off_required"] is True and r.plan["referral"] is None
    assert h.offered and all("book_appointment" not in names for names in h.offered)
    (flag,) = h.tools.flags
    assert (flag["category"], flag["status"]) == ("administrative", "awaiting_staff")
    assert h.tools.booked == {}
    (pending,) = h.tools.store.pending_reviews()
    assert pending["review_id"] == r.review_id and pending["flag_id"] == flag["flag_id"]
    payload = pending_review(r.thread_id, h.deps, h.checkpointer)
    assert payload["reasons"] == r.reasons and payload["response_sent"] == r.response


def test_staff_decision_completes_the_case_and_is_recorded():
    h = Harness(parsed={"urgency": "Urgent"})
    r = h.start(_patient(h.tools))
    assert r.route == "human_review" and "SOP 4.8" in r.reasons[0]
    done = h.resume(r.thread_id, {"reviewer_id": "RN-0042", "action": "approve", "note": "Called the patient."})
    assert done.status == "completed" and done.steps[-1] == "finalize"
    assert done.decision["reviewer_id"] == "RN-0042"
    row = h.tools.store.review(r.review_id)
    assert (row["status"], row["reviewer_id"], row["action"], row["note"]) == ("decided", "RN-0042", "approve", "Called the patient.")
    assert row["decided_at"]
    assert h.tools.flags[0]["status"] == "reviewed_approve" and h.tools.store.pending_reviews() == []
    assert pending_review(r.thread_id, h.deps, h.checkpointer) is None
    with pytest.raises(NotAwaitingReview):
        h.resume(r.thread_id)


def test_invalid_decisions_are_rejected_and_the_case_stays_paused():
    h = Harness(parsed={"urgency": "Urgent"})
    r = h.start(_patient(h.tools))
    for bad in ({"action": "approve"}, {"reviewer_id": " ", "action": "approve"},
                {"reviewer_id": "RN-1", "action": "maybe"}, {"reviewer_id": "RN-1", "action": "approve", "x": 1}):
        with pytest.raises(ValidationError):
            h.resume(r.thread_id, bad)
    with pytest.raises(ValueError, match="SOP 4.5"):  # intake never lowers a band; staff must say why
        h.resume(r.thread_id, {"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine"})
    assert pending_review(r.thread_id, h.deps, h.checkpointer) is not None
    assert h.tools.flags[0]["status"] == "awaiting_staff" and len(h.tools.store.pending_reviews()) == 1
    done = h.resume(r.thread_id, {"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine", "note": "Nurse triage."})
    assert done.band == "Routine" and h.tools.flags[0]["urgency"] == "Routine"


def test_assistant_asking_for_staff_or_finding_nothing_becomes_human_review():
    steps = [_msg(calls=[_call("c1", "flag_for_human", reason="Needs a coverage letter", urgency="Routine")]),
             _msg("I have passed this to our staff.")]
    h = Harness(steps=steps)
    r = h.start(_patient(h.tools))
    assert (r.status, r.route) == ("awaiting_human_review", "human_review")
    assert r.reasons == ["The assistant asked for staff help"] and len(h.tools.flags) == 1  # no second flag


def test_model_failure_hands_the_case_to_staff():
    h = Harness()
    h.deps.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
    r = h.start(_patient(h.tools))
    assert (r.status, r.route) == ("awaiting_human_review", "human_review")
    assert "unavailable" in r.reasons[0] and "staff" in r.response


def test_unknown_caller_and_missing_specialty_go_to_staff_without_running_the_assistant_when_asked():
    h = Harness(parsed={"specialty": None})
    r = h.start(None, answer=False)
    assert r.route == "human_review" and len(r.reasons) == 2
    assert r.response == HANDOFF_ANSWER and h.offered == []
    assert h.tools.flags[0]["patient_ref"] is None


# ---- eligibility and referrals -----------------------------------------------------------------------
def test_lapsed_coverage_is_never_a_refusal():
    h = Harness()
    r = h.start(_patient(h.tools, status="lapsed"))
    assert (r.status, r.route) == ("completed", "auto")  # SOP 5.4
    assert r.eligibility["status"] == "lapsed" and r.eligibility["payment"] == "self_pay_offer"


def test_pending_verification_can_still_be_booked():
    h = Harness()
    r = h.start(_patient(h.tools, status="pending_verification"))
    assert r.route == "auto" and r.eligibility["payment"] == "insurance_pending_verification"


def test_referral_on_file_gives_the_plan_and_pending_preauth_needs_staff():
    tools = CareFlowTools()
    ref, row = _referral(tools, status="open", preauth_status="pending")
    h = Harness(tools=tools, parsed={"specialty": row["to_specialty"]})
    r = h.start(ref, answer=False)
    assert r.plan["referral"]["referral_id"] == row["referral_id"]
    assert r.plan["referral"]["sla_due_on"] == row["sla_due_on"]
    assert r.plan["preauth"]["status"] == "pending" and r.plan["preauth"]["procedure_code"] == row["procedure_code"]
    assert r.route == "human_review" and any("pending" in x for x in r.reasons)


def test_accepted_referral_to_a_sign_off_specialty_is_auto():
    tools = CareFlowTools()
    ref, row = _referral(tools, status="accepted", to_specialty="Cardiology", urgency="Routine", preauth_status="not_required")
    h = Harness(tools=tools, parsed={"specialty": "Cardiology"})
    r = h.start(ref, answer=False)
    assert r.plan["referral"]["overdue"] is False  # accepted = already actioned
    assert (r.route, r.status) == ("auto", "completed")


def test_referral_urgency_raises_the_band():
    tools = CareFlowTools()
    ref, row = _referral(tools, status="accepted", urgency="Urgent")
    h = Harness(tools=tools, parsed={"specialty": row["to_specialty"]})
    r = h.start(ref, answer=False)
    assert r.band == "Urgent" and r.route == "human_review"


# ---- repeat contacts ---------------------------------------------------------------------------------
def test_repeat_contact_keeps_at_least_the_earlier_band():
    h = Harness()
    ref = _patient(h.tools)
    earlier = (h.tools.clock.now() - timedelta(hours=2)).isoformat(timespec="minutes")
    h.tools.store.add_interaction(ref, earlier, "sms", "appointment_booking", "callback_scheduled", "Asked to book.", thread_id="old")
    h.tools.store.add_flag(ref, "Earlier contact", "Urgent", earlier, contact_id="old")
    r = h.start(ref, answer=False)
    assert r.band == "Urgent" and r.route == "human_review"  # SOP 7.3: parsed as Routine, kept at Urgent


def test_third_clinical_contact_in_72_hours_is_repeat_priority():
    h = Harness(parsed={"seeks_clinical_advice": True})
    ref = _patient(h.tools)
    for hours in (30, 50):
        at = (h.tools.clock.now() - timedelta(hours=hours)).isoformat(timespec="minutes")
        h.tools.store.add_interaction(ref, at, "sms", "clinical_question", "escalated_to_clinician", "Asked.", thread_id=f"t{hours}")
    r = h.start(ref)
    flag = h.tools.flags[-1]
    assert flag["reason"].startswith("Repeat - Priority") and flag["urgency"] == "Priority"  # OP-CS-014 8.2
    assert r.response == DEFLECTION_SCRIPT  # the flag changes the callback priority, not what is said


# ---- checkpoint: pause, restart, resume --------------------------------------------------------------
def test_paused_case_survives_a_restart_and_resumes_without_repeating_side_effects(tmp_path):
    def open_all():
        store = RuntimeStore(tmp_path / "runtime.db")
        h = Harness(tools=CareFlowTools(store=store), parsed={"specialty": "Cardiology"})
        h.checkpointer = session_checkpointer(tmp_path / "sessions.db")
        return h

    h = open_all()
    ref = _patient(h.tools, specialty="Cardiology")
    r = h.start(ref, thread_id="case-1")
    assert r.status == "awaiting_human_review"
    calls_before = len(h.offered)
    close_checkpointer(h.checkpointer)
    h.tools.store.close()

    h2 = open_all()  # a new process: new store connection, new checkpointer, new graph
    assert pending_review("case-1", h2.deps, h2.checkpointer)["review_id"] == r.review_id
    assert h2.tools.booked == {} and h2.tools.flags[0]["status"] == "awaiting_staff"
    done = h2.resume("case-1")
    assert done.status == "completed" and done.patient_ref == ref
    assert done.response == r.response and done.reasons == r.reasons and done.plan == r.plan
    assert done.steps == [*r.steps, "finalize"]
    assert len(h2.tools.flags) == 1 and len(h2.tools.store.rows("reviews")) == 1  # nothing raised twice
    assert h2.offered == [] and calls_before > 0  # the model was not called again on resume (a sign-off case)
    row = h2.tools.store.review(r.review_id)
    assert (row["status"], row["reviewer_id"]) == ("decided", "RN-0042")
    close_checkpointer(h2.checkpointer)
    h2.tools.store.close()


def test_threads_are_independent():
    h = Harness(parsed={"urgency": "Urgent"})
    a, b = h.start(_patient(h.tools)), h.start(_patient(h.tools))
    h.resume(a.thread_id)
    assert pending_review(b.thread_id, h.deps, h.checkpointer) is not None
    assert [x["review_id"] for x in h.tools.store.pending_reviews()] == [b.review_id]
