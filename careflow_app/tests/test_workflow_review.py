"""Offline tests for the fixes the architect asked for in the M5 review (notes/feedback.md, M5-1 to M5-6)."""

from datetime import timedelta

import pytest

from careflow.domain import policies
from careflow.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from careflow.store import RuntimeStore
from careflow.tools import CareFlowTools
from careflow.workflows import AlreadyTriaged, pending_review, resume_triage, start_triage
from careflow.workflows.triage import ASSISTANT_DEFLECTED, URGENT_HANDOFF

from .test_workflow import Harness, _call, _msg, _patient

RECEIVED = "2026-08-01T09:00:00Z"


def _start(h, text, patient_ref=None, **kw):
    record = {"channel": "web_form", "received_at": RECEIVED, "raw_text": text, "patient_ref": patient_ref}
    return start_triage(record, h.deps, h.checkpointer, **kw)


def _contact(h, ref, thread_id):
    rows = [c for c in h.tools.store.rows_for_patient("patient_interactions", ref) if c["thread_id"] == thread_id]
    return rows[0] if rows else None


def _earlier(h, ref, topic, *hours):
    for n in hours:
        at = (h.tools.clock.now() - timedelta(hours=n)).isoformat(timespec="minutes")
        h.tools.store.add_interaction(ref, at, "sms", topic, "callback_scheduled", "Earlier contact.", thread_id=f"old-{n}")


# ---- M5-1: identity ----------------------------------------------------------------------------------
def test_a_member_number_alone_or_with_the_wrong_name_identifies_nobody():
    tools = CareFlowTools()
    ref = _patient(tools)
    p = tools.patients[ref]
    for name in (None, "John Impersonator", p["given_name"]):
        h = Harness(tools=CareFlowTools(), parsed={"patient_name": name, "insurance_member_number": p["member_number"]})
        r = _start(h, "What are my referrals?")
        assert r.patient_ref is None, name
        assert r.route == "human_review" and "Patient not identified on record" in r.reasons
        assert h.offered and all("book_appointment" not in names for names in h.offered)
        # Nothing of the real patient's reached the model.
        assert h.tools.store.rows_for_patient("patient_interactions", ref) == tools.store.rows_for_patient("patient_interactions", ref)


def test_full_name_plus_member_number_identifies_the_caller():
    tools = CareFlowTools()
    ref = _patient(tools)
    p = tools.patients[ref]
    h = Harness(tools=tools, parsed={"patient_name": f"{p['given_name']} {p['family_name']}".lower(),
                                     "insurance_member_number": p["member_number"]})
    assert _start(h, "I need an appointment.").patient_ref == ref


def test_verified_patient_is_kept_when_the_message_cannot_be_parsed():
    h = Harness(parse_error=RuntimeError("model down"))
    ref = _patient(h.tools)
    r = _start(h, "hello", ref, thread_id="t-fail")
    assert r.patient_ref == ref and h.tools.flags[0]["patient_ref"] == ref
    assert h.tools.store.review(r.review_id)["patient_ref"] == ref
    assert _contact(h, ref, "t-fail")["topic"] == "clinical_question"


# ---- M5-2: the band is never lowered -----------------------------------------------------------------
def test_assistant_flag_is_raised_to_the_workflow_band_and_keeps_the_triage_reasons():
    steps = [_msg(calls=[_call("c1", "flag_for_human", reason="Needs a callback", urgency="Routine")]), _msg("Passed to staff.")]
    h = Harness(steps=steps)  # a third contact: held at Priority, and the specialists still write a reply
    ref = _patient(h.tools)
    _earlier(h, ref, "appointment_booking", 30, 50)
    r = _start(h, "I need an appointment.", ref)
    (flag,) = h.tools.flags
    assert flag["urgency"] == "Priority" and r.band == "Priority"
    assert "Needs a callback" in flag["reason"] and "8.2" in flag["reason"]


def test_a_higher_band_from_the_assistant_is_kept():
    steps = [_msg(calls=[_call("c1", "flag_for_human", reason="Sounds serious", urgency="Urgent")]), _msg("Passed to staff.")]
    h = Harness(steps=steps)
    r = _start(h, "I need an appointment.", _patient(h.tools))
    assert r.band == "Urgent" and h.tools.flags[0]["urgency"] == "Urgent"


def test_third_non_clinical_contact_is_priority_and_marked_on_the_flag():
    h = Harness()
    ref = _patient(h.tools)
    _earlier(h, ref, "appointment_booking", 30, 50)
    r = _start(h, "I need an appointment.", ref, answer=False)
    (flag,) = h.tools.flags
    assert (r.band, flag["urgency"]) == ("Priority", "Priority")
    assert flag["reason"].startswith("Repeat - Priority: ") and any("8.2" in x for x in r.reasons)


def test_repeat_within_24_hours_is_marked_on_the_flag():
    h = Harness(parsed={"urgency": "Urgent"})
    ref = _patient(h.tools)
    _earlier(h, ref, "appointment_booking", 3)
    _start(h, "I need an appointment.", ref, answer=False)
    assert h.tools.flags[0]["reason"].startswith("REPEAT: ")


@pytest.mark.parametrize("text, band", [
    ("I have crushing chest pain and can’t breathe", "Emergent"),
    ("My stitches came out last night", "Urgent"),
    ("I would like a routine check-up", None),
])
def test_trigger_phrases(text, band):
    assert policies.trigger_band(text) == band


def test_emergency_statement_is_given_when_an_emergency_message_cannot_be_parsed():
    h = Harness(parse_error=RuntimeError("model down"))
    r = _start(h, "I have crushing chest pain and can't breathe", _patient(h.tools))
    assert r.band == "Emergent" and r.response == f"{EMERGENCY_STATEMENT}\n{DEFLECTION_SCRIPT}"
    assert h.tools.flags[0]["urgency"] == "Emergent"


def test_trigger_phrase_is_a_floor_under_the_parsed_band():
    h = Harness(parsed={"urgency": "Routine"})
    r = _start(h, "The wound opened up and two stitches came out. Can I get a dermatology slot?", _patient(h.tools))
    assert r.band == "Urgent" and r.route == "human_review"
    h = Harness(parsed={"urgency": "Routine"})
    assert _start(h, "I get chest pain when I climb stairs, can I see someone?", _patient(h.tools)).route == "refuse"


def test_emergent_is_never_lowered_even_with_a_note():
    h = Harness(parsed={"urgency": "Emergent"})
    r = _start(h, "help", _patient(h.tools))
    decision = {"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine", "note": "Felt better."}
    with pytest.raises(ValueError, match="3.3"):
        resume_triage(r.thread_id, decision, h.deps, h.checkpointer)
    assert pending_review(r.thread_id, h.deps, h.checkpointer) is not None
    assert h.tools.flags[0]["urgency"] == "Emergent"
    assert policies.validate_reband("Urgent", "Routine", "Nurse triage.") is None
    assert policies.validate_reband("Routine", "Urgent", None) is None


# ---- M5-3: referrals and repeat topics ---------------------------------------------------------------
def test_rejected_referral_with_denied_preauth_goes_to_staff():
    # Intake record CAREFLOW-00175: MHP-P-00005 holds a rejected Dermatology referral with pre-auth denied.
    h = Harness(parsed={"specialty": "Dermatology"})
    r = _start(h, "I need a dermatology appointment.", "MHP-P-00005", answer=False)
    assert r.route == "human_review"
    assert r.plan["referral"]["status"] == "rejected" and r.plan["preauth"]["status"] == "denied"
    assert any("rejected" in x for x in r.reasons) and any("denied" in x for x in r.reasons)


@pytest.mark.parametrize("ref, specialty", [("MHP-P-00041", "Pulmonology"), ("MHP-P-00212", "ENT")])
def test_unlisted_specialty_with_an_accepted_referral_is_auto(ref, specialty):
    # Intake records CAREFLOW-00057 and CAREFLOW-00154.
    h = Harness(parsed={"specialty": specialty})
    r = _start(h, "I need an appointment.", ref, answer=False)
    assert r.plan["referral"]["status"] == "accepted" and (r.route, r.reasons) == ("auto", [])


def test_unlisted_specialty_without_a_referral_still_goes_to_staff():
    h = Harness(parsed={"specialty": "ENT"})
    r = _start(h, "I need an appointment.", _patient(h.tools, specialty="ENT"), answer=False)
    assert r.route == "human_review" and "neither" in r.reasons[0]


def test_open_referral_waits_for_triage_acceptance():
    ok = dict(seeks_clinical_advice=False, urgency="Routine", specialty="Cardiology", has_referral=True)
    assert policies.triage_route(**ok, referral_status="accepted")[0] == "auto"
    route, reasons = policies.triage_route(**ok, referral_status="open")
    assert route == "human_review" and "triage acceptance" in reasons[0]


@pytest.mark.parametrize("text, topic", [
    ("Any news on my referral MHP-R-00012?", "referral_status"),
    ("How much will I owe for the scan?", "billing_question"),
    ("Is my plan active?", "eligibility_check"),
    ("Need an ortho appt next week", "appointment_booking"),
])
def test_intake_topic(text, topic):
    assert policies.intake_topic(text) == topic


def test_repeat_rules_use_what_the_message_is_about():
    h = Harness()
    ref = _patient(h.tools)
    _earlier(h, ref, "referral_status", 30, 50)
    assert _start(h, "I need a dermatology appointment.", ref, answer=False).band == "Routine"  # a different issue
    r = _start(h, "Any update on my referral?", ref, answer=False)
    assert r.band == "Priority" and h.tools.flags[-1]["reason"].startswith("Repeat - Priority")


# ---- M5-4: the contact record matches the outcome ----------------------------------------------------
def test_human_review_is_recorded_as_a_callback_with_the_flag_and_the_real_topic():
    h = Harness(parsed={"urgency": "Urgent"})
    ref = _patient(h.tools)
    r = _start(h, "I need an appointment.", ref, thread_id="t-urgent")
    row = _contact(h, ref, "t-urgent")
    assert (row["topic"], row["outcome"]) == ("appointment_booking", "callback_scheduled")
    assert r.flag_id in row["summary"] and "(Urgent)" in row["summary"]
    # So the next contact within 24 hours is a REPEAT and keeps the Urgent band (SOP 7.3).
    h2 = Harness(tools=h.tools)
    again = _start(h2, "Still need that appointment.", ref, answer=False)
    assert again.band == "Urgent" and h.tools.flags[-1]["reason"].startswith("REPEAT: ")


def test_contact_is_recorded_when_the_assistant_did_not_run():
    for kw, setup in (({"answer": False}, None), ({}, "down")):
        h = Harness(parsed={"urgency": "Urgent"})
        if setup:
            h.deps.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
        ref = _patient(h.tools)
        _start(h, "I need an appointment.", ref, thread_id="t1", **kw)
        assert _contact(h, ref, "t1")["outcome"] == "callback_scheduled"


def test_auto_contact_is_recorded_once_by_the_assistant():
    h = Harness()
    ref = _patient(h.tools)
    _start(h, "I need an appointment.", ref, thread_id="t-auto")
    assert _contact(h, ref, "t-auto")["outcome"] == "resolved"


# ---- M5-5: one case per thread, one store per case ---------------------------------------------------
def test_a_thread_holds_one_case():
    h = Harness(parsed={"urgency": "Urgent"})
    ref = _patient(h.tools)
    r = _start(h, "I need an appointment.", ref, thread_id="t1")
    with pytest.raises(AlreadyTriaged):
        _start(h, "A different message.", ref, thread_id="t1")
    assert pending_review("t1", h.deps, h.checkpointer)["review_id"] == r.review_id  # the paused case is untouched
    resume_triage("t1", {"reviewer_id": "RN-1", "action": "approve"}, h.deps, h.checkpointer)
    with pytest.raises(AlreadyTriaged):
        _start(h, "And again.", ref, thread_id="t1")
    assert len(h.tools.flags) == 1


def test_a_paused_case_needs_its_checkpointer():
    h = Harness(parsed={"urgency": "Urgent"})
    r = _start(h, "I need an appointment.", _patient(h.tools))
    with pytest.raises(ValueError, match="checkpointer"):
        pending_review(r.thread_id, h.deps)
    with pytest.raises(ValueError, match="checkpointer"):
        resume_triage(r.thread_id, {"reviewer_id": "RN-1", "action": "approve"}, h.deps)


def test_default_deps_with_a_checkpointer_share_the_durable_store(tmp_path, monkeypatch):
    import careflow.store as store_module
    from careflow.agent import close_checkpointer, session_checkpointer

    monkeypatch.setattr(store_module, "RUNTIME_DB", tmp_path / "runtime.db")
    cp = session_checkpointer(tmp_path / "sessions.db")
    assert pending_review("nobody", checkpointer=cp) is None  # default deps open tmp_path/runtime.db
    assert (tmp_path / "runtime.db").exists()
    close_checkpointer(cp)
    close_checkpointer(None)


def test_a_clinical_deflection_by_the_assistant_replaces_the_administrative_reasons():
    steps = [_msg(calls=[_call("c1", "flag_for_human", reason="Asked about a dose", urgency="Routine", clinical=True)]),
             _msg(DEFLECTION_SCRIPT)]
    h = Harness(parsed={"specialty": "Cardiology"}, steps=steps)
    ref = _patient(h.tools, specialty="Cardiology")
    _earlier(h, ref, "appointment_booking", 30, 50)  # a third contact: the workflow's band is Priority
    r = _start(h, "I need an appointment.", ref)
    assert (r.route, r.reasons) == ("refuse", [ASSISTANT_DEFLECTED])
    assert h.tools.flags[0]["urgency"] == "Priority"  # the ticket still carries the workflow's band


def test_step_limit_flag_is_not_counted_as_the_assistant_asking_for_help():
    loop = _msg(calls=[_call("c1", "check_eligibility", patient_ref="MHP-P-00001")])
    h = Harness(steps=[loop])
    r = _start(h, "I need an appointment.", _patient(h.tools))
    assert r.reasons == ["The assistant reached its step limit"] and len(h.tools.flags) == 1


# ---- M5-6: ids ---------------------------------------------------------------------------------------
def test_flag_and_review_ids_do_not_collide_after_a_row_is_removed():
    s = RuntimeStore(":memory:")
    for n in (1, 2):
        s.add_flag(None, f"reason {n}", "Routine", "2026-08-01T09:00", contact_id=f"t{n}")
        s.add_review(f"t{n}", "human_review", ["x"], "2026-08-01T09:00")
    s.conn.execute("DELETE FROM staff_flags WHERE flag_id = 'FLAG-0001'")
    s.conn.execute("DELETE FROM reviews WHERE review_id = 'REV-0001'")
    assert s.add_flag(None, "reason 3", "Routine", "2026-08-01T09:00", contact_id="t3")[0]["flag_id"] == "FLAG-0003"
    assert s.add_review("t3", "human_review", ["x"], "2026-08-01T09:00")["review_id"] == "REV-0003"


# ---- Lead's reply 0.8 (2026-11-22): the model is not asked about a case the code has decided is Urgent ----
@pytest.mark.parametrize("parsed, text", [
    ({"urgency": "Urgent"}, "I need an appointment."),
    ({"urgency": "Routine"}, "The wound has opened up and two stitches have come out. I need an appointment soon."),
])
def test_an_urgent_case_gets_the_fixed_handoff_and_no_model_call(parsed, text):
    clinical = [_msg(calls=[_call("c1", "flag_for_human", reason="Wound", urgency="Urgent", clinical=True)]), _msg(DEFLECTION_SCRIPT)]
    h = Harness(parsed=parsed, steps=clinical)  # what the live model did when it was asked
    r = _start(h, text, _patient(h.tools))
    assert (r.route, r.band, r.status) == ("human_review", "Urgent", "awaiting_human_review")
    assert h.offered == [] and r.response == URGENT_HANDOFF and not r.agent
    assert "SOP 4.8" in r.reasons[0] and h.tools.booked == {}
    (flag,) = h.tools.flags
    assert flag["urgency"] == "Urgent" and "SOP 4.8" in flag["reason"]


def test_a_routine_case_held_for_sign_off_still_gets_the_specialists_reply():
    h = Harness(parsed={"specialty": "Cardiology"})
    r = _start(h, "I need an appointment.", _patient(h.tools, specialty="Cardiology"))
    assert r.route == "human_review" and h.offered and r.response == "Here is what I found."
    assert "book_appointment" not in h.offered[0]
