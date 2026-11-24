"""Offline tests for the triage workflow: identity, bands, referrals, repeat topics, contact records, threads and ids."""

from datetime import timedelta

import pytest

from src.domain import policies
from src.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from src.store import RuntimeStore
from src.tools import CareFlowTools
from src.workflows import AlreadyTriagedError, get_pending_review, resume_triage, start_triage
from src.workflows.triage import ASSISTANT_DEFLECTED, URGENT_HANDOFF

from .test_workflow import Harness, _find_patient, _make_call, _make_message

RECEIVED = "2026-08-01T09:00:00Z"


def _start(harness, text, patient_ref=None, **options):
    record = {"channel": "web_form", "received_at": RECEIVED, "raw_text": text, "patient_ref": patient_ref}
    return start_triage(record, harness.dependencies, harness.checkpointer, **options)


def _find_contact(harness, ref, thread_id):
    rows = [interaction for interaction in harness.tools.store.list_rows_for_patient("patient_interactions", ref) if interaction["thread_id"] == thread_id]
    return rows[0] if rows else None


def _add_earlier_contacts(harness, ref, topic, *hours):
    for hours_ago in hours:
        contacted_at = (harness.tools.clock.now() - timedelta(hours=hours_ago)).isoformat(timespec="minutes")
        harness.tools.store.add_interaction(ref, contacted_at, "sms", topic, "callback_scheduled", "Earlier contact.", thread_id=f"old-{hours_ago}")


# ---- identity ----------------------------------------------------------------------------------
def test_a_member_number_alone_or_with_the_wrong_name_identifies_nobody():
    tools = CareFlowTools()
    ref = _find_patient(tools)
    patient = tools.patients[ref]
    for name in (None, "John Impersonator", patient["given_name"]):
        harness = Harness(tools=CareFlowTools(), parsed={"patient_name": name, "insurance_member_number": patient["member_number"]})
        result = _start(harness, "What are my referrals?")
        assert result.patient_ref is None, name
        assert result.route == "human_review" and "Patient not identified on record" in result.reasons
        assert harness.offered and all("book_appointment" not in names for names in harness.offered)
        # Nothing of the real patient's reached the model.
        assert harness.tools.store.list_rows_for_patient("patient_interactions", ref) == tools.store.list_rows_for_patient("patient_interactions", ref)


def test_full_name_plus_member_number_identifies_the_caller():
    tools = CareFlowTools()
    ref = _find_patient(tools)
    patient = tools.patients[ref]
    harness = Harness(tools=tools, parsed={"patient_name": f"{patient['given_name']} {patient['family_name']}".lower(),
                                     "insurance_member_number": patient["member_number"]})
    assert _start(harness, "I need an appointment.").patient_ref == ref


def test_verified_patient_is_kept_when_the_message_cannot_be_parsed():
    harness = Harness(parse_error=RuntimeError("model down"))
    ref = _find_patient(harness.tools)
    result = _start(harness, "hello", ref, thread_id="t-fail")
    assert result.patient_ref == ref and harness.tools.flags[0]["patient_ref"] == ref
    assert harness.tools.store.get_review(result.review_id)["patient_ref"] == ref
    assert _find_contact(harness, ref, "t-fail")["topic"] == "clinical_question"


# ---- the band is never lowered -----------------------------------------------------------------
def test_assistant_flag_is_raised_to_the_workflow_band_and_keeps_the_triage_reasons():
    steps = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Needs a callback", urgency="Routine")]), _make_message("Passed to staff.")]
    harness = Harness(steps=steps)  # a third contact: held at Priority, and the specialists still write a reply
    ref = _find_patient(harness.tools)
    _add_earlier_contacts(harness, ref, "appointment_booking", 30, 50)
    result = _start(harness, "I need an appointment.", ref)
    (flag,) = harness.tools.flags
    assert flag["urgency"] == "Priority" and result.band == "Priority"
    assert "Needs a callback" in flag["reason"] and "8.2" in flag["reason"]


def test_a_higher_band_from_the_assistant_is_kept():
    steps = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Sounds serious", urgency="Urgent")]), _make_message("Passed to staff.")]
    harness = Harness(steps=steps)
    result = _start(harness, "I need an appointment.", _find_patient(harness.tools))
    assert result.band == "Urgent" and harness.tools.flags[0]["urgency"] == "Urgent"


def test_third_non_clinical_contact_is_priority_and_marked_on_the_flag():
    harness = Harness()
    ref = _find_patient(harness.tools)
    _add_earlier_contacts(harness, ref, "appointment_booking", 30, 50)
    result = _start(harness, "I need an appointment.", ref, answer=False)
    (flag,) = harness.tools.flags
    assert (result.band, flag["urgency"]) == ("Priority", "Priority")
    assert flag["reason"].startswith("Repeat - Priority: ") and any("8.2" in reason for reason in result.reasons)


def test_repeat_within_24_hours_is_marked_on_the_flag():
    harness = Harness(parsed={"urgency": "Urgent"})
    ref = _find_patient(harness.tools)
    _add_earlier_contacts(harness, ref, "appointment_booking", 3)
    _start(harness, "I need an appointment.", ref, answer=False)
    assert harness.tools.flags[0]["reason"].startswith("REPEAT: ")


@pytest.mark.parametrize("text, band", [
    ("I have crushing chest pain and can’t breathe", "Emergent"),
    ("My stitches came out last night", "Urgent"),
    ("I would like a routine check-up", None),
])
def test_trigger_phrases(text, band):
    assert policies.find_trigger_band(text) == band


def test_emergency_statement_is_given_when_an_emergency_message_cannot_be_parsed():
    harness = Harness(parse_error=RuntimeError("model down"))
    result = _start(harness, "I have crushing chest pain and can't breathe", _find_patient(harness.tools))
    assert result.band == "Emergent" and result.response == f"{EMERGENCY_STATEMENT}\n{DEFLECTION_SCRIPT}"
    assert harness.tools.flags[0]["urgency"] == "Emergent"


def test_trigger_phrase_is_a_floor_under_the_parsed_band():
    harness = Harness(parsed={"urgency": "Routine"})
    result = _start(harness, "The wound opened up and two stitches came out. Can I get a dermatology slot?", _find_patient(harness.tools))
    assert result.band == "Urgent" and result.route == "human_review"
    harness = Harness(parsed={"urgency": "Routine"})
    assert _start(harness, "I get chest pain when I climb stairs, can I see someone?", _find_patient(harness.tools)).route == "refuse"


def test_emergent_is_never_lowered_even_with_a_note():
    harness = Harness(parsed={"urgency": "Emergent"})
    result = _start(harness, "help", _find_patient(harness.tools))
    decision = {"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine", "note": "Felt better."}
    with pytest.raises(ValueError, match=r"3\.3"):
        resume_triage(result.thread_id, decision, harness.dependencies, harness.checkpointer)
    assert get_pending_review(result.thread_id, harness.dependencies, harness.checkpointer) is not None
    assert harness.tools.flags[0]["urgency"] == "Emergent"
    assert policies.validate_reband("Urgent", "Routine", "Nurse triage.") is None
    assert policies.validate_reband("Routine", "Urgent", None) is None


# ---- referrals and repeat topics ---------------------------------------------------------------
def test_rejected_referral_with_denied_preauth_goes_to_staff():
    # Intake record CAREFLOW-00175: MHP-P-00005 holds a rejected Dermatology referral with pre-auth denied.
    harness = Harness(parsed={"specialty": "Dermatology"})
    result = _start(harness, "I need a dermatology appointment.", "MHP-P-00005", answer=False)
    assert result.route == "human_review"
    assert result.plan["referral"]["status"] == "rejected" and result.plan["preauth"]["status"] == "denied"
    assert any("rejected" in reason for reason in result.reasons) and any("denied" in reason for reason in result.reasons)


@pytest.mark.parametrize("ref, specialty", [("MHP-P-00041", "Pulmonology"), ("MHP-P-00212", "ENT")])
def test_unlisted_specialty_with_an_accepted_referral_is_auto(ref, specialty):
    # Intake records CAREFLOW-00057 and CAREFLOW-00154.
    harness = Harness(parsed={"specialty": specialty})
    result = _start(harness, "I need an appointment.", ref, answer=False)
    assert result.plan["referral"]["status"] == "accepted" and (result.route, result.reasons) == ("auto", [])


def test_unlisted_specialty_without_a_referral_still_goes_to_staff():
    harness = Harness(parsed={"specialty": "ENT"})
    result = _start(harness, "I need an appointment.", _find_patient(harness.tools, specialty="ENT"), answer=False)
    assert result.route == "human_review" and "neither" in result.reasons[0]


def test_open_referral_waits_for_triage_acceptance():
    ok = dict(seeks_clinical_advice=False, urgency="Routine", specialty="Cardiology", has_referral=True)
    assert policies.decide_triage_route(**ok, referral_status="accepted")[0] == "auto"
    route, reasons = policies.decide_triage_route(**ok, referral_status="open")
    assert route == "human_review" and "triage acceptance" in reasons[0]


@pytest.mark.parametrize("text, topic", [
    ("Any news on my referral MHP-R-00012?", "referral_status"),
    ("How much will I owe for the scan?", "billing_question"),
    ("Is my plan active?", "eligibility_check"),
    ("Need an ortho appt next week", "appointment_booking"),
])
def test_classify_intake_topic(text, topic):
    assert policies.classify_intake_topic(text) == topic


def test_repeat_rules_use_what_the_message_is_about():
    harness = Harness()
    ref = _find_patient(harness.tools)
    _add_earlier_contacts(harness, ref, "referral_status", 30, 50)
    assert _start(harness, "I need a dermatology appointment.", ref, answer=False).band == "Routine"  # a different issue
    result = _start(harness, "Any update on my referral?", ref, answer=False)
    assert result.band == "Priority" and harness.tools.flags[-1]["reason"].startswith("Repeat - Priority")


# ---- the contact record matches the outcome ----------------------------------------------------
def test_human_review_is_recorded_as_a_callback_with_the_flag_and_the_real_topic():
    harness = Harness(parsed={"urgency": "Urgent"})
    ref = _find_patient(harness.tools)
    result = _start(harness, "I need an appointment.", ref, thread_id="t-urgent")
    row = _find_contact(harness, ref, "t-urgent")
    assert (row["topic"], row["outcome"]) == ("appointment_booking", "callback_scheduled")
    assert result.flag_id in row["summary"] and "(Urgent)" in row["summary"]
    # So the next contact within 24 hours is a REPEAT and keeps the Urgent band (SOP 7.3).
    second_harness = Harness(tools=harness.tools)
    again = _start(second_harness, "Still need that appointment.", ref, answer=False)
    assert again.band == "Urgent" and harness.tools.flags[-1]["reason"].startswith("REPEAT: ")


def test_contact_is_recorded_when_the_assistant_did_not_run():
    for options, setup in (({"answer": False}, None), ({}, "down")):
        harness = Harness(parsed={"urgency": "Urgent"})
        if setup:
            harness.dependencies.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
        ref = _find_patient(harness.tools)
        _start(harness, "I need an appointment.", ref, thread_id="t1", **options)
        assert _find_contact(harness, ref, "t1")["outcome"] == "callback_scheduled"


def test_auto_contact_is_recorded_once_by_the_assistant():
    harness = Harness()
    ref = _find_patient(harness.tools)
    _start(harness, "I need an appointment.", ref, thread_id="t-auto")
    assert _find_contact(harness, ref, "t-auto")["outcome"] == "resolved"


# ---- one case per thread, one store per case ---------------------------------------------------
def test_a_thread_holds_one_case():
    harness = Harness(parsed={"urgency": "Urgent"})
    ref = _find_patient(harness.tools)
    result = _start(harness, "I need an appointment.", ref, thread_id="t1")
    with pytest.raises(AlreadyTriagedError):
        _start(harness, "A different message.", ref, thread_id="t1")
    assert get_pending_review("t1", harness.dependencies, harness.checkpointer)["review_id"] == result.review_id  # the paused case is untouched
    resume_triage("t1", {"reviewer_id": "RN-1", "action": "approve"}, harness.dependencies, harness.checkpointer)
    with pytest.raises(AlreadyTriagedError):
        _start(harness, "And again.", ref, thread_id="t1")
    assert len(harness.tools.flags) == 1


def test_a_paused_case_needs_its_checkpointer():
    harness = Harness(parsed={"urgency": "Urgent"})
    result = _start(harness, "I need an appointment.", _find_patient(harness.tools))
    with pytest.raises(ValueError, match="checkpointer"):
        get_pending_review(result.thread_id, harness.dependencies)
    with pytest.raises(ValueError, match="checkpointer"):
        resume_triage(result.thread_id, {"reviewer_id": "RN-1", "action": "approve"}, harness.dependencies)


def test_default_deps_with_a_checkpointer_share_the_durable_store(tmp_path, monkeypatch):
    import src.store as store_module
    from src.agent import close_checkpointer, open_session_checkpointer

    monkeypatch.setattr(store_module, "RUNTIME_DB", tmp_path / "runtime.db")
    checkpointer = open_session_checkpointer(tmp_path / "sessions.db")
    assert get_pending_review("nobody", checkpointer=checkpointer) is None  # default dependencies open tmp_path/runtime.db
    assert (tmp_path / "runtime.db").exists()
    close_checkpointer(checkpointer)
    close_checkpointer(None)


def test_a_clinical_deflection_by_the_assistant_replaces_the_administrative_reasons():
    steps = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Asked about a dose", urgency="Routine", clinical=True)]),
             _make_message(DEFLECTION_SCRIPT)]
    harness = Harness(parsed={"specialty": "Cardiology"}, steps=steps)
    ref = _find_patient(harness.tools, specialty="Cardiology")
    _add_earlier_contacts(harness, ref, "appointment_booking", 30, 50)  # a third contact: the workflow's band is Priority
    result = _start(harness, "I need an appointment.", ref)
    assert (result.route, result.reasons) == ("refuse", [ASSISTANT_DEFLECTED])
    assert harness.tools.flags[0]["urgency"] == "Priority"  # the ticket still carries the workflow's band


def test_step_limit_flag_is_not_counted_as_the_assistant_asking_for_help():
    loop = _make_message(calls=[_make_call("c1", "check_eligibility", patient_ref="MHP-P-00001")])
    harness = Harness(steps=[loop])
    result = _start(harness, "I need an appointment.", _find_patient(harness.tools))
    assert result.reasons == ["The assistant reached its step limit"] and len(harness.tools.flags) == 1


# ---- ids ---------------------------------------------------------------------------------------
def test_flag_and_review_ids_do_not_collide_after_a_row_is_removed():
    store = RuntimeStore(":memory:")
    for number in (1, 2):
        store.add_flag(None, f"reason {number}", "Routine", "2026-08-01T09:00", contact_id=f"t{number}")
        store.add_review(f"t{number}", "human_review", ["x"], "2026-08-01T09:00")
    store.connection.execute("DELETE FROM staff_flags WHERE flag_id = 'FLAG-0001'")
    store.connection.execute("DELETE FROM reviews WHERE review_id = 'REV-0001'")
    assert store.add_flag(None, "reason 3", "Routine", "2026-08-01T09:00", contact_id="t3")[0]["flag_id"] == "FLAG-0003"
    assert store.add_review("t3", "human_review", ["x"], "2026-08-01T09:00")["review_id"] == "REV-0003"


# ---- the model is not asked about a case the code has decided is Urgent ----
@pytest.mark.parametrize("parsed, text", [
    ({"urgency": "Urgent"}, "I need an appointment."),
    ({"urgency": "Routine"}, "The wound has opened up and two stitches have come out. I need an appointment soon."),
])
def test_an_urgent_case_gets_the_fixed_handoff_and_no_model_call(parsed, text):
    clinical = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Wound", urgency="Urgent", clinical=True)]), _make_message(DEFLECTION_SCRIPT)]
    harness = Harness(parsed=parsed, steps=clinical)  # what the model does when it is asked
    result = _start(harness, text, _find_patient(harness.tools))
    assert (result.route, result.band, result.status) == ("human_review", "Urgent", "awaiting_human_review")
    assert harness.offered == [] and result.response == URGENT_HANDOFF and not result.agent
    assert "SOP 4.8" in result.reasons[0] and harness.tools.booked == {}
    (flag,) = harness.tools.flags
    assert flag["urgency"] == "Urgent" and "SOP 4.8" in flag["reason"]


def test_a_routine_case_held_for_sign_off_still_gets_the_specialists_reply():
    harness = Harness(parsed={"specialty": "Cardiology"})
    result = _start(harness, "I need an appointment.", _find_patient(harness.tools, specialty="Cardiology"))
    assert result.route == "human_review" and harness.offered and result.response == "Here is what I found."
    assert "book_appointment" not in harness.offered[0]
