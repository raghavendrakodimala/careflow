"""Long-term patient memory: recall from the runtime store, privacy, repeat contacts, recording."""

import json
from datetime import datetime, timedelta

import pytest

from src.domain.clock import FROZEN_NOW, FrozenClock
from src.memory import PatientMemory, summarise_turn
from src.store import RuntimeStore


@pytest.fixture()
def store():
    runtime_store = RuntimeStore(":memory:")
    yield runtime_store
    runtime_store.close()


@pytest.fixture()
def memory(store):
    return PatientMemory(store, FrozenClock())


def _parse_time(value):
    return datetime.fromisoformat(value)


def _find_patient(store, predicate):
    for patient in store.list_rows("patients"):
        if predicate(patient["patient_ref"]):
            return patient
    raise AssertionError("no such patient in the seed data")


def test_recall_reads_only_this_patients_rows_newest_first(store, memory):
    counts = {}
    for row in store.list_rows("patient_interactions"):
        counts[row["patient_ref"]] = counts.get(row["patient_ref"], 0) + 1
    ref = max(counts, key=counts.get)
    recalled = memory.recall(ref)
    assert recalled["total_contacts"] == counts[ref] and len(recalled["recent_contacts"]) == min(5, counts[ref])
    times = [contact["occurred_at"] for contact in recalled["recent_contacts"]]
    assert times == sorted(times, reverse=True)
    mine = {row["occurred_at"] for row in store.list_rows("patient_interactions") if row["patient_ref"] == ref}
    assert set(times) <= mine


def test_memory_text_has_no_identifiers_and_no_other_patient(store, memory):
    ref = store.list_rows("patient_interactions")[0]["patient_ref"]
    patient = store.find_one("patients", "patient_ref", ref)
    text = memory.build_context(ref)
    assert text.startswith(f"Patient memory for {ref}")
    for secret in (patient["given_name"], patient["family_name"], patient["birth_date"], patient["member_number"]):
        assert secret not in text
    others = {row["patient_ref"] for row in store.list_rows("patients")} - {ref}
    assert not any(other in text for other in others)


def test_unknown_patient_has_no_memory_and_records_nothing(store, memory):
    before = len(store.list_rows("patient_interactions"))
    assert memory.recall("MHP-P-99999") is None and memory.build_context(None) is None
    assert memory.record_turn("MHP-P-99999", [{"name": "check_eligibility", "arguments": "{}", "result": {"ok": True}}]) is None
    assert len(store.list_rows("patient_interactions")) == before


def test_patient_ref_is_matched_case_insensitively(store, memory):
    ref = store.list_rows("patient_interactions")[0]["patient_ref"]
    assert memory.recall(ref.lower())["patient_ref"] == ref


def test_repeat_contact_within_24_hours(store, memory):
    # Triage SOP 7.3: a contact within 24 hours makes the new one a REPEAT.
    recent = [row for row in store.list_rows("patient_interactions") if FROZEN_NOW - _parse_time(row["occurred_at"]) <= timedelta(hours=24)]
    assert recent, "the seed has contacts in the 24 hours before 2026-08-01 09:00"
    recalled = memory.recall(recent[0]["patient_ref"])
    assert recalled["repeat_contact"] and recent[0]["topic"] in recalled["repeat_topics"]
    assert "it is a REPEAT (triage SOP 7.3)" in memory.build_context(recent[0]["patient_ref"])
    recent_refs = {row["patient_ref"] for row in recent}
    quiet = _find_patient(store, lambda ref: ref not in recent_refs)
    assert memory.recall(quiet["patient_ref"])["repeat_contact"] is False
    assert "REPEAT" not in memory.build_context(quiet["patient_ref"])


def test_future_dated_contacts_are_not_memory_yet(store, memory):
    ref = store.list_rows("patients")[0]["patient_ref"]
    store.add_interaction(ref, "2026-09-01T10:00", "sms", "billing_question", "resolved", "x")
    assert all(contact["occurred_at"] < "2026-09" for contact in memory.recall(ref)["recent_contacts"])


def test_open_referrals_carry_preauth_status_and_sla(store, memory):
    ref_row = next(row for row in store.list_rows("referrals") if row["status"] in ("open", "accepted", "scheduled"))
    recalled = memory.recall(ref_row["patient_ref"])
    details = store.find_one("referral_details", "referral_id", ref_row["referral_id"])
    got = next(referral for referral in recalled["open_referrals"] if referral["referral_id"] == ref_row["referral_id"])
    assert got["preauth_status"] == details["preauth_status"] and got["sla_due_on"] == details["sla_due_on"]
    closed = {row["referral_id"] for row in store.list_rows("referrals") if row["status"] in ("closed", "rejected")}
    assert not closed & {referral["referral_id"] for referral in recalled["open_referrals"]}


def test_upcoming_appointments_are_scheduled_and_in_the_future(store, memory):
    appointment = next(
        candidate for candidate in store.list_rows("appointments") if candidate["status"] == "scheduled" and _parse_time(candidate["scheduled_for"]) >= FROZEN_NOW
    )
    recalled = memory.recall(appointment["patient_ref"])
    assert appointment["appointment_id"] in {upcoming["appointment_id"] for upcoming in recalled["upcoming_appointments"]}
    assert all(_parse_time(upcoming["scheduled_for"]) >= FROZEN_NOW for upcoming in recalled["upcoming_appointments"])


# ---- recording a turn ------------------------------------------------------------------------
def _make_call_record(name, result, **args):
    return {"name": name, "arguments": json.dumps(args), "result": {"ok": True, **result}}


def test_summarise_turn_uses_the_seed_vocabulary_and_no_patient_words():
    booking = summarise_turn([
        _make_call_record("find_appointment_slots", {}, specialty="ENT"),
        _make_call_record("book_appointment", {"appointment_id": "MHP-A-000900", "starts_at": "2026-08-04T09:00", "payment": "insurance"}),
    ])
    assert booking["topic"] == "appointment_booking" and booking["outcome"] == "resolved"
    assert "MHP-A-000900" in booking["summary"]

    clinical = summarise_turn([
        _make_call_record("flag_for_human", {"flag_id": "FLAG-0001", "urgency": "Priority", "category": "clinical"},
           reason="I have chest pain at night", clinical=True),
    ])
    assert clinical["topic"] == "clinical_question" and clinical["outcome"] == "escalated_to_clinician"
    assert "chest" not in clinical["summary"]  # the patient's words are never stored

    coverage = summarise_turn([
        _make_call_record("check_eligibility", {}), _make_call_record("flag_for_human", {"flag_id": "FLAG-0002", "urgency": "Routine"}, reason="x"),
    ])
    assert coverage["topic"] == "eligibility_check" and coverage["outcome"] == "callback_scheduled"

    referral = summarise_turn([_make_call_record("check_preauth", {}, referral_id="mhp-r-00007")])
    assert referral["topic"] == "referral_status" and referral["referral_id"] == "MHP-R-00007"

    assert summarise_turn([_make_call_record("calculate_copay", {})])["topic"] == "billing_question"
    assert summarise_turn([_make_call_record("search_policies", {})])["topic"] == "policy_question"


def test_a_flag_alone_is_not_a_clinical_question():
    # Step-cap and administrative handoffs must not be recorded as clinical.
    from src.graph import ESCALATION_REASON

    failed = {"name": "find_appointment_slots", "arguments": "{}", "result": {"ok": False, "error": "x"}}
    step_cap = summarise_turn([failed, _make_call_record("flag_for_human", {"flag_id": "FLAG-0003", "urgency": "Routine",
                                                             "category": "administrative"}, reason=ESCALATION_REASON)])
    assert step_cap["topic"] == "general_enquiry" and step_cap["outcome"] == "callback_scheduled"
    assert "clinical" not in step_cap["summary"].lower()


def test_contacts_with_no_tool_work_are_still_recorded():
    # A question answered from memory is a contact; naming one of the patient's own
    # referrals makes it a referral_status contact. Someone else's referral ID in the text does not count.
    plain = summarise_turn([])
    assert plain["topic"] == "general_enquiry" and plain["outcome"] == "resolved"
    assert summarise_turn([_make_call_record("find_patient", {"found": True})])["topic"] == "general_enquiry"
    from_memory = summarise_turn([], "Your referral MHP-R-00089 is open.", {"MHP-R-00089"})
    assert from_memory["topic"] == "referral_status" and from_memory["referral_id"] == "MHP-R-00089"
    assert summarise_turn([], "Referral MHP-R-00001 is open.", {"MHP-R-00089"})["topic"] == "general_enquiry"


def test_recorded_turn_persists_and_shows_up_in_memory(tmp_path):
    db_path = tmp_path / "rt.db"
    store = RuntimeStore(db_path)
    ref = store.list_rows("patients")[3]["patient_ref"]
    row = PatientMemory(store).record_turn(ref, [_make_call_record("calculate_copay", {})], channel="sms")
    assert row["interaction_id"].startswith("MHP-I-") and row["occurred_at"] == "2026-08-01T09:00"
    store.close()

    reopened = RuntimeStore(db_path)  # long-term memory survives a restart
    recalled = PatientMemory(reopened).recall(ref)
    assert recalled["recent_contacts"][0]["topic"] == "billing_question" and recalled["repeat_contact"]
    reopened.close()


def test_interaction_ids_continue_the_seed_sequence(store):
    last = max(int(row["interaction_id"][6:]) for row in store.list_rows("patient_interactions"))
    ref = store.list_rows("patients")[0]["patient_ref"]
    row = store.add_interaction(ref, "2026-08-01T09:00", "web_form", "eligibility_check", "resolved", "x")
    assert row["interaction_id"] == f"MHP-I-{last + 1:05d}"


# ---- repeat rules per issue, one contact per conversation -----------------
def test_third_contact_on_the_same_issue_within_72_hours_is_repeat_priority(store, memory):
    # OP-CS-014 8.2. The seed has three such patients on referral_status (e.g. one contact 65 hours ago
    # and one 10 minutes ago), so a contact now is their third.
    hits = {}
    for row in store.list_rows("patient_interactions"):
        if FROZEN_NOW - _parse_time(row["occurred_at"]) <= timedelta(hours=72):
            hits.setdefault((row["patient_ref"], row["topic"]), []).append(row)
    expected = {ref for (ref, _), rows in hits.items() if len(rows) >= 2}
    assert expected, "seed has patients with two same-issue contacts in 72 hours"
    for ref in expected:
        recalled = memory.recall(ref)
        assert recalled["repeat_priority_topics"], ref
        assert "Repeat - Priority (OP-CS-014 8.2)" in memory.build_context(ref)
    one_only = next(ref for (ref, _), rows in hits.items() if len(rows) == 1 and ref not in expected)
    assert memory.recall(one_only)["repeat_priority_topics"] == []


def test_repeat_rules_count_issues_not_contacts(store, memory):
    # Two different issues in 72 hours are not "the same issue" twice; general enquiries are not issues.
    ref = next(patient["patient_ref"] for patient in store.list_rows("patients") if memory.recall(patient["patient_ref"])["total_contacts"] == 0)
    store.add_interaction(ref, "2026-07-31T10:00", "sms", "billing_question", "resolved", "x")
    store.add_interaction(ref, "2026-07-31T11:00", "sms", "referral_status", "resolved", "x")
    store.add_interaction(ref, "2026-07-31T12:00", "sms", "general_enquiry", "resolved", "x")
    store.add_interaction(ref, "2026-07-31T13:00", "sms", "general_enquiry", "resolved", "x")
    recalled = memory.recall(ref)
    assert recalled["repeat_topics"] == ["billing_question", "referral_status"] and recalled["repeat_priority_topics"] == []


def test_a_conversation_is_one_contact_and_does_not_repeat_itself(store, memory):
    ref = next(patient["patient_ref"] for patient in store.list_rows("patients") if memory.recall(patient["patient_ref"])["total_contacts"] == 0)
    memory.record_turn(ref, [], thread_id="call-9")  # "hi": a contact with no issue yet
    memory.record_turn(ref, [_make_call_record("book_appointment", {"appointment_id": "MHP-A-000901", "starts_at": "2026-08-05T10:00",
                                                  "payment": "insurance"})], thread_id="call-9")
    memory.record_turn(ref, [_make_call_record("flag_for_human", {"flag_id": "FLAG-0009", "urgency": "Routine"}, reason="x")],
                    thread_id="call-9")
    rows = store.list_rows_for_patient("patient_interactions", ref)
    assert len(rows) == 1
    row = rows[0]
    assert row["topic"] == "appointment_booking" and row["outcome"] == "callback_scheduled"
    assert "MHP-A-000901" in row["summary"] and "FLAG-0009" in row["summary"]
    assert memory.recall(ref, exclude_thread="call-9")["repeat_topics"] == []  # its own row is not "earlier"
    assert memory.recall(ref, exclude_thread="call-10")["repeat_topics"] == ["appointment_booking"]  # next call: REPEAT
    memory.record_turn(ref, [], thread_id="call-10")
    assert len(store.list_rows_for_patient("patient_interactions", ref)) == 2


def test_policy_turn_summary_says_what_happened():
    from src.memory import summarise_turn

    def search(count):
        return {"name": "search_policies", "arguments": {}, "result": {"ok": True, "count": count}}

    flag = {"name": "flag_for_human", "arguments": {"reason": "x"},
            "result": {"ok": True, "flag_id": "FLAG-0001", "urgency": "Routine", "category": "administrative"}}
    assert summarise_turn([search(2)])["summary"] == "Asked about a clinic policy. Answered from the policy documents."
    assert summarise_turn([search(0)])["summary"] == "Asked about a clinic policy. Not covered by the policy documents."
    withheld = summarise_turn([search(2), flag])
    assert "Answered" not in withheld["summary"] and withheld["outcome"] == "callback_scheduled"


def test_a_repeated_detail_is_stored_once_in_a_conversation_record():
    from src.memory import merge_contact

    flag = "Staff flag FLAG-0001 (Routine)."
    first = {"topic": "appointment_booking", "outcome": "callback_scheduled", "referral_id": None,
             "summary": f"Asked about an appointment. Booked APT-0001. {flag}"}
    again = {"topic": "appointment_booking", "outcome": "callback_scheduled", "referral_id": None,
             "summary": f"Asked about an appointment. {flag}"}
    assert merge_contact(first, again)["summary"].count(flag) == 1
