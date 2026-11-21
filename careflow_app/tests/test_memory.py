"""M3 long-term patient memory: recall from the runtime store, privacy, repeat contacts, recording."""

import json
from datetime import datetime, timedelta

import pytest

from careflow.domain.clock import FROZEN_NOW, FrozenClock
from careflow.memory import PatientMemory, summarise_turn
from careflow.store import RuntimeStore


@pytest.fixture()
def store():
    s = RuntimeStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def mem(store):
    return PatientMemory(store, FrozenClock())


def _ts(v):
    return datetime.fromisoformat(v)


def _patient_with(store, predicate):
    for p in store.rows("patients"):
        if predicate(p["patient_ref"]):
            return p
    raise AssertionError("no such patient in the seed data")


def test_recall_reads_only_this_patients_rows_newest_first(store, mem):
    counts = {}
    for r in store.rows("patient_interactions"):
        counts[r["patient_ref"]] = counts.get(r["patient_ref"], 0) + 1
    ref = max(counts, key=counts.get)
    m = mem.recall(ref)
    assert m["total_contacts"] == counts[ref] and len(m["recent_contacts"]) == min(5, counts[ref])
    times = [c["occurred_at"] for c in m["recent_contacts"]]
    assert times == sorted(times, reverse=True)
    mine = {r["occurred_at"] for r in store.rows("patient_interactions") if r["patient_ref"] == ref}
    assert set(times) <= mine


def test_memory_text_has_no_identifiers_and_no_other_patient(store, mem):
    ref = store.rows("patient_interactions")[0]["patient_ref"]
    p = store.one("patients", "patient_ref", ref)
    text = mem.context(ref)
    assert text.startswith(f"Patient memory for {ref}")
    for secret in (p["given_name"], p["family_name"], p["birth_date"], p["member_number"]):
        assert secret not in text
    others = {r["patient_ref"] for r in store.rows("patients")} - {ref}
    assert not any(o in text for o in others)


def test_unknown_patient_has_no_memory_and_records_nothing(store, mem):
    before = len(store.rows("patient_interactions"))
    assert mem.recall("MHP-P-99999") is None and mem.context(None) is None
    assert mem.record_turn("MHP-P-99999", [{"name": "check_eligibility", "arguments": "{}", "result": {"ok": True}}]) is None
    assert len(store.rows("patient_interactions")) == before


def test_patient_ref_is_matched_case_insensitively(store, mem):
    ref = store.rows("patient_interactions")[0]["patient_ref"]
    assert mem.recall(ref.lower())["patient_ref"] == ref


def test_repeat_contact_within_24_hours(store, mem):
    # Triage SOP 7.3: a contact within 24 hours makes the new one a REPEAT.
    recent = [r for r in store.rows("patient_interactions") if FROZEN_NOW - _ts(r["occurred_at"]) <= timedelta(hours=24)]
    assert recent, "the seed has contacts in the 24 hours before 2026-08-01 09:00"
    m = mem.recall(recent[0]["patient_ref"])
    assert m["repeat_contact"] and recent[0]["topic"] in m["repeat_topics"]
    assert "it is a REPEAT (triage SOP 7.3)" in mem.context(recent[0]["patient_ref"])
    recent_refs = {r["patient_ref"] for r in recent}
    quiet = _patient_with(store, lambda ref: ref not in recent_refs)
    assert mem.recall(quiet["patient_ref"])["repeat_contact"] is False
    assert "REPEAT" not in mem.context(quiet["patient_ref"])


def test_future_dated_contacts_are_not_memory_yet(store, mem):
    ref = store.rows("patients")[0]["patient_ref"]
    store.add_interaction(ref, "2026-09-01T10:00", "sms", "billing_question", "resolved", "x")
    assert all(c["occurred_at"] < "2026-09" for c in mem.recall(ref)["recent_contacts"])


def test_open_referrals_carry_preauth_status_and_sla(store, mem):
    ref_row = next(r for r in store.rows("referrals") if r["status"] in ("open", "accepted", "scheduled"))
    m = mem.recall(ref_row["patient_ref"])
    details = store.one("referral_details", "referral_id", ref_row["referral_id"])
    got = next(r for r in m["open_referrals"] if r["referral_id"] == ref_row["referral_id"])
    assert got["preauth_status"] == details["preauth_status"] and got["sla_due_on"] == details["sla_due_on"]
    closed = {r["referral_id"] for r in store.rows("referrals") if r["status"] in ("closed", "rejected")}
    assert not closed & {r["referral_id"] for r in m["open_referrals"]}


def test_upcoming_appointments_are_scheduled_and_in_the_future(store, mem):
    a = next(
        a for a in store.rows("appointments") if a["status"] == "scheduled" and _ts(a["scheduled_for"]) >= FROZEN_NOW
    )
    m = mem.recall(a["patient_ref"])
    assert a["appointment_id"] in {u["appointment_id"] for u in m["upcoming_appointments"]}
    assert all(_ts(u["scheduled_for"]) >= FROZEN_NOW for u in m["upcoming_appointments"])


# ---- recording a turn ------------------------------------------------------------------------
def _c(name, result, **args):
    return {"name": name, "arguments": json.dumps(args), "result": {"ok": True, **result}}


def test_summarise_turn_uses_the_seed_vocabulary_and_no_patient_words():
    booking = summarise_turn([
        _c("find_appointment_slots", {}, specialty="ENT"),
        _c("book_appointment", {"appointment_id": "MHP-A-000900", "starts_at": "2026-08-04T09:00", "payment": "insurance"}),
    ])
    assert booking["topic"] == "appointment_booking" and booking["outcome"] == "resolved"
    assert "MHP-A-000900" in booking["summary"]

    clinical = summarise_turn([
        _c("flag_for_human", {"flag_id": "FLAG-0001", "urgency": "Priority", "category": "clinical"},
           reason="I have chest pain at night", clinical=True),
    ])
    assert clinical["topic"] == "clinical_question" and clinical["outcome"] == "escalated_to_clinician"
    assert "chest" not in clinical["summary"]  # the patient's words are never stored

    coverage = summarise_turn([
        _c("check_eligibility", {}), _c("flag_for_human", {"flag_id": "FLAG-0002", "urgency": "Routine"}, reason="x"),
    ])
    assert coverage["topic"] == "eligibility_check" and coverage["outcome"] == "callback_scheduled"

    referral = summarise_turn([_c("check_preauth", {}, referral_id="mhp-r-00007")])
    assert referral["topic"] == "referral_status" and referral["referral_id"] == "MHP-R-00007"

    assert summarise_turn([_c("calculate_copay", {})])["topic"] == "billing_question"
    assert summarise_turn([_c("search_policies", {})])["topic"] == "policy_question"  # feedback M3-1


def test_a_flag_alone_is_not_a_clinical_question():
    # Feedback M3-1: step-cap and administrative handoffs must not be recorded as clinical.
    from careflow.graph import ESCALATION_REASON

    failed = {"name": "find_appointment_slots", "arguments": "{}", "result": {"ok": False, "error": "x"}}
    step_cap = summarise_turn([failed, _c("flag_for_human", {"flag_id": "FLAG-0003", "urgency": "Routine",
                                                             "category": "administrative"}, reason=ESCALATION_REASON)])
    assert step_cap["topic"] == "general_enquiry" and step_cap["outcome"] == "callback_scheduled"
    assert "clinical" not in step_cap["summary"].lower()


def test_contacts_with_no_tool_work_are_still_recorded():
    # Feedback M3-2: a question answered from memory is a contact; naming one of the patient's own
    # referrals makes it a referral_status contact. Someone else's referral ID in the text does not count.
    plain = summarise_turn([])
    assert plain["topic"] == "general_enquiry" and plain["outcome"] == "resolved"
    assert summarise_turn([_c("find_patient", {"found": True})])["topic"] == "general_enquiry"
    from_memory = summarise_turn([], "Your referral MHP-R-00089 is open.", {"MHP-R-00089"})
    assert from_memory["topic"] == "referral_status" and from_memory["referral_id"] == "MHP-R-00089"
    assert summarise_turn([], "Referral MHP-R-00001 is open.", {"MHP-R-00089"})["topic"] == "general_enquiry"


def test_recorded_turn_persists_and_shows_up_in_memory(tmp_path):
    db = tmp_path / "rt.db"
    s = RuntimeStore(db)
    ref = s.rows("patients")[3]["patient_ref"]
    row = PatientMemory(s).record_turn(ref, [_c("calculate_copay", {})], channel="sms")
    assert row["interaction_id"].startswith("MHP-I-") and row["occurred_at"] == "2026-08-01T09:00"
    s.close()

    reopened = RuntimeStore(db)  # long-term memory survives a restart
    m = PatientMemory(reopened).recall(ref)
    assert m["recent_contacts"][0]["topic"] == "billing_question" and m["repeat_contact"]
    reopened.close()


def test_interaction_ids_continue_the_seed_sequence(store):
    last = max(int(r["interaction_id"][6:]) for r in store.rows("patient_interactions"))
    ref = store.rows("patients")[0]["patient_ref"]
    row = store.add_interaction(ref, "2026-08-01T09:00", "web_form", "eligibility_check", "resolved", "x")
    assert row["interaction_id"] == f"MHP-I-{last + 1:05d}"


# ---- repeat rules per issue, one contact per conversation (feedback M3-2, M3-3) -----------------
def test_third_contact_on_the_same_issue_within_72_hours_is_repeat_priority(store, mem):
    # OP-CS-014 8.2. The seed has three such patients on referral_status (e.g. one contact 65 hours ago
    # and one 10 minutes ago), so a contact now is their third.
    hits = {}
    for r in store.rows("patient_interactions"):
        if FROZEN_NOW - _ts(r["occurred_at"]) <= timedelta(hours=72):
            hits.setdefault((r["patient_ref"], r["topic"]), []).append(r)
    expected = {ref for (ref, _), rows in hits.items() if len(rows) >= 2}
    assert expected, "seed has patients with two same-issue contacts in 72 hours"
    for ref in expected:
        m = mem.recall(ref)
        assert m["repeat_priority_topics"], ref
        assert "Repeat - Priority (OP-CS-014 8.2)" in mem.context(ref)
    one_only = next(ref for (ref, _), rows in hits.items() if len(rows) == 1 and ref not in expected)
    assert mem.recall(one_only)["repeat_priority_topics"] == []


def test_repeat_rules_count_issues_not_contacts(store, mem):
    # Two different issues in 72 hours are not "the same issue" twice; general enquiries are not issues.
    ref = next(p["patient_ref"] for p in store.rows("patients") if mem.recall(p["patient_ref"])["total_contacts"] == 0)
    store.add_interaction(ref, "2026-07-31T10:00", "sms", "billing_question", "resolved", "x")
    store.add_interaction(ref, "2026-07-31T11:00", "sms", "referral_status", "resolved", "x")
    store.add_interaction(ref, "2026-07-31T12:00", "sms", "general_enquiry", "resolved", "x")
    store.add_interaction(ref, "2026-07-31T13:00", "sms", "general_enquiry", "resolved", "x")
    m = mem.recall(ref)
    assert m["repeat_topics"] == ["billing_question", "referral_status"] and m["repeat_priority_topics"] == []


def test_a_conversation_is_one_contact_and_does_not_repeat_itself(store, mem):
    ref = next(p["patient_ref"] for p in store.rows("patients") if mem.recall(p["patient_ref"])["total_contacts"] == 0)
    mem.record_turn(ref, [], thread_id="call-9")  # "hi": a contact with no issue yet
    mem.record_turn(ref, [_c("book_appointment", {"appointment_id": "MHP-A-000901", "starts_at": "2026-08-05T10:00",
                                                  "payment": "insurance"})], thread_id="call-9")
    mem.record_turn(ref, [_c("flag_for_human", {"flag_id": "FLAG-0009", "urgency": "Routine"}, reason="x")],
                    thread_id="call-9")
    rows = store.rows_for_patient("patient_interactions", ref)
    assert len(rows) == 1
    row = rows[0]
    assert row["topic"] == "appointment_booking" and row["outcome"] == "callback_scheduled"
    assert "MHP-A-000901" in row["summary"] and "FLAG-0009" in row["summary"]
    assert mem.recall(ref, exclude_thread="call-9")["repeat_topics"] == []  # its own row is not "earlier"
    assert mem.recall(ref, exclude_thread="call-10")["repeat_topics"] == ["appointment_booking"]  # next call: REPEAT
    mem.record_turn(ref, [], thread_id="call-10")
    assert len(store.rows_for_patient("patient_interactions", ref)) == 2


def test_policy_turn_summary_says_what_happened():  # feedback M4-1b
    from careflow.memory import summarise_turn

    def search(count):
        return {"name": "search_policies", "arguments": {}, "result": {"ok": True, "count": count}}

    flag = {"name": "flag_for_human", "arguments": {"reason": "x"},
            "result": {"ok": True, "flag_id": "FLAG-0001", "urgency": "Routine", "category": "administrative"}}
    assert summarise_turn([search(2)])["summary"] == "Asked about a clinic policy. Answered from the policy documents."
    assert summarise_turn([search(0)])["summary"] == "Asked about a clinic policy. Not covered by the policy documents."
    withheld = summarise_turn([search(2), flag])
    assert "Answered" not in withheld["summary"] and withheld["outcome"] == "callback_scheduled"
