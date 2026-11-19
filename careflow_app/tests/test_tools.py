"""Offline tests for the M2 tools, against the generated mock tables. No LLM, no network."""

import pytest

from careflow.tools import CareFlowTools


@pytest.fixture()
def t():
    return CareFlowTools()  # fresh in-memory state per test


def _patient(t, **match):
    for p in t.patients.values():
        if all(p[k] == v for k, v in match.items()):
            return p
    pytest.skip(f"no patient with {match} in this dataset")


# ---- co-pay: the toolkit README's worked example and the edges around it -------------------
def test_copay_readme_example_silver_500_met_colonoscopy_is_1480(t):
    p = _patient(t, plan_code="MERIDIAN-SILVER", deductible_met_usd=500, eligibility_status="active")
    r = t.calculate_copay(p["patient_ref"], "GAST-COLO")
    assert r["ok"] and r["patient_owes_usd"] == 1480
    assert r["basis"] == "deductible_then_coinsurance"


def test_copay_flat_for_consultation_imaging_and_telehealth(t):
    p = _patient(t, plan_code="MERIDIAN-SILVER", eligibility_status="active")
    ref = p["patient_ref"]
    assert t.calculate_copay(ref, "CARD-CONS")["patient_owes_usd"] == 45
    assert t.calculate_copay(ref, "CARD-ECHO")["patient_owes_usd"] == 110
    assert t.calculate_copay(ref, "CARD-CONS", telehealth=True)["patient_owes_usd"] == 18


def test_copay_deductible_fully_met_pays_coinsurance_only(t):
    # Deductible met above the plan deductible is treated as fully met (toolkit README).
    p = _patient(t, plan_code="MERIDIAN-GOLD", eligibility_status="active")
    t.patients[p["patient_ref"]] = {**p, "deductible_met_usd": 99999}
    r = t.calculate_copay(p["patient_ref"], "GAST-COLO")
    assert r["patient_owes_usd"] == round(2600 * 0.2)  # 520


def test_copay_never_exceeds_list_price_or_oop_max(t):
    p = _patient(t, plan_code="MERIDIAN-BRONZE", eligibility_status="active")
    t.patients[p["patient_ref"]] = {**p, "deductible_met_usd": 0}
    r = t.calculate_copay(p["patient_ref"], "GAST-COLO")
    assert r["patient_owes_usd"] == 2600  # deductible 3000 > price: pays the whole price
    t.plans["MERIDIAN-BRONZE"] = {**t.plans["MERIDIAN-BRONZE"], "out_of_pocket_max_usd": 1000}
    assert t.calculate_copay(p["patient_ref"], "GAST-COLO")["patient_owes_usd"] == 1000


def test_copay_inactive_coverage_gives_self_pay_price_not_an_insurance_estimate(t):
    p = _patient(t, eligibility_status="lapsed")
    r = t.calculate_copay(p["patient_ref"], "CARD-CONS")
    assert r["ok"] and r["patient_owes_usd"] is None and "lapsed" in r["note"]
    assert r["self_pay_available"] and r["self_pay_usd"] == 220  # list price (preauth matrix 2.5)
    assert "check-in" in r["payment_terms"]


def test_copay_without_insurance_quotes_self_pay(t):
    r = t.calculate_copay(procedure_code="gast-colo")
    assert r["ok"] and r["self_pay_usd"] == 2600 and r["patient_owes_usd"] is None
    assert t.calculate_copay("NOPE", "GAST-COLO")["self_pay_usd"] == 2600  # unknown patient: still told the price


def test_copay_not_covered_procedure_has_no_amount_but_a_self_pay_price(t):
    (proc, plan), _ = next(((k, v) for k, v in t.preauth.items() if not v["covered"]), (None, None))
    p = _patient(t, plan_code=plan, eligibility_status="active")
    r = t.calculate_copay(p["patient_ref"], proc)
    assert r["covered"] is False and r["patient_owes_usd"] is None
    assert r["self_pay_usd"] == t.procedures[proc]["list_price_usd"]


def test_copay_unknown_inputs_are_errors_not_exceptions(t):
    assert t.calculate_copay("NOPE", "CARD-CONS")["ok"] is False
    p = _patient(t, eligibility_status="active")
    r = t.calculate_copay(p["patient_ref"], "XXXX")
    assert r["ok"] is False and "valid_codes" in r


# ---- eligibility ----------------------------------------------------------------------------
def test_eligibility_by_ref_and_by_member_number_agree(t):
    p = _patient(t, eligibility_status="active")
    a = t.check_eligibility(patient_ref=p["patient_ref"])
    b = t.check_eligibility(member_number=p["member_number"].lower())
    assert a == b and a["eligible"] is True and a["plan_code"] == p["plan_code"]


def test_eligibility_statuses_and_unknowns(t):
    for status in ("lapsed", "pending_verification"):
        r = t.check_eligibility(patient_ref=_patient(t, eligibility_status=status)["patient_ref"])
        assert r["eligible"] is False and r["eligibility_status"] == status
    assert t.check_eligibility(patient_ref="NOPE") == {"ok": False, "error": "no matching patient on record", "found": False}
    assert t.check_eligibility()["ok"] is False


def test_eligibility_does_not_return_name_or_birth_date(t):
    r = t.check_eligibility(patient_ref=_patient(t, eligibility_status="active")["patient_ref"])
    assert not {"given_name", "family_name", "birth_date"} & set(r)


# ---- appointment finder and booking ---------------------------------------------------------
def test_slots_are_open_correct_specialty_sorted_and_future(t):
    r = t.find_appointment_slots("Cardiology", limit=20)
    assert r["ok"] and r["count"] > 0
    times = [s["starts_at"] for s in r["slots"]]
    assert times == sorted(times) and times[0] >= "2026-08-01T09:00"
    for s in r["slots"]:
        assert t.slots[s["slot_id"]]["specialty"] == "Cardiology" and s["provider"].startswith("Dr.")


def test_slots_telehealth_filter_and_earliest_date(t):
    r = t.find_appointment_slots("Dermatology", mode="telehealth", limit=20)
    assert all(s["mode"] == "telehealth" for s in r["slots"])
    late = t.find_appointment_slots("Cardiology", earliest_date="2026-08-10", limit=20)
    assert all(s["starts_at"] >= "2026-08-10" for s in late["slots"])


def test_slots_bad_input(t):
    assert "valid_specialties" in t.find_appointment_slots("Podiatry")
    assert t.find_appointment_slots("ENT", mode="video")["ok"] is False
    assert t.find_appointment_slots("ENT", earliest_date="soon")["ok"] is False


def test_booking_removes_slot_and_cannot_double_book(t):
    p = _patient(t, eligibility_status="active")
    slot = t.find_appointment_slots("ENT", patient_ref=p["patient_ref"])["slots"][0]["slot_id"]
    assert t.book_appointment(p["patient_ref"], slot)["ok"]
    assert slot not in [s["slot_id"] for s in t.find_appointment_slots("ENT", limit=20)["slots"]]
    again = t.book_appointment(p["patient_ref"], slot)
    assert again["ok"] is False and "no longer available" in again["error"]


def test_booking_refused_without_active_coverage_and_unknowns(t):
    lapsed = _patient(t, eligibility_status="lapsed")
    slot = t.find_appointment_slots("ENT")["slots"][0]["slot_id"]
    r = t.book_appointment(lapsed["patient_ref"], slot)
    assert r["ok"] is False and slot not in t.booked
    assert t.book_appointment("NOPE", slot)["ok"] is False
    assert t.book_appointment(lapsed["patient_ref"], "NOPE")["ok"] is False


# ---- payment rule (owner decision 2026-11-18): self-pay when insurance cannot be used --------------
def _open_slot_for(t, ref, specialty="ENT"):
    return t.find_appointment_slots(specialty, patient_ref=ref)["slots"][0]["slot_id"]


def test_lapsed_patient_is_offered_self_pay_not_refused_care(t):
    ref = _patient(t, eligibility_status="lapsed")["patient_ref"]
    slot = _open_slot_for(t, ref)
    r = t.book_appointment(ref, slot)
    assert r["ok"] is False and slot not in t.booked  # nothing booked without the patient's agreement
    assert r["self_pay_available"] is True and r["self_pay_usd"] == t._visit_price("ENT")
    assert "self_pay=true" in r["error"] and "check-in" in r["payment_terms"]
    assert t.flags == []


def test_lapsed_patient_who_agrees_is_booked_as_self_pay_and_staff_are_told(t):
    ref = _patient(t, eligibility_status="lapsed")["patient_ref"]
    slot = _open_slot_for(t, ref)
    r = t.book_appointment(ref, slot, self_pay=True)
    assert r["ok"] and r["payment"] == "self_pay" and r["self_pay_usd"] == t._visit_price("ENT")
    assert t.booked[slot] == ref
    assert [f["flag_id"] for f in t.flags] == [r["flag_id"]] and "Self-pay" in t.flags[0]["reason"]


def test_pending_verification_is_booked_and_flagged_for_verification(t):
    ref = _patient(t, eligibility_status="pending_verification")["patient_ref"]
    slot = _open_slot_for(t, ref)
    r = t.book_appointment(ref, slot)
    assert r["ok"] and r["payment"] == "insurance_pending_verification" and "verif" in r["note"]
    assert "Verify coverage" in t.flags[0]["reason"] and t.flags[0]["flag_id"] == r["flag_id"]
    assert t.calculate_copay(ref, "CARD-CONS")["patient_owes_usd"] is None  # still no insurance quote


def test_active_patient_books_on_insurance_without_a_flag(t):
    ref = _patient(t, eligibility_status="active")["patient_ref"]
    r = t.book_appointment(ref, _open_slot_for(t, ref))
    assert r["ok"] and r["payment"] == "insurance" and "flag_id" not in r and t.flags == []


def test_every_specialty_has_a_self_pay_visit_price(t):
    assert all(isinstance(t._visit_price(s), int) for s in t.specialties)


def test_state_is_per_instance():
    a, b = CareFlowTools(), CareFlowTools()
    p = next(p for p in a.patients.values() if p["eligibility_status"] == "active")
    slot = a.find_appointment_slots("ENT", patient_ref=p["patient_ref"])["slots"][0]["slot_id"]
    assert a.book_appointment(p["patient_ref"], slot)["ok"]
    assert a.slots[slot]["status"] == "booked"
    assert slot in [s["slot_id"] for s in b.find_appointment_slots("ENT", patient_ref=p["patient_ref"], limit=20)["slots"]]


def test_injected_clock_moves_today(t):
    from datetime import datetime

    from careflow.domain.clock import FrozenClock

    # Slots in the data run 3-15 August 2026.
    later = CareFlowTools(store=t.store, clock=FrozenClock(datetime(2026, 8, 12, 9, 0)))
    slots = later.find_appointment_slots("ENT", limit=20)["slots"]
    assert slots and all(s["starts_at"] >= "2026-08-12T09:00" for s in slots)
    assert t.find_appointment_slots("ENT")["slots"][0]["starts_at"] < "2026-08-12"
    assert later.flag_for_human("x")["created_at"] == "2026-08-12T09:00"


def test_timezone_aware_earliest_date_is_accepted_not_a_crash(t):
    # Audit A3: a "...Z" or "+05:30" date used to raise TypeError (naive vs aware comparison).
    for value in ("2026-08-05T09:00:00Z", "2026-08-05T09:00:00+05:30"):
        r = t.find_appointment_slots("ENT", earliest_date=value, limit=20)
        assert r["ok"] and all(s["starts_at"] >= "2026-08-05" for s in r["slots"])
    assert t.call("find_appointment_slots", {"specialty": "ENT", "earliest_date": "2026-08-05T09:00:00Z"})["ok"]


def test_ids_and_codes_are_case_insensitive(t):
    # Audit A5: "gast-colo" and a lower-case patient ref used to be rejected.
    p = _patient(t, plan_code="MERIDIAN-SILVER", deductible_met_usd=500, eligibility_status="active")
    assert t.calculate_copay(p["patient_ref"].lower(), " gast-colo ")["patient_owes_usd"] == 1480
    assert t.find_appointment_slots("ent")["specialty"] == "ENT"
    slot = t.find_appointment_slots("ENT", patient_ref=p["patient_ref"])["slots"][0]["slot_id"]
    assert t.book_appointment(p["patient_ref"].lower(), slot.lower())["ok"]
    assert t.flag_for_human("x", "Routine", p["patient_ref"].lower())["patient_ref"] == p["patient_ref"]


# ---- new patients (audit A5) ------------------------------------------------------------------
def _closed_slot_and_patients(t):
    """An open slot whose provider is not accepting new patients, plus a new and an established patient."""
    slot = next(
        s["slot_id"]
        for s in sorted(t.slots.values(), key=lambda r: r["starts_at"])
        if s["status"] == "open" and s["starts_at"] >= "2026-08-01T09:00"
        and not t.providers[s["provider_id"]]["accepts_new_patients"]
    )
    active = [p["patient_ref"] for p in t.patients.values() if p["eligibility_status"] == "active"]
    new = next(r for r in active if r not in t.established)
    old = next(r for r in active if r in t.established)
    return slot, new, old


def test_new_patient_cannot_book_a_provider_not_accepting_new_patients(t):
    slot, new, old = _closed_slot_and_patients(t)
    r = t.book_appointment(new, slot)
    assert r["ok"] is False and "not accepting new patients" in r["error"] and slot not in t.booked
    assert t.book_appointment(old, slot)["ok"]  # an established patient can


def test_explicit_new_patient_override_wins_over_the_inferred_value(t):
    slot, new, _ = _closed_slot_and_patients(t)
    assert t.book_appointment(new, slot, new_patient=False)["ok"]


def test_slot_search_hides_closed_providers_for_a_new_patient_only(t):
    slot, new, old = _closed_slot_and_patients(t)
    spec = t.slots[slot]["specialty"]
    for_new = t.find_appointment_slots(spec, patient_ref=new, limit=20)["slots"]
    assert for_new and all(s["accepts_new_patients"] for s in for_new)
    for_old = t.find_appointment_slots(spec, patient_ref=old, limit=20)["slots"]
    assert any(not s["accepts_new_patients"] for s in for_old)
    assert t.find_appointment_slots(spec, new_patient=True, limit=20)["slots"] == for_new


def test_only_completed_past_visits_make_a_patient_established(t):
    # Audit 5.2.3: cancelled, no-show and future rows must not count.
    rows = t.store.rows("appointments")
    by_patient: dict[str, set] = {}
    for r in rows:
        past = r["scheduled_for"] < "2026-08-01T09:00"
        by_patient.setdefault(r["patient_ref"], set()).add((r["status"], past))
    not_completed = next(ref for ref, s in by_patient.items() if not any(st == "completed" and past for st, past in s))
    assert t.is_new_patient(not_completed)
    completed = next(ref for ref, s in by_patient.items() if ("completed", True) in s)
    assert not t.is_new_patient(completed)
    assert t.check_eligibility(patient_ref=completed)["new_patient"] is False


# ---- flag for human -------------------------------------------------------------------------
def test_flag_records_validates_and_is_idempotent(t):
    p = _patient(t, eligibility_status="active")
    first = t.flag_for_human("asks about a dosing change", "Urgent", p["patient_ref"])
    assert first["ok"] and first["duplicate"] is False and first["status"] == "awaiting_staff"
    second = t.flag_for_human("asks about a dosing change", "Urgent", p["patient_ref"])
    assert second["duplicate"] is True and second["flag_id"] == first["flag_id"] and len(t.flags) == 1
    assert t.flag_for_human("", "Urgent")["ok"] is False
    assert t.flag_for_human("x", "Critical")["ok"] is False
    assert t.flag_for_human("x", "Routine", "NOPE")["ok"] is False


# ---- dispatch: what the model sends -----------------------------------------------------------
def test_call_accepts_json_string_and_survives_bad_arguments(t):
    p = _patient(t, eligibility_status="active")
    assert t.call("check_eligibility", f'{{"patient_ref": "{p["patient_ref"]}"}}')["ok"]
    assert t.call("check_eligibility", "{not json")["ok"] is False
    assert "bad arguments" in t.call("calculate_copay", {"patient": "x"})["error"]
    assert "unknown tool" in t.call("drop_tables", {})["error"]
    assert t.call("flag_for_human", None)["ok"] is False  # missing required reason
