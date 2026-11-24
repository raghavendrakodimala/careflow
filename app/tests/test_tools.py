"""Offline tests for the clinic tools, against the generated mock tables. No LLM, no network."""

import pytest

from src.tools import CareFlowTools


@pytest.fixture()
def tools():
    return CareFlowTools()  # fresh in-memory state per test


def _find_patient(tools, **match):
    for patient in tools.patients.values():
        if all(patient[key] == value for key, value in match.items()):
            return patient
    pytest.skip(f"no patient with {match} in this dataset")


# ---- co-pay: the toolkit README's worked example and the edges around it -------------------
def test_copay_readme_example_silver_500_met_colonoscopy_is_1480(tools):
    patient = _find_patient(tools, plan_code="MERIDIAN-SILVER", deductible_met_usd=500, eligibility_status="active")
    result = tools.calculate_copay(patient["patient_ref"], "GAST-COLO")
    assert result["ok"] and result["patient_owes_usd"] == 1480
    assert result["basis"] == "deductible_then_coinsurance"


def test_copay_flat_for_consultation_imaging_and_telehealth(tools):
    patient = _find_patient(tools, plan_code="MERIDIAN-SILVER", eligibility_status="active")
    ref = patient["patient_ref"]
    assert tools.calculate_copay(ref, "CARD-CONS")["patient_owes_usd"] == 45
    assert tools.calculate_copay(ref, "CARD-ECHO")["patient_owes_usd"] == 110
    assert tools.calculate_copay(ref, "CARD-CONS", telehealth=True)["patient_owes_usd"] == 18


def test_copay_deductible_fully_met_pays_coinsurance_only(tools):
    # Deductible met above the plan deductible is treated as fully met (toolkit README).
    patient = _find_patient(tools, plan_code="MERIDIAN-GOLD", eligibility_status="active")
    tools.patients[patient["patient_ref"]] = {**patient, "deductible_met_usd": 99999}
    result = tools.calculate_copay(patient["patient_ref"], "GAST-COLO")
    assert result["patient_owes_usd"] == round(2600 * 0.2)  # 520


def test_copay_never_exceeds_list_price_or_oop_max(tools):
    patient = _find_patient(tools, plan_code="MERIDIAN-BRONZE", eligibility_status="active")
    tools.patients[patient["patient_ref"]] = {**patient, "deductible_met_usd": 0}
    result = tools.calculate_copay(patient["patient_ref"], "GAST-COLO")
    assert result["patient_owes_usd"] == 2600  # deductible 3000 > price: pays the whole price
    tools.plans["MERIDIAN-BRONZE"] = {**tools.plans["MERIDIAN-BRONZE"], "out_of_pocket_max_usd": 1000}
    assert tools.calculate_copay(patient["patient_ref"], "GAST-COLO")["patient_owes_usd"] == 1000


def test_copay_inactive_coverage_gives_self_pay_price_not_an_insurance_estimate(tools):
    patient = _find_patient(tools, eligibility_status="lapsed")
    result = tools.calculate_copay(patient["patient_ref"], "CARD-CONS")
    assert result["ok"] and result["patient_owes_usd"] is None and "lapsed" in result["note"]
    assert result["self_pay_available"] and result["self_pay_usd"] == 220  # list price (preauth matrix 2.5)
    assert "check-in" in result["payment_terms"]


def test_copay_without_insurance_quotes_self_pay(tools):
    result = tools.calculate_copay(procedure_code="gast-colo")
    assert result["ok"] and result["self_pay_usd"] == 2600 and result["patient_owes_usd"] is None
    assert tools.calculate_copay("NOPE", "GAST-COLO")["self_pay_usd"] == 2600  # unknown patient: still told the price


def test_copay_not_covered_procedure_has_no_amount_but_a_self_pay_price(tools):
    (proc, plan), _ = next(((key, value) for key, value in tools.preauth.items() if not value["covered"]), (None, None))
    patient = _find_patient(tools, plan_code=plan, eligibility_status="active")
    result = tools.calculate_copay(patient["patient_ref"], proc)
    assert result["covered"] is False and result["patient_owes_usd"] is None
    assert result["self_pay_usd"] == tools.procedures[proc]["list_price_usd"]


def test_copay_unknown_inputs_are_errors_not_exceptions(tools):
    assert tools.calculate_copay("NOPE", "CARD-CONS")["ok"] is False
    patient = _find_patient(tools, eligibility_status="active")
    result = tools.calculate_copay(patient["patient_ref"], "XXXX")
    assert result["ok"] is False and "valid_codes" in result


# ---- eligibility ----------------------------------------------------------------------------
def test_eligibility_by_ref_and_by_member_number_agree(tools):
    patient = _find_patient(tools, eligibility_status="active")
    by_ref = tools.check_eligibility(patient_ref=patient["patient_ref"])
    by_member = tools.check_eligibility(member_number=patient["member_number"].lower())
    assert by_ref == by_member and by_ref["eligible"] is True and by_ref["plan_code"] == patient["plan_code"]


def test_eligibility_statuses_and_unknowns(tools):
    for status in ("lapsed", "pending_verification"):
        result = tools.check_eligibility(patient_ref=_find_patient(tools, eligibility_status=status)["patient_ref"])
        assert result["eligible"] is False and result["eligibility_status"] == status
    assert tools.check_eligibility(patient_ref="NOPE") == {"ok": False, "error": "no matching patient on record", "found": False}
    assert tools.check_eligibility()["ok"] is False


def test_eligibility_does_not_return_name_or_birth_date(tools):
    result = tools.check_eligibility(patient_ref=_find_patient(tools, eligibility_status="active")["patient_ref"])
    assert not {"given_name", "family_name", "birth_date"} & set(result)


# ---- appointment finder and booking ---------------------------------------------------------
def test_slots_are_open_correct_specialty_sorted_and_future(tools):
    result = tools.find_appointment_slots("Cardiology", limit=20)
    assert result["ok"] and result["count"] > 0
    times = [slot["starts_at"] for slot in result["slots"]]
    assert times == sorted(times) and times[0] >= "2026-08-01T09:00"
    for slot in result["slots"]:
        assert tools.slots[slot["slot_id"]]["specialty"] == "Cardiology" and slot["provider"].startswith("Dr.")


def test_slots_telehealth_filter_and_earliest_date(tools):
    result = tools.find_appointment_slots("Dermatology", mode="telehealth", limit=20)
    assert all(slot["mode"] == "telehealth" for slot in result["slots"])
    late = tools.find_appointment_slots("Cardiology", earliest_date="2026-08-10", limit=20)
    assert all(slot["starts_at"] >= "2026-08-10" for slot in late["slots"])


def test_slots_bad_input(tools):
    assert "valid_specialties" in tools.find_appointment_slots("Podiatry")
    assert tools.find_appointment_slots("ENT", mode="video")["ok"] is False
    assert tools.find_appointment_slots("ENT", earliest_date="soon")["ok"] is False


def test_booking_removes_slot_and_cannot_double_book(tools):
    patient = _find_patient(tools, eligibility_status="active")
    slot = tools.find_appointment_slots("ENT", patient_ref=patient["patient_ref"])["slots"][0]["slot_id"]
    assert tools.book_appointment(patient["patient_ref"], slot)["ok"]
    assert slot not in [offered["slot_id"] for offered in tools.find_appointment_slots("ENT", limit=20)["slots"]]
    again = tools.book_appointment(patient["patient_ref"], slot)
    assert again["ok"] is False and "no longer available" in again["error"]


def test_booking_refused_without_active_coverage_and_unknowns(tools):
    lapsed = _find_patient(tools, eligibility_status="lapsed")
    slot = tools.find_appointment_slots("ENT")["slots"][0]["slot_id"]
    result = tools.book_appointment(lapsed["patient_ref"], slot)
    assert result["ok"] is False and slot not in tools.booked
    assert tools.book_appointment("NOPE", slot)["ok"] is False
    assert tools.book_appointment(lapsed["patient_ref"], "NOPE")["ok"] is False


# ---- payment rule: self-pay when insurance cannot be used -------------------------------------------
def _find_open_slot_for(tools, ref, specialty="ENT"):
    return tools.find_appointment_slots(specialty, patient_ref=ref)["slots"][0]["slot_id"]


def test_lapsed_patient_is_offered_self_pay_not_refused_care(tools):
    ref = _find_patient(tools, eligibility_status="lapsed")["patient_ref"]
    slot = _find_open_slot_for(tools, ref)
    result = tools.book_appointment(ref, slot)
    assert result["ok"] is False and slot not in tools.booked  # nothing booked without the patient's agreement
    assert result["self_pay_available"] is True and result["self_pay_usd"] == tools._get_visit_price("ENT")
    assert "self_pay=true" in result["error"] and "check-in" in result["payment_terms"]
    assert tools.flags == []


def test_lapsed_patient_who_agrees_is_booked_as_self_pay_and_staff_are_told(tools):
    ref = _find_patient(tools, eligibility_status="lapsed")["patient_ref"]
    slot = _find_open_slot_for(tools, ref)
    result = tools.book_appointment(ref, slot, self_pay=True)
    assert result["ok"] and result["payment"] == "self_pay" and result["self_pay_usd"] == tools._get_visit_price("ENT")
    assert tools.booked[slot] == ref
    assert [flag["flag_id"] for flag in tools.flags] == [result["flag_id"]] and "Self-pay" in tools.flags[0]["reason"]


def test_pending_verification_is_booked_and_flagged_for_verification(tools):
    ref = _find_patient(tools, eligibility_status="pending_verification")["patient_ref"]
    slot = _find_open_slot_for(tools, ref)
    result = tools.book_appointment(ref, slot)
    assert result["ok"] and result["payment"] == "insurance_pending_verification" and "verif" in result["note"]
    assert "Verify coverage" in tools.flags[0]["reason"] and tools.flags[0]["flag_id"] == result["flag_id"]
    assert tools.calculate_copay(ref, "CARD-CONS")["patient_owes_usd"] is None  # still no insurance quote


def test_active_patient_books_on_insurance_without_a_flag(tools):
    ref = _find_patient(tools, eligibility_status="active")["patient_ref"]
    result = tools.book_appointment(ref, _find_open_slot_for(tools, ref))
    assert result["ok"] and result["payment"] == "insurance" and "flag_id" not in result and tools.flags == []


def test_every_specialty_has_a_self_pay_visit_price(tools):
    assert all(isinstance(tools._get_visit_price(specialty), int) for specialty in tools.specialties)


def test_state_is_per_instance():
    first, second = CareFlowTools(), CareFlowTools()
    patient = next(candidate for candidate in first.patients.values() if candidate["eligibility_status"] == "active")
    slot = first.find_appointment_slots("ENT", patient_ref=patient["patient_ref"])["slots"][0]["slot_id"]
    assert first.book_appointment(patient["patient_ref"], slot)["ok"]
    assert first.slots[slot]["status"] == "booked"
    assert slot in [offered["slot_id"] for offered in second.find_appointment_slots("ENT", patient_ref=patient["patient_ref"], limit=20)["slots"]]


def test_injected_clock_moves_today(tools):
    from datetime import datetime

    from src.domain.clock import FrozenClock

    # Slots in the data run 3-15 August 2026.
    later = CareFlowTools(store=tools.store, clock=FrozenClock(datetime(2026, 8, 12, 9, 0)))
    slots = later.find_appointment_slots("ENT", limit=20)["slots"]
    assert slots and all(slot["starts_at"] >= "2026-08-12T09:00" for slot in slots)
    assert tools.find_appointment_slots("ENT")["slots"][0]["starts_at"] < "2026-08-12"
    assert later.flag_for_human("x")["created_at"] == "2026-08-12T09:00"


def test_timezone_aware_earliest_date_is_accepted_not_a_crash(tools):
    # A "...Z" or "+05:30" date is compared with the naive slot times without raising TypeError.
    for value in ("2026-08-05T09:00:00Z", "2026-08-05T09:00:00+05:30"):
        result = tools.find_appointment_slots("ENT", earliest_date=value, limit=20)
        assert result["ok"] and all(slot["starts_at"] >= "2026-08-05" for slot in result["slots"])
    assert tools.call("find_appointment_slots", {"specialty": "ENT", "earliest_date": "2026-08-05T09:00:00Z"})["ok"]


def test_ids_and_codes_are_case_insensitive(tools):
    # "gast-colo" and a lower-case patient ref are accepted like their upper-case forms.
    patient = _find_patient(tools, plan_code="MERIDIAN-SILVER", deductible_met_usd=500, eligibility_status="active")
    assert tools.calculate_copay(patient["patient_ref"].lower(), " gast-colo ")["patient_owes_usd"] == 1480
    assert tools.find_appointment_slots("ent")["specialty"] == "ENT"
    slot = tools.find_appointment_slots("ENT", patient_ref=patient["patient_ref"])["slots"][0]["slot_id"]
    assert tools.book_appointment(patient["patient_ref"].lower(), slot.lower())["ok"]
    assert tools.flag_for_human("x", "Routine", patient["patient_ref"].lower())["patient_ref"] == patient["patient_ref"]


# ---- new patients ---------------------------------------------------------------------------
def _pick_closed_slot_and_patients(tools):
    """An open slot whose provider is not accepting new patients, plus a new and an established patient."""
    slot = next(
        candidate["slot_id"]
        for candidate in sorted(tools.slots.values(), key=lambda row: row["starts_at"])
        if candidate["status"] == "open" and candidate["starts_at"] >= "2026-08-01T09:00"
        and not tools.providers[candidate["provider_id"]]["accepts_new_patients"]
    )
    active = [patient["patient_ref"] for patient in tools.patients.values() if patient["eligibility_status"] == "active"]
    new = next(ref for ref in active if ref not in tools.established)
    old = next(ref for ref in active if ref in tools.established)
    return slot, new, old


def test_new_patient_cannot_book_a_provider_not_accepting_new_patients(tools):
    slot, new, old = _pick_closed_slot_and_patients(tools)
    result = tools.book_appointment(new, slot)
    assert result["ok"] is False and "not accepting new patients" in result["error"] and slot not in tools.booked
    assert tools.book_appointment(old, slot)["ok"]  # an established patient can


def test_explicit_new_patient_override_wins_over_the_inferred_value(tools):
    slot, new, _ = _pick_closed_slot_and_patients(tools)
    assert tools.book_appointment(new, slot, new_patient=False)["ok"]


def test_slot_search_hides_closed_providers_for_a_new_patient_only(tools):
    slot, new, old = _pick_closed_slot_and_patients(tools)
    spec = tools.slots[slot]["specialty"]
    for_new = tools.find_appointment_slots(spec, patient_ref=new, limit=20)["slots"]
    assert for_new and all(offered["accepts_new_patients"] for offered in for_new)
    for_old = tools.find_appointment_slots(spec, patient_ref=old, limit=20)["slots"]
    assert any(not offered["accepts_new_patients"] for offered in for_old)
    assert tools.find_appointment_slots(spec, new_patient=True, limit=20)["slots"] == for_new


def test_only_completed_past_visits_make_a_patient_established(tools):
    # Cancelled, no-show and future rows must not count.
    rows = tools.store.list_rows("appointments")
    by_patient: dict[str, set] = {}
    for row in rows:
        past = row["scheduled_for"] < "2026-08-01T09:00"
        by_patient.setdefault(row["patient_ref"], set()).add((row["status"], past))
    not_completed = next(ref for ref, visits in by_patient.items() if not any(status == "completed" and past for status, past in visits))
    assert tools.is_new_patient(not_completed)
    completed = next(ref for ref, visits in by_patient.items() if ("completed", True) in visits)
    assert not tools.is_new_patient(completed)
    assert tools.check_eligibility(patient_ref=completed)["new_patient"] is False


# ---- flag for human -------------------------------------------------------------------------
def test_flag_records_validates_and_is_idempotent(tools):
    patient = _find_patient(tools, eligibility_status="active")
    first = tools.flag_for_human("asks about a dosing change", "Urgent", patient["patient_ref"])
    assert first["ok"] and first["duplicate"] is False and first["status"] == "awaiting_staff"
    second = tools.flag_for_human("asks about a dosing change", "Urgent", patient["patient_ref"])
    assert second["duplicate"] is True and second["flag_id"] == first["flag_id"] and len(tools.flags) == 1
    assert tools.flag_for_human("", "Urgent")["ok"] is False
    assert tools.flag_for_human("x", "Critical")["ok"] is False
    assert tools.flag_for_human("x", "Routine", "NOPE")["ok"] is False


# ---- dispatch: what the model sends -----------------------------------------------------------
def test_call_accepts_json_string_and_survives_bad_arguments(tools):
    patient = _find_patient(tools, eligibility_status="active")
    assert tools.call("check_eligibility", f'{{"patient_ref": "{patient["patient_ref"]}"}}')["ok"]
    assert tools.call("check_eligibility", "{not json")["ok"] is False
    assert "bad arguments" in tools.call("calculate_copay", {"patient": "x"})["error"]
    assert "unknown tool" in tools.call("drop_tables", {})["error"]
    assert tools.call("flag_for_human", None)["ok"] is False  # missing required reason
