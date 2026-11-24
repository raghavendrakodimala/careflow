"""Offline tests for the domain rules: business days, referral SLA, PCP sign-off and the triage route."""

from datetime import date

import pytest

from src.domain import policies
from src.store import RuntimeStore


def test_compute_referral_due_date_reproduces_every_seeded_sla_date():
    store = RuntimeStore(":memory:")
    details = {detail["referral_id"]: detail for detail in store.list_rows("referral_details")}
    referrals = store.list_rows("referrals")
    assert len(referrals) == 300
    for referral in referrals:
        due = policies.compute_referral_due_date(date.fromisoformat(referral["raised_on"]), referral["urgency"])
        assert due.isoformat() == details[referral["referral_id"]]["sla_due_on"], referral["referral_id"]


def test_urgent_referral_raised_on_a_friday_is_due_on_monday():
    # Toolkit README and referral policy 3.9.
    friday = date(2026, 7, 31)
    assert friday.weekday() == 4
    assert policies.compute_referral_due_date(friday, "Urgent") == date(2026, 8, 3)
    assert policies.compute_referral_due_date(friday, "Priority") == date(2026, 8, 4)
    assert policies.compute_referral_due_date(friday, "Routine") == date(2026, 8, 7)


def test_emergent_has_no_referral_sla():
    assert policies.compute_referral_due_date(date(2026, 7, 31), "Emergent") is None


def test_holidays_are_skipped():
    assert policies.add_business_days(date(2026, 7, 31), 1, holidays=[date(2026, 8, 3)]) == date(2026, 8, 4)


def test_sign_off_lists():
    for specialty in ("Cardiology", "Gastroenterology", "Neurology", "Endocrinology", "Orthopaedics"):
        assert policies.is_sign_off_required(specialty) is True
    assert policies.is_sign_off_required("Dermatology") is False
    # The policy lists these two in neither 3.4 nor 3.5: staff decide.
    assert policies.is_sign_off_required("ENT") is None
    assert policies.is_sign_off_required("Pulmonology") is None


def test_find_highest_band_wins_and_never_lowers():
    assert policies.find_highest_band("Routine", "Urgent", "Priority") == "Urgent"
    assert policies.find_highest_band("Priority", None) == "Priority"
    assert policies.is_lower_band("Urgent", "Routine")
    assert not policies.is_lower_band("Routine", "Urgent")


OK = dict(seeks_clinical_advice=False, urgency="Routine", patient_known=True, specialty="Dermatology")


@pytest.mark.parametrize(
    "change, route, fragment",
    [
        ({}, "auto", None),
        ({"urgency": "Priority"}, "auto", None),
        ({"specialty": "Cardiology", "has_referral": True}, "auto", None),
        ({"seeks_clinical_advice": True}, "refuse", "OP-CS-014 5.1"),
        ({"urgency": "Emergent"}, "refuse", "referral policy 3.3"),
        ({"urgency": "Urgent"}, "human_review", "SOP 4.8"),
        ({"patient_known": False}, "human_review", "not identified"),
        ({"specialty": None}, "human_review", "No specialty"),
        ({"specialty": "Cardiology"}, "human_review", "sign-off"),
        ({"specialty": "ENT"}, "human_review", "neither"),
        ({"preauth_status": "pending"}, "human_review", "pending"),
        ({"preauth_status": "denied"}, "human_review", "denied"),
        ({"referral_overdue": True}, "human_review", "SLA"),
        ({"repeat_priority": True}, "human_review", "8.2"),
    ],
)
def test_decide_triage_route(change, route, fragment):
    got, reasons = policies.decide_triage_route(**{**OK, **change})
    assert got == route
    if fragment:
        assert any(fragment in reason for reason in reasons), reasons
    else:
        assert reasons == []


def test_refusal_is_decided_before_anything_else():
    # A clinical question from an unknown caller with no specialty is still only a refusal.
    route, reasons = policies.decide_triage_route(seeks_clinical_advice=True, urgency="Emergent", patient_known=False)
    assert route == "refuse" and len(reasons) == 2
