"""Pure domain rules: no database, no LLM, no tool layer. Plain dicts in, plain values out."""

from datetime import datetime

import pytest

from src.domain import policies as policies
from src.domain.clock import FROZEN_NOW, FrozenClock

SILVER = {"plan_code": "MERIDIAN-SILVER", "annual_deductible_usd": 1500, "coinsurance_rate": 0.3,
          "out_of_pocket_max_usd": 6000}
COLONOSCOPY = {"procedure_code": "GAST-COLO", "cost_share_basis": "deductible_then_coinsurance", "category": "procedure",
        "list_price_usd": 2600, "specialty": "Gastroenterology"}
CONSULTATION = {"procedure_code": "CARD-CONS", "cost_share_basis": "copay", "category": "consultation",
        "list_price_usd": 220, "specialty": "Cardiology"}


def test_copay_readme_worked_example():
    quote = policies.compute_copay_quote(SILVER, COLONOSCOPY, deductible_met_usd=500, flat_copays={})
    assert quote.patient_owes_usd == 1480 and quote.basis == "deductible_then_coinsurance"  # 1000 + 30% of 1600


def test_copay_caps_and_over_met_deductible():
    assert policies.compute_copay_quote(SILVER, COLONOSCOPY, 99999, {}).patient_owes_usd == round(2600 * 0.3)
    assert policies.compute_copay_quote({**SILVER, "annual_deductible_usd": 3000}, COLONOSCOPY, 0, {}).patient_owes_usd == 2600
    assert policies.compute_copay_quote({**SILVER, "out_of_pocket_max_usd": 1000}, COLONOSCOPY, 0, {}).patient_owes_usd == 1000


def test_flat_copay_and_telehealth():
    flat = {"consultation": 45, "telehealth": 18}
    assert policies.compute_copay_quote(SILVER, CONSULTATION, 0, flat).patient_owes_usd == 45
    assert policies.compute_copay_quote(SILVER, CONSULTATION, 0, flat, telehealth=True).patient_owes_usd == 18


@pytest.mark.parametrize(
    "status, agreed, payment, flagged",
    [
        ("active", False, "insurance", False),
        ("pending_verification", False, "insurance_pending_verification", True),
        ("lapsed", False, None, False),
        ("unknown_status", False, None, False),
        ("lapsed", True, "self_pay", True),
        ("active", True, "self_pay", True),
    ],
)
def test_decide_booking_payment_rule(status, agreed, payment, flagged):
    decision = policies.decide_booking_payment(status, agreed, "MHP-SL-00001")
    assert decision.payment == payment and decision.bookable == (payment is not None) and bool(decision.staff_reason) == flagged


def test_get_self_pay_price_is_list_price_and_visit_uses_the_consultation():
    assert policies.get_self_pay_price(COLONOSCOPY) == 2600
    assert policies.find_visit_procedure([COLONOSCOPY, CONSULTATION], "Cardiology") is CONSULTATION
    assert policies.find_visit_procedure([COLONOSCOPY], "Cardiology") is None


def test_find_established_patients_counts_only_completed_past_visits():
    rows = [
        {"patient_ref": "A", "status": "completed", "scheduled_for": "2026-07-01T10:00"},
        {"patient_ref": "B", "status": "cancelled", "scheduled_for": "2026-07-01T10:00"},
        {"patient_ref": "C", "status": "no_show", "scheduled_for": "2026-07-01T10:00"},
        {"patient_ref": "D", "status": "completed", "scheduled_for": "2026-09-01T10:00"},  # future
    ]
    assert policies.find_established_patients(rows, FROZEN_NOW) == {"A"}
    assert policies.find_established_patients(rows, datetime(2026, 10, 1)) == {"A", "D"}


def test_new_patient_rule():
    assert policies.can_book_with({"accepts_new_patients": False}, is_new_patient=False)
    assert not policies.can_book_with({"accepts_new_patients": False}, is_new_patient=True)
    assert policies.can_book_with({"accepts_new_patients": True}, is_new_patient=True)


def test_build_preauth_verdict():
    required = policies.build_preauth_verdict({"covered": True, "preauth_required": True, "approval_window_business_days": 5},
                            {"procedure_code": "ORTH-MRI", "list_price_usd": 1400}, "MERIDIAN-BRONZE")
    assert required.preauth_required and "5 business days" in required.note
    excluded = policies.build_preauth_verdict({"covered": False, "preauth_required": False, "approval_window_business_days": None},
                             {"procedure_code": "CARD-CATH", "list_price_usd": 7800}, "CIVIC-BASE")
    assert not excluded.covered and "$7,800" in excluded.note and "Self-pay" in excluded.note


def test_frozen_clock():
    assert FrozenClock().now() == FROZEN_NOW


def test_domain_layer_imports_no_framework_or_outer_layer():
    # ADR-01: dependencies point inward. The domain may import only the standard library and itself.
    import ast
    from pathlib import Path

    import src.domain as domain

    allowed_top = {"__future__", "collections", "dataclasses", "datetime", "enum", "re", "typing", "unicodedata"}  # standard library only
    for path in Path(domain.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:  # relative: must stay inside domain/
                assert node.level == 1, f"{path.name} imports outside domain/"
            elif isinstance(node, ast.ImportFrom | ast.Import):
                names = [node.module] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
                for name in names:
                    assert name.split(".")[0] in allowed_top, f"{path.name} imports {name}"
