"""Clinic business rules as pure functions over plain table rows (dicts). No I/O, no LLM.

Each rule is used by the M2 tools today and will be called unchanged by the M5 workflow nodes, the M6 MCP
server and the M8 API, so every path gives the same numbers and decisions.

Sources: toolkit README worked example and copay-schedule (co-pay), preauth-matrix 2.5 (self-pay rate =
list price), intake-triage-sop 5.3-5.4 (self-pay, never refuse care), owner decision 2026-11-18 (payment rule).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

SELF_PAY_TERMS = "payment due at check-in; no deposit at booking"


# ---- co-pay ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CopayQuote:
    basis: str  # "copay" or "deductible_then_coinsurance"
    patient_owes_usd: int
    working: str


def copay_quote(
    plan: dict, procedure: dict, deductible_met_usd: int, flat_copays: dict[str, int], telehealth: bool = False
) -> CopayQuote:
    """What an insured patient with active cover owes.

    copay procedures: the plan's flat co-pay for the service category (a telehealth consultation uses the
    telehealth co-pay). Other procedures: unmet deductible, plus coinsurance on the rest of the list price,
    never more than the out-of-pocket maximum or the list price. A deductible met above the plan deductible
    counts as fully met.
    """
    if procedure["cost_share_basis"] == "copay":
        category = "telehealth" if telehealth and procedure["category"] == "consultation" else procedure["category"]
        return CopayQuote("copay", flat_copays[category], f"flat {category} co-pay for {plan['plan_code']}")
    deductible_left = max(0, plan["annual_deductible_usd"] - deductible_met_usd)
    price = procedure["list_price_usd"]
    from_deductible = min(price, deductible_left)
    coinsurance = round((price - from_deductible) * plan["coinsurance_rate"])
    owes = min(from_deductible + coinsurance, plan["out_of_pocket_max_usd"], price)
    working = (
        f"unmet deductible {from_deductible} + {plan['coinsurance_rate']:.0%} of "
        f"{price - from_deductible} = {coinsurance}"
    )
    return CopayQuote("deductible_then_coinsurance", owes, working)


def self_pay_price(procedure: dict) -> int:
    """The self-pay rate is the list price (preauth matrix 2.5)."""
    return procedure["list_price_usd"]


def visit_procedure(procedures: Iterable[dict], specialty: str) -> dict | None:
    """The consultation procedure for a specialty; its list price is the self-pay rate for a visit."""
    return next((p for p in procedures if p["specialty"] == specialty and p["category"] == "consultation"), None)


def insurance_quote_allowed(eligibility_status: str) -> bool:
    """Insurance estimates only on confirmed, active cover."""
    return eligibility_status == "active"


# ---- booking payment (owner rule) -------------------------------------------------------------
@dataclass(frozen=True)
class PaymentDecision:
    payment: str | None  # insurance | insurance_pending_verification | self_pay | None = not bookable yet
    staff_reason: str | None = None  # a staff flag to raise with the booking, if any

    @property
    def bookable(self) -> bool:
        return self.payment is not None


def booking_payment(eligibility_status: str, self_pay_agreed: bool, slot_id: str) -> PaymentDecision:
    """How a booking is paid.

    active -> insurance. pending_verification -> insurance, and staff verify cover before the visit.
    Anything else (lapsed, invalid, unknown) -> only as self-pay, and only once the patient has agreed.
    A self-pay booking always notifies staff (payment due at check-in).
    """
    if self_pay_agreed:
        return PaymentDecision(
            "self_pay", f"Self-pay booking {slot_id} (coverage {eligibility_status}); payment due at check-in"
        )
    if eligibility_status == "active":
        return PaymentDecision("insurance")
    if eligibility_status == "pending_verification":
        return PaymentDecision("insurance_pending_verification", f"Verify coverage before visit {slot_id}")
    return PaymentDecision(None)


# ---- new vs established patient ----------------------------------------------------------------
def established_patients(appointments: Iterable[dict], now: datetime) -> set[str]:
    """Established = at least one completed visit before now. Cancelled, no-show and future rows do not count."""
    today = now.isoformat(timespec="minutes")
    return {a["patient_ref"] for a in appointments if a["status"] == "completed" and a["scheduled_for"] < today}


def may_book_with(provider: dict, is_new_patient: bool) -> bool:
    return provider["accepts_new_patients"] or not is_new_patient


# ---- pre-authorisation -------------------------------------------------------------------------
@dataclass(frozen=True)
class PreauthVerdict:
    covered: bool
    preauth_required: bool
    approval_window_business_days: int | None
    note: str


def preauth_verdict(rule: dict, procedure: dict, plan_code: str) -> PreauthVerdict:
    code = procedure["procedure_code"]
    if not rule["covered"]:
        note = (
            f"{code} is not covered under {plan_code}. Self-pay is available at the list price "
            f"${self_pay_price(procedure):,}; the patient must agree before a date is offered."
        )
    elif rule["preauth_required"]:
        note = f"Pre-authorisation required; approval takes up to {rule['approval_window_business_days']} business days."
    else:
        note = "No pre-authorisation needed."
    return PreauthVerdict(rule["covered"], rule["preauth_required"], rule["approval_window_business_days"], note)
