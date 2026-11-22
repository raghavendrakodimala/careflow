"""Clinic business rules as pure functions over plain table rows (dicts). No I/O, no LLM.

Each rule is used by the M2 tools today and will be called unchanged by the M5 workflow nodes, the M6 MCP
server and the M8 API, so every path gives the same numbers and decisions.

Sources: toolkit README worked example and copay-schedule (co-pay), preauth-matrix 2.5 (self-pay rate =
list price), intake-triage-sop 5.3-5.4 (self-pay, never refuse care), owner decision 2026-11-18 (payment rule).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

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


# ---- referrals (M5): business days, SLA, PCP sign-off -------------------------------------------
# Referral Management Policy section 3. A business day is Monday to Friday (3.2); the clinic's holiday
# list is not in the toolkit data, so `holidays` is empty unless a caller supplies one.
URGENCY_ORDER = ("Routine", "Priority", "Urgent", "Emergent")
REFERRAL_SLA_BUSINESS_DAYS = {"Routine": 5, "Priority": 2, "Urgent": 1}  # 3.7-3.9; Emergent is never referred (3.3)
DIRECT_SPECIALTIES = frozenset({  # 3.4: no primary-care sign-off needed
    "Physiotherapy", "Podiatry", "Audiology", "Optometry", "Dietetics", "Occupational Therapy", "Dermatology",
})
SIGN_OFF_SPECIALTIES = frozenset({  # 3.5: written sign-off by the patient's primary-care physician
    "Cardiology", "Gastroenterology", "Neurology", "Endocrinology", "Rheumatology", "Orthopaedics",
    "Pain Management", "Psychiatry",
})


def add_business_days(day: date, n: int, holidays: Iterable[date] = ()) -> date:
    closed = set(holidays)
    while n > 0:
        day += timedelta(days=1)
        if day.weekday() < 5 and day not in closed:
            n -= 1
    return day


def referral_due(raised_on: date, urgency: str, holidays: Iterable[date] = ()) -> date | None:
    """The date a referral must be actioned by, or None for Emergent (no referral, no SLA: 3.3)."""
    days = REFERRAL_SLA_BUSINESS_DAYS.get(urgency)
    return add_business_days(raised_on, days, holidays) if days else None


def sign_off_required(specialty: str | None) -> bool | None:
    """True / False per 3.5 / 3.4; None when the policy lists the specialty in neither (staff decide)."""
    if specialty in SIGN_OFF_SPECIALTIES:
        return True
    if specialty in DIRECT_SPECIALTIES:
        return False
    return None


def highest_band(*bands: str | None) -> str:
    """Highest band wins (triage SOP 4.3); intake never lowers a band (4.5)."""
    known = [b for b in bands if b in URGENCY_ORDER]
    return max(known, key=URGENCY_ORDER.index) if known else "Routine"


def lowers_band(current: str, proposed: str) -> bool:
    return URGENCY_ORDER.index(proposed) < URGENCY_ORDER.index(current)


def validate_reband(current: str, proposed: str | None, note: str | None) -> str | None:
    """Why staff may not move a case from `current` to `proposed`, or None if they may.
    Emergent is never lowered into a referral band (referral policy 3.3); any other lowering needs a
    recorded reason (triage SOP 4.5)."""
    if not proposed or not lowers_band(current, proposed):
        return None
    if current == "Emergent":
        return "an Emergent case cannot be lowered into a referral band (referral policy 3.3)"
    if not (note or "").strip():
        return "lowering a band needs a recorded reason (triage SOP 4.5)"
    return None


# Explicit trigger phrases from the triage SOP table 4.6 (Emergent, Urgent). Two near-synonyms of the SOP's chest
# phrases ("chest tightness", "heart attack") are included on 4.4, upgrade on doubt: the M8 live run showed a
# message with them was deflected without the emergency statement. A deterministic floor under
# the model's band: highest band wins (4.3), upgrade on doubt (4.4). Also the only band available when the
# message could not be parsed at all.
_EMERGENT = re.compile(
    r"chest pain|pressure in my chest|chest (?:tightness|pressure)|tight(?:ness)? in (?:my|the) chest|heart attack|"
    r"can'?t breathe|cannot breathe|struggling to breathe|face (?:is )?drooping|"
    r"slurred speech|sudden weakness on one side|unresponsive|won'?t wake up|seizure|throat (?:is )?closing|"
    r"lips (?:are )?swelling|bleeding that won'?t stop|overdose|took too many|want to end my life|"
    r"kill myself|sudden loss of vision", re.I)
_URGENT = re.compile(
    r"fever (?:of |is )?over 103|can'?t keep fluids down|pain (?:is )?(?:an? )?(?:8|9|10) out of 10|"
    r"blood in my (?:urine|stool)|wound (?:has )?opened|stitches (?:have )?c[ao]me out|ran out of (?:my )?insulin|"
    r"hit my head", re.I)


# A phrase the person rules out in the same breath ("no chest pain", "I do not have chest pain") is not
# a report of it. Only a negation directly before the phrase counts; anything else still triggers
# (SOP 4.4: upgrade on doubt).
_NEGATED = re.compile(
    r"\b(?:no|not|never|without|denies|deny|denying|(?:do|does|did)(?:n'?t| not) have|(?:don|doesn|didn)'?t have)"
    r"\s+(?:any\s+|having\s+|a\s+|an\s+)?$", re.I)


def _stated(pattern: re.Pattern, text: str) -> bool:
    return any(not _NEGATED.search(text[:m.start()]) for m in pattern.finditer(text))


def trigger_band(text: str | None) -> str | None:
    """The band the SOP's own trigger phrases put a message in, or None if it contains none of them."""
    text = (text or "").replace("\u2019", "'")
    if _stated(_EMERGENT, text):
        return "Emergent"
    if _stated(_URGENT, text):
        return "Urgent"
    return None


# OP-CS-014 8.3: a caller who says they are a healthcare professional gets the same script, and the
# Clinical Routing Ticket is marked so the clinician who picks it up knows. The claim is never verified.
HCP_MARK = "Caller states HCP"
_HCP = re.compile(
    r"\b(?:i'?m|i am|as) an? (?:\w+ ){0,2}?(?:doctor|physician|cardiologist|surgeon|nurse|pharmacist|paramedic|"
    r"medical student|clinician|gp|consultant|registrar|healthcare professional|health professional)\b|"
    r"\b(?:clinician to clinician|doctor to doctor|colleague to colleague)\b|"
    r"\bi work (?:for|at) (?:meridian|another (?:provider|clinic|hospital|practice))", re.I)


def states_hcp(text: str | None) -> bool:
    return bool(_HCP.search((text or "").replace("\u2019", "'")))


def mark_hcp(reason: str, text: str | None) -> str:
    """`reason` for a clinical ticket, marked when the caller's message states they are a professional."""
    return f"{HCP_MARK}: {reason}" if states_hcp(text) and HCP_MARK not in reason else reason


_TOPICS = (  # first match wins; the topics are the seed table's (memory.TOPIC_SUMMARY)
    ("referral_status", re.compile(r"referr|MHP-R-\d", re.I)),
    ("billing_question", re.compile(r"co-?pay|\bcost|\bbill|statement|\bowe\b|how much", re.I)),
    ("eligibility_check", re.compile(r"eligib|plan (?:is )?active|\bcovered\b|coverage", re.I)),
)


def intake_topic(text: str | None) -> str:
    """What a non-clinical message is about, for the repeat-contact rules (same issue = same topic)."""
    return next((topic for topic, pattern in _TOPICS if pattern.search(text or "")), "appointment_booking")


# ---- triage route (M5) --------------------------------------------------------------------------
ROUTE_AUTO, ROUTE_HUMAN, ROUTE_REFUSE = "auto", "human_review", "refuse"  # the golden set's expected_route values


def triage_route(
    *,
    seeks_clinical_advice: bool,
    urgency: str,
    patient_known: bool = True,
    specialty: str | None = None,
    has_referral: bool = False,
    referral_status: str | None = None,
    referral_rejected: bool = False,
    preauth_status: str | None = None,
    referral_overdue: bool = False,
    repeat_priority: bool = False,
) -> tuple[str, list[str]]:
    """Where a triaged request goes, and why.

    refuse        the assistant does not act on the request itself: a clinical question is deflected to a
                  licensed clinician (OP-CS-014 5.1, 8.1) and an Emergent presentation is sent to emergency
                  care and never referred or booked (referral policy 3.3). Checked first, on nothing else.
    human_review  staff must decide before anything is booked.
    auto          the assistant may handle it.
    """
    refuse = []
    if urgency == "Emergent":
        refuse.append("Emergent presentation: emergency care, no referral or booking (referral policy 3.3)")
    if seeks_clinical_advice:
        refuse.append("Clinical question: deflected to a licensed clinician (OP-CS-014 5.1)")
    if refuse:
        return ROUTE_REFUSE, refuse

    reasons = []
    if urgency == "Urgent":
        reasons.append("Urgent band: live clinician contact within 1 hour (triage SOP 4.8)")
    if not patient_known:
        reasons.append("Patient not identified on record")
    if not specialty:
        reasons.append("No specialty stated or implied")
    else:
        sign_off = sign_off_required(specialty)
        if referral_rejected:
            reasons.append(f"The latest {specialty} referral was rejected")
        elif sign_off is None and not has_referral:
            reasons.append(f"{specialty} is in neither the direct nor the sign-off list (referral policy 3.4, 3.5)")
        elif sign_off and not has_referral:
            reasons.append(f"{specialty} needs primary-care sign-off and no referral is on file (referral policy 3.5)")
    if has_referral and referral_status == "open":
        reasons.append("Referral is still awaiting clinical triage acceptance (referral policy 3.14)")
    if preauth_status in ("pending", "denied"):
        reasons.append(f"Referral pre-authorisation is {preauth_status}; no deemed approval (pre-auth matrix 3.7)")
    if referral_overdue:
        reasons.append("Open referral is past its SLA date (referral policy 3.12)")
    if repeat_priority:
        reasons.append("Third contact on the same issue within 72 hours: Repeat - Priority (OP-CS-014 8.2)")
    return (ROUTE_HUMAN, reasons) if reasons else (ROUTE_AUTO, [])
