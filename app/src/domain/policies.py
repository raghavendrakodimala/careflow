"""Clinic business rules as pure functions over plain table rows (dicts). No I/O, no LLM.

Every caller (the tools, the triage workflow, the MCP server, the API) goes through these functions, so
every path gives the same numbers and decisions.

Sources: toolkit README worked example and copay-schedule (co-pay), preauth-matrix 2.5 (self-pay rate =
list price), intake-triage-sop 5.3-5.4 (self-pay, never refuse care), the clinic owner's booking-payment rule.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum


# ---- states and categories -----------------------------------------------------------------------
# Every member equals its string value, so comparisons and dict lookups against table rows, parsed JSON and
# saved state work unchanged. A member names or tests a value; whatever is written to a row, a saved state,
# a trace or a reply is the plain string (`.value`), never the member.
class Route(StrEnum):
    """Where a triaged request goes."""

    AUTO = "auto"
    HUMAN_REVIEW = "human_review"
    REFUSE = "refuse"


class UrgencyBand(StrEnum):
    """The triage bands, lowest to highest (triage SOP 4.6)."""

    ROUTINE = "Routine"
    PRIORITY = "Priority"
    URGENT = "Urgent"
    EMERGENT = "Emergent"


class EligibilityStatus(StrEnum):
    """patients.eligibility_status. UNKNOWN is not in the table: it stands for cover that could not be read."""

    ACTIVE = "active"
    PENDING_VERIFICATION = "pending_verification"
    LAPSED = "lapsed"
    UNKNOWN = "unknown"


class PaymentKind(StrEnum):
    """How a booking is paid (appointments.payment). SELF_PAY_OFFER is never stored on an appointment: it is
    the payment path of a case whose cover is not usable and who has not yet agreed to self-pay."""

    INSURANCE = "insurance"
    INSURANCE_PENDING_VERIFICATION = "insurance_pending_verification"
    SELF_PAY = "self_pay"
    SELF_PAY_OFFER = "self_pay_offer"


class CostShareBasis(StrEnum):
    """procedures.cost_share_basis: a flat co-pay, or the deductible and then coinsurance."""

    COPAY = "copay"
    DEDUCTIBLE_THEN_COINSURANCE = "deductible_then_coinsurance"


class ServiceCategory(StrEnum):
    """procedures.category and plan_copays.service_category (telehealth is a co-pay category only)."""

    CONSULTATION = "consultation"
    TELEHEALTH = "telehealth"
    IMAGING = "imaging"
    LAB = "lab"
    PROCEDURE = "procedure"


class AppointmentMode(StrEnum):
    """appointment_slots.mode."""

    IN_PERSON = "in_person"
    TELEHEALTH = "telehealth"


class AppointmentStatus(StrEnum):
    """appointments.status."""

    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class ReferralStatus(StrEnum):
    """referrals.status."""

    OPEN = "open"
    ACCEPTED = "accepted"
    SCHEDULED = "scheduled"
    CLOSED = "closed"
    REJECTED = "rejected"


class PreauthStatus(StrEnum):
    """referral_details.preauth_status."""

    APPROVED = "approved"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    DENIED = "denied"


class ContactTopic(StrEnum):
    """What a contact is about (patient_interactions.topic). The first five are the seed table's; the last
    two are the assistant's own: a policy question answered from the documents, and a contact with no
    specific issue."""

    CLINICAL_QUESTION = "clinical_question"
    APPOINTMENT_BOOKING = "appointment_booking"
    REFERRAL_STATUS = "referral_status"
    BILLING_QUESTION = "billing_question"
    ELIGIBILITY_CHECK = "eligibility_check"
    POLICY_QUESTION = "policy_question"
    GENERAL_ENQUIRY = "general_enquiry"


# The payment terms returned with every self-pay price.
SELF_PAY_TERMS = "payment due at check-in; no deposit at booking"


# ---- co-pay ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CopayQuote:
    basis: str  # a CostShareBasis value
    patient_owes_usd: int
    working: str  # how the amount was reached, in words the assistant can repeat to the patient


def compute_copay_quote(
    plan: dict, procedure: dict, deductible_met_usd: int, flat_copays: dict[str, int], telehealth: bool = False
) -> CopayQuote:
    """What an insured patient with active cover owes.

    copay procedures: the plan's flat co-pay for the service category (a telehealth consultation uses the
    telehealth co-pay). Other procedures: unmet deductible, plus coinsurance on the rest of the list price,
    never more than the out-of-pocket maximum or the list price. A deductible met above the plan deductible
    counts as fully met.

    `flat_copays` maps service category -> this plan's flat co-pay in USD; it must hold the category of a
    copay-basis procedure (KeyError otherwise).
    """
    if procedure["cost_share_basis"] == CostShareBasis.COPAY:
        category = procedure["category"]
        if telehealth and category == ServiceCategory.CONSULTATION:
            category = ServiceCategory.TELEHEALTH.value
        return CopayQuote(
            CostShareBasis.COPAY.value, flat_copays[category], f"flat {category} co-pay for {plan['plan_code']}")
    deductible_left = max(0, plan["annual_deductible_usd"] - deductible_met_usd)
    price = procedure["list_price_usd"]
    from_deductible = min(price, deductible_left)
    coinsurance = round((price - from_deductible) * plan["coinsurance_rate"])
    owes = min(from_deductible + coinsurance, plan["out_of_pocket_max_usd"], price)
    working = (
        f"unmet deductible {from_deductible} + {plan['coinsurance_rate']:.0%} of "
        f"{price - from_deductible} = {coinsurance}"
    )
    return CopayQuote(CostShareBasis.DEDUCTIBLE_THEN_COINSURANCE.value, owes, working)


def get_self_pay_price(procedure: dict) -> int:
    """The self-pay rate is the list price (preauth matrix 2.5)."""
    return procedure["list_price_usd"]


def find_visit_procedure(procedures: Iterable[dict], specialty: str) -> dict | None:
    """The consultation procedure for a specialty; its list price is the self-pay rate for a visit."""
    return next(
        (procedure for procedure in procedures
         if procedure["specialty"] == specialty and procedure["category"] == ServiceCategory.CONSULTATION),
        None)


def is_insurance_quote_allowed(eligibility_status: str) -> bool:
    """Insurance estimates only on confirmed, active cover."""
    return eligibility_status == EligibilityStatus.ACTIVE


# ---- booking payment (owner rule) -------------------------------------------------------------
@dataclass(frozen=True)
class PaymentDecision:
    payment: str | None  # insurance | insurance_pending_verification | self_pay | None = not bookable yet
    staff_reason: str | None = None  # a staff flag to raise with the booking, if any

    @property
    def bookable(self) -> bool:
        """False when cover is not usable and the patient has not yet agreed to self-pay."""
        return self.payment is not None


def decide_booking_payment(eligibility_status: str, self_pay_agreed: bool, slot_id: str) -> PaymentDecision:
    """How a booking is paid.

    active -> insurance. pending_verification -> insurance, and staff verify cover before the visit.
    Anything else (lapsed, invalid, unknown) -> only as self-pay, and only once the patient has agreed.
    A self-pay booking always notifies staff (payment due at check-in).
    """
    if self_pay_agreed:
        return PaymentDecision(
            PaymentKind.SELF_PAY.value,
            f"Self-pay booking {slot_id} (coverage {eligibility_status}); payment due at check-in",
        )
    if eligibility_status == EligibilityStatus.ACTIVE:
        return PaymentDecision(PaymentKind.INSURANCE.value)
    if eligibility_status == EligibilityStatus.PENDING_VERIFICATION:
        return PaymentDecision(
            PaymentKind.INSURANCE_PENDING_VERIFICATION.value, f"Verify coverage before visit {slot_id}")
    return PaymentDecision(None)


# ---- new vs established patient ----------------------------------------------------------------
def find_established_patients(appointments: Iterable[dict], now: datetime) -> set[str]:
    """Established = at least one completed visit before now. Cancelled, no-show and future rows do not count."""
    # scheduled_for is stored as ISO text to the minute, so comparing strings compares times.
    now_text = now.isoformat(timespec="minutes")
    return {appointment["patient_ref"] for appointment in appointments
            if appointment["status"] == AppointmentStatus.COMPLETED and appointment["scheduled_for"] < now_text}


def can_book_with(provider: dict, is_new_patient: bool) -> bool:
    """A provider who is not accepting new patients still sees established ones."""
    return provider["accepts_new_patients"] or not is_new_patient


# ---- pre-authorisation -------------------------------------------------------------------------
@dataclass(frozen=True)
class PreauthVerdict:
    covered: bool
    preauth_required: bool
    approval_window_business_days: int | None
    note: str


def build_preauth_verdict(rule: dict, procedure: dict, plan_code: str) -> PreauthVerdict:
    """A preauth_rules row (one procedure under one plan) as a verdict, with a note worded for the patient."""
    code = procedure["procedure_code"]
    if not rule["covered"]:
        note = (
            f"{code} is not covered under {plan_code}. Self-pay is available at the list price "
            f"${get_self_pay_price(procedure):,}; the patient must agree before a date is offered."
        )
    elif rule["preauth_required"]:
        note = f"Pre-authorisation required; approval takes up to {rule['approval_window_business_days']} business days."
    else:
        note = "No pre-authorisation needed."
    return PreauthVerdict(rule["covered"], rule["preauth_required"], rule["approval_window_business_days"], note)


# ---- referrals: business days, SLA, PCP sign-off ------------------------------------------------
# Referral Management Policy section 3. A business day is Monday to Friday (3.2); the clinic's holiday
# list is not in the toolkit data, so `holidays` is empty unless a caller supplies one.
URGENCY_ORDER = tuple(band.value for band in UrgencyBand)  # lowest to highest band, as plain strings
SAME_DAY_BANDS = (UrgencyBand.URGENT, UrgencyBand.EMERGENT)  # the "urgency flag" bands: same-day attention
DEFAULT_BAND = UrgencyBand.ROUTINE.value  # the band of a case or a staff flag until something raises it
# The contact topic of a clinical question. The repeat-contact rules (SOP 7.3, OP-CS-014 8.2) count by topic.
CLINICAL_ISSUE = ContactTopic.CLINICAL_QUESTION
# Referral statuses still in progress: worth recalling, and able to lead to a booking.
LIVE_REFERRAL_STATUSES = (ReferralStatus.OPEN, ReferralStatus.ACCEPTED, ReferralStatus.SCHEDULED)
_SATURDAY = 5  # date.weekday(): Monday is 0, so anything below this is a weekday
# 3.7-3.9; Emergent is never referred (3.3)
REFERRAL_SLA_BUSINESS_DAYS = {UrgencyBand.ROUTINE: 5, UrgencyBand.PRIORITY: 2, UrgencyBand.URGENT: 1}
DIRECT_SPECIALTIES = frozenset({  # 3.4: no primary-care sign-off needed
    "Physiotherapy", "Podiatry", "Audiology", "Optometry", "Dietetics", "Occupational Therapy", "Dermatology",
})
SIGN_OFF_SPECIALTIES = frozenset({  # 3.5: written sign-off by the patient's primary-care physician
    "Cardiology", "Gastroenterology", "Neurology", "Endocrinology", "Rheumatology", "Orthopaedics",
    "Pain Management", "Psychiatry",
})


def add_business_days(day: date, count: int, holidays: Iterable[date] = ()) -> date:
    """The date `count` business days after `day`. `day` itself is never counted; count <= 0 returns it unchanged."""
    closed = set(holidays)
    while count > 0:
        day += timedelta(days=1)
        if day.weekday() < _SATURDAY and day not in closed:
            count -= 1
    return day


def compute_referral_due_date(raised_on: date, urgency: str, holidays: Iterable[date] = ()) -> date | None:
    """The date a referral must be actioned by, or None for Emergent (no referral, no SLA: 3.3)."""
    days = REFERRAL_SLA_BUSINESS_DAYS.get(urgency)
    return add_business_days(raised_on, days, holidays) if days else None


def is_sign_off_required(specialty: str | None) -> bool | None:
    """True / False per 3.5 / 3.4; None when the policy lists the specialty in neither (staff decide)."""
    if specialty in SIGN_OFF_SPECIALTIES:
        return True
    if specialty in DIRECT_SPECIALTIES:
        return False
    return None


def find_highest_band(*bands: str | None) -> str:
    """Highest band wins (triage SOP 4.3); intake never lowers a band (4.5)."""
    known = [band for band in bands if band in URGENCY_ORDER]
    return max(known, key=URGENCY_ORDER.index) if known else DEFAULT_BAND


def is_lower_band(current: str, proposed: str) -> bool:
    """True when `proposed` is a lower band than `current`. Both must be in URGENCY_ORDER (ValueError if not)."""
    return URGENCY_ORDER.index(proposed) < URGENCY_ORDER.index(current)


def validate_reband(current: str, proposed: str | None, note: str | None) -> str | None:
    """Why staff may not move a case from `current` to `proposed`, or None if they may.
    Emergent is never lowered into a referral band (referral policy 3.3); any other lowering needs a
    recorded reason (triage SOP 4.5)."""
    if not proposed or not is_lower_band(current, proposed):
        return None
    if current == UrgencyBand.EMERGENT:
        return "an Emergent case cannot be lowered into a referral band (referral policy 3.3)"
    if not (note or "").strip():
        return "lowering a band needs a recorded reason (triage SOP 4.5)"
    return None


# Explicit trigger phrases from the triage SOP table 4.6 (Emergent, Urgent). Two near-synonyms of the SOP's chest
# phrases ("chest tightness", "heart attack") are included under 4.4, upgrade on doubt. Together they are a
# deterministic floor under the model's band: highest band wins (4.3), upgrade on doubt (4.4). They are also
# the only band available when the message could not be parsed at all.
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
# (SOP 4.4: upgrade on doubt). The pattern is matched against the text before a trigger phrase, hence the
# closing `$`.
_NEGATED = re.compile(
    r"\b(?:no|not|never|without|denies|deny|denying|(?:do|does|did)(?:n'?t| not) have|(?:don|doesn|didn)'?t have)"
    r"\s+(?:any\s+|having\s+|a\s+|an\s+)?$", re.I)


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))  # removed before matching
_APOSTROPHES = dict.fromkeys(map(ord, "\u2018\u2019\u02bc`\u00b4"), "'")     # every form a keyboard produces


def normalise_text(text: str | None) -> str:
    """The message as the rule patterns expect it: compatibility characters folded to their plain forms
    (NFKC: full-width letters, no-break spaces), zero-width characters removed, every apostrophe form made
    a plain one, and runs of whitespace collapsed to one space. A phrase must not escape a rule because of
    how it was typed."""
    folded = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH).translate(_APOSTROPHES)
    return " ".join(folded.split())


def _is_stated(pattern: re.Pattern, text: str) -> bool:
    """True if any match of `pattern` in `text` is not negated directly before it."""
    return any(not _NEGATED.search(text[:match.start()]) for match in pattern.finditer(text))


def find_trigger_band(text: str | None) -> str | None:
    """The band the SOP's own trigger phrases put a message in, or None if it contains none of them."""
    text = normalise_text(text)
    if _is_stated(_EMERGENT, text):
        return UrgencyBand.EMERGENT.value
    if _is_stated(_URGENT, text):
        return UrgencyBand.URGENT.value
    return None


# OP-CS-014 8.3: a caller who says they are a healthcare professional gets the same script, and the
# Clinical Routing Ticket is marked so the clinician who picks it up knows. The claim is never verified.
HCP_MARK = "Caller states HCP"
_HCP = re.compile(
    r"\b(?:i'?m|i am|as) an? (?:\w+ ){0,2}?(?:doctor|physician|cardiologist|surgeon|nurse|pharmacist|paramedic|"
    r"medical student|clinician|gp|consultant|registrar|healthcare professional|health professional)\b|"
    r"\b(?:clinician to clinician|doctor to doctor|colleague to colleague)\b|"
    r"\bi work (?:for|at) (?:meridian|another (?:provider|clinic|hospital|practice))", re.I)


def has_hcp_statement(text: str | None) -> bool:
    """True when the message claims a clinical profession or asks for a clinician-to-clinician answer."""
    return bool(_HCP.search(normalise_text(text)))


def mark_hcp(reason: str, text: str | None) -> str:
    """`reason` for a clinical ticket, marked when the caller's message states they are a professional."""
    return f"{HCP_MARK}: {reason}" if has_hcp_statement(text) and HCP_MARK not in reason else reason


_TOPICS = (  # first match wins; the topics are the seed table's (memory.TOPIC_SUMMARY)
    (ContactTopic.REFERRAL_STATUS, re.compile(r"referr|MHP-R-\d", re.I)),
    (ContactTopic.BILLING_QUESTION, re.compile(r"co-?pay|\bcost|\bbill|statement|\bowe\b|how much", re.I)),
    (ContactTopic.ELIGIBILITY_CHECK, re.compile(r"eligib|plan (?:is )?active|\bcovered\b|coverage", re.I)),
)


def classify_intake_topic(text: str | None) -> str:
    """What a non-clinical message is about, for the repeat-contact rules (same issue = same topic)."""
    return next(
        (topic for topic, pattern in _TOPICS if pattern.search(text or "")), ContactTopic.APPOINTMENT_BOOKING).value


# ---- triage route -------------------------------------------------------------------------------
ROUTE_AUTO, ROUTE_HUMAN, ROUTE_REFUSE = Route.AUTO, Route.HUMAN_REVIEW, Route.REFUSE  # aliases of the Route members


def decide_triage_route(
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

    Returns (route, reasons). Every reason that applies is listed, not just the first; auto has none.
    `has_referral`, `referral_status`, `referral_rejected`, `preauth_status` and `referral_overdue` describe
    the patient's latest referral to `specialty`; `repeat_priority` is the OP-CS-014 8.2 repeat-contact rule.
    """
    refuse = []
    if urgency == UrgencyBand.EMERGENT:
        refuse.append("Emergent presentation: emergency care, no referral or booking (referral policy 3.3)")
    if seeks_clinical_advice:
        refuse.append("Clinical question: deflected to a licensed clinician (OP-CS-014 5.1)")
    if refuse:
        return Route.REFUSE.value, refuse

    reasons = []
    if urgency == UrgencyBand.URGENT:
        reasons.append("Urgent band: live clinician contact within 1 hour (triage SOP 4.8)")
    if not patient_known:
        reasons.append("Patient not identified on record")
    if not specialty:
        reasons.append("No specialty stated or implied")
    else:
        sign_off = is_sign_off_required(specialty)
        if referral_rejected:
            reasons.append(f"The latest {specialty} referral was rejected")
        elif sign_off is None and not has_referral:
            reasons.append(f"{specialty} is in neither the direct nor the sign-off list (referral policy 3.4, 3.5)")
        elif sign_off and not has_referral:
            reasons.append(f"{specialty} needs primary-care sign-off and no referral is on file (referral policy 3.5)")
    if has_referral and referral_status == ReferralStatus.OPEN:
        reasons.append("Referral is still awaiting clinical triage acceptance (referral policy 3.14)")
    if preauth_status in (PreauthStatus.PENDING, PreauthStatus.DENIED):
        reasons.append(f"Referral pre-authorisation is {preauth_status}; no deemed approval (pre-auth matrix 3.7)")
    if referral_overdue:
        reasons.append("Open referral is past its SLA date (referral policy 3.12)")
    if repeat_priority:
        reasons.append("Third contact on the same issue within 72 hours: Repeat - Priority (OP-CS-014 8.2)")
    return (Route.HUMAN_REVIEW.value, reasons) if reasons else (Route.AUTO.value, [])
