"""Guardrails: the clinic assistant never gives clinical advice (OP-CS-014). Three layers:

    1. input gate    check_input()   before any agent sees the message
    2. scoped prompt prompts.py      the clinical-advice rules every agent carries
    3. output check  check_output()  before a reply is sent; then the Clinical Safety Reviewer and the
                                     grounding check

Input gate, in order:
- rules, no model call: the SOP's Emergent trigger phrases, prompt-injection markers, and wording that
  asks for a diagnosis, a dose or what to do about a symptom;
- then, if a classifier is given, a model score. At or above THRESHOLD (0.30, OP-CS-014 9.2) the message
  is blocked. If the classifier does not answer within the time limit (2 seconds, 9.3) or fails, the
  message is blocked: fail closed.

A blocked message is treated exactly like a clinical question: the fixed deflection script (with the
emergency statement for an Emergent phrase) and a Clinical Routing Ticket. "If unsure whether a question
is clinical, it is clinical" (5.4).

The rules are deliberately one-sided: they may block a harmless message (a nurse then calls back), but
they are never the reason a clinical answer is sent. tests/test_guardrails.py holds the table that
measures both directions.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .config import GUARD_TIME_LIMIT_S, Settings
from .domain import policies
from .domain.policies import UrgencyBand
from .llm_client import complete_structured
from .prompts import CLASSIFIER_PROMPT, DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from .state import Role

THRESHOLD = 0.30   # OP-CS-014 9.2: a classifier score at or above this blocks; only the CMO may change it
TIME_LIMIT_S = GUARD_TIME_LIMIT_S  # OP-CS-014 9.3: seconds the classifier has to answer before the message is blocked

# The patterns below are tuned against the table in tests/test_guardrails.py; re-run it after any change.

# Prompt-injection markers: telling the assistant to ignore or disregard its instructions, giving it a new
# role or persona (doctor, "DAN", "MedGPT"), or asking for its system prompt or an unrestricted mode.
_INJECTION = re.compile(
    r"ignore (?:all |any |the |your |these |previous |prior |above )+(?:\w+ ){0,3}?(?:instructions?|rules?|polic\w+|prompts?|documents?)|"
    r"disregard (?:all |the |your |previous )+|you are now\b|from now on you|pretend (?:to be|you(?:'re| are))|"
    r"\bact as (?:a |an |my )?(?:doctor|physician|nurse|medical|unrestricted)|role-?play|unrestricted|jailbreak|"
    r"developer mode|system prompt|\bDAN\b|medgpt|no (?:rules|restrictions|filters)", re.I)
# Fragments shared by _CLINICAL and _LEAK (strings, not compiled on their own).
# A medicine or home remedy: the object that makes "use / apply / try ..." a treatment question.
_TREATMENT = (r"(?:cream|ointment|lotion|inhaler|drops|spray|medication|medicine|meds|tablets?|pills?|painkillers?|"
              r"ice|heat|compress\w*|bandage|leftover\w*|antibiotics?|ibuprofen|paracetamol|aspirin)\b")
# An administrative action: "is it okay to <verb>" / "is it okay if I <verb>" is clinic business, not clinical.
_ADMIN_VERB = (r"(?:pay|book|cancel|reschedule|arrive|bring|park|come|call|change|move|use my|email|send|upload|"
               r"switch (?:my )?(?:appointment|plan))\b")
# An administrative subject: a reply saying "looks like a referral / your plan / a $40 co-pay" is not a diagnosis.
_ADMIN_NOUN = (r"(?:referral|appointment|slot|plan|coverage|deductible|record|booking|pre-?authori[sz]ation|co-?pay\w*|"
               r"bill|statement|claim|payment|policy|member\w*|visit|balance|account)\b")
# A message that asks for clinical advice (OP-CS-014 5.1): a dose or an amount of medicine, whether to take,
# stop or change something, whether a symptom is safe, normal or serious, what it is or what causes it, what
# a result means, or what to do about it.
_CLINICAL = re.compile(
    r"\bdos(?:e|es|ing|age)\b|\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|milligrams?)\b|"
    r"\bshould i (?:take|stop|start|keep taking|increase|reduce|lower|cut|double|skip|switch|be worried|worry)\b|"
    # OP-CS-014 5.2: an administrative "should I" / "can I" / "is it okay" is answerable ("Should I bring my
    # insurance card?"), so use/apply/try need a treatment as their object, "go to" an emergency setting,
    # and "is it okay to <pay, book, ...>" is left alone.
    rf"\b(?:should|can|could|may) i (?:use|apply|try) (?:[\w'-]+ ){{0,3}}?{_TREATMENT}|"
    r"\bshould i go to (?:the |an? )?(?:er|ed|a&e|hospital|urgent care|emergency)\b|"
    r"\b(?:can|could|may) i (?:take|stop|mix|combine|drink)\b|\bis it (?:safe|normal|serious|dangerous)\b|"
    rf"\bis it (?:ok|okay)\b(?! (?:to|if i) {_ADMIN_VERB})|"
    r"\bis (?:this|that|it) something (?:serious|to worry)|\ba sign of\b|\bshould i (?:just )?(?:wait|rest|keep \w+ing)\b|"
    r"\bis this (?:normal|serious|dangerous|cancer|an? \w+)\b|\bdo i have\b|\bcould (?:it|this) be\b|"
    r"\bwhat (?:do you think|could|might|would) (?:it|this|that|be)\b|\bwhat(?:'s| is) (?:wrong|causing|the cause)\b|"
    r"\bwhat (?:does|do) (?:my|the|this|these) [\w\s-]{0,30}?(?:results?|scan|x-?ray|mri|ct|bloods?|test|labs?|rash|lump|symptoms?) mean\b|"
    r"\b(?:interpret|diagnos\w+|prognosis|side effects?|second opinion on)\b|\bhow (?:much|many|often) (?:should|can|do) i take\b|"
    r"\bwhat (?:should|can) i (?:do|take) (?:about|for)\b|\bis (?:it|this|that) (?:a |an )?(?:heart attack|stroke|infection|allergic|allergy)",
    re.I)
# Clinical content in a reply: a dose, an instruction to take, stop or change something, a likely cause or
# diagnosis, a recommendation, reassurance, or an adopted persona. The fixed scripts are removed before this
# is applied.
_LEAK = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|milligrams?)\b|"
    r"\byou (?:should|could|can|may want to|might want to) (?:take|stop taking|start|increase|reduce|lower|halve|double|skip|rest|ice)\b|"
    # "try" and "apply" need a treatment ("You can apply for financial assistance" is clinic business), and
    # "looks like" is not a diagnosis when it is about a referral, a plan or a bill.
    rf"\byou (?:should|could|can|may want to|might want to) (?:try|apply) (?:[\w'-]+ ){{0,3}}?{_TREATMENT}|"
    rf"\b(?:sounds|looks|seems) like (?:a|an|you|it)\b(?! (?:\w+ |\$[\d,.]+ ){{0,6}}?{_ADMIN_NOUN})|"
    r"\b(?:is|it's|its|are) (?:likely|probably|most likely|possibly) (?:a|an|just|due|caused)\b|"
    r"\bi(?: would|'d) (?:go with|recommend|suggest|advise)\b|\bmy (?:advice|recommendation|opinion) (?:is|would be)\b|"
    r"\b(?:take|taking) (?:an? |some |your )?(?:aspirin|ibuprofen|paracetamol|acetaminophen|antibiotics?|antihistamines?)\b|"
    r"\bas medgpt\b|\bi(?: am|'m) medgpt\b|\bsince you(?:'re| are) a clinician\b|\bnothing to worry about\b|"
    r"\b(?:this|that|it) (?:is|means|indicates|suggests) (?:you have|a sign of|an? (?:infection|allergy|fracture))", re.I)


class GuardCategory(StrEnum):
    """Why a guardrail check blocked (GuardResult.category). A result holds the plain string."""

    NONE = "none"
    EMERGENCY = "emergency"
    CLINICAL = "clinical"
    INJECTION = "injection"
    UNVERIFIED = "unverified"  # the classifier gave no answer: blocked unseen (fail closed)


class GuardLayer(StrEnum):
    """Which part of the input gate decided (GuardResult.layer). A result holds the plain string."""

    NONE = "none"
    RULES = "rules"
    CLASSIFIER = "classifier"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class GuardResult:
    """The decision of one guardrail check, as it is logged and traced."""

    blocked: bool
    category: str = GuardCategory.NONE.value  # a GuardCategory value
    layer: str = GuardLayer.NONE.value        # a GuardLayer value
    reason: str = ""
    band: str | None = None  # the SOP trigger-phrase band of a blocked message (Emergent or Urgent), if any
    score: float | None = None  # the classifier's score, when it answered
    latency_ms: float = 0.0  # time the whole input gate took (set by check_input only)

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "category": self.category, "layer": self.layer, "reason": self.reason,
                "band": self.band, "score": self.score, "latency_ms": round(self.latency_ms, 1)}


ALLOWED = GuardResult(False)  # the shared "nothing found" result of the rule checks


class ClinicalRisk(BaseModel):
    """What a classifier returns for one message (also the hosted classifier's structured-output schema)."""

    score: float = Field(ge=0.0, le=1.0)  # how likely the message asks for clinical advice
    category: Literal["none", "clinical", "injection"] = "none"


Classifier = Callable[[str], ClinicalRisk]  # message text -> risk; may raise or be slow (the gate fails closed)



def build_hosted_classifier(settings: Settings | None = None) -> Classifier:
    """The hosted classifier: the chat model with CLASSIFIER_PROMPT, limited to one short attempt."""
    # The gate gives up after the time limit, but it cannot stop a call already in flight. Without these
    # limits an abandoned call keeps retrying across the fallback models for minutes and uses up the
    # provider's rate limit for the messages that follow.
    from .config import get_settings

    settings = settings or get_settings()
    settings = settings.model_copy(
        update={"request_timeout_s": max(1, math.ceil(settings.careflow_guard_timeout_s)), "transport_retries": 0,
                "careflow_fallback_models": ""})

    def classify(text: str) -> ClinicalRisk:
        messages = [{"role": Role.SYSTEM.value, "content": CLASSIFIER_PROMPT},
                    {"role": Role.USER.value, "content": text}]
        return complete_structured(messages, ClinicalRisk, settings, validation_retries=0)

    return classify


def check_rules(text: str) -> GuardResult:
    """The deterministic part of the input gate. No model call."""
    text = policies.normalise_text(text)  # the rules must not depend on how a phrase was typed
    band = policies.find_trigger_band(text)  # an Urgent phrase does not block, but it sets the ticket's urgency
    rules = GuardLayer.RULES.value
    if band == UrgencyBand.EMERGENT:
        return GuardResult(True, GuardCategory.EMERGENCY.value, rules, "Emergent trigger phrase (triage SOP 4.6)",
                           band=UrgencyBand.EMERGENT.value)
    if _INJECTION.search(text):
        return GuardResult(True, GuardCategory.INJECTION.value, rules,
                           "Attempt to change the assistant's rules or role", band=band)
    if _CLINICAL.search(text):
        return GuardResult(True, GuardCategory.CLINICAL.value, rules, "Asks for clinical advice (OP-CS-014 5.1)",
                           band=band)
    return ALLOWED


def check_input(text: str, classifier: Classifier | None = None, time_limit_s: float = TIME_LIMIT_S,
                threshold: float = THRESHOLD) -> GuardResult:
    """The input gate. Rules first; then the classifier, if there is one, within the time limit."""
    started = time.perf_counter()

    def finish(result: GuardResult) -> GuardResult:
        """Stamp the result with the time taken and, for a block, the message's trigger-phrase band."""
        band = result.band or (policies.find_trigger_band(text) if result.blocked else None)
        return GuardResult(**{**result.__dict__, "band": band, "latency_ms": (time.perf_counter() - started) * 1000})

    def fail_closed(why: str) -> GuardResult:
        return finish(GuardResult(True, GuardCategory.UNVERIFIED.value, GuardLayer.FAIL_CLOSED.value,
                                  f"{why} (OP-CS-014 9.3)"))

    by_rule = check_rules(text)
    if by_rule.blocked or classifier is None:
        return finish(by_rule)
    # The classifier runs on its own thread so the gate can stop waiting at the time limit. A call that is
    # already running cannot be cancelled: its thread is left to finish and its answer is discarded.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="careflow-guard")
    try:
        risk = pool.submit(classifier, text).result(timeout=time_limit_s)
    except FutureTimeout:
        return fail_closed(f"Classifier gave no answer in {time_limit_s:g}s")
    except Exception as error:
        return fail_closed(f"Classifier failed: {type(error).__name__}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if risk.score >= threshold:
        category = GuardCategory.INJECTION if risk.category == GuardCategory.INJECTION else GuardCategory.CLINICAL
        reason = f"Classifier score {risk.score:.2f} >= {threshold:.2f}"
        return finish(GuardResult(True, category.value, GuardLayer.CLASSIFIER.value, reason, score=risk.score))
    return finish(GuardResult(False, score=risk.score, layer=GuardLayer.CLASSIFIER.value))


def check_output(reply: str) -> GuardResult:
    """Clinical content in a reply about to be sent. The fixed scripts are exempt (they name the topics
    the assistant will not discuss)."""
    text = policies.normalise_text(reply)
    for script in (DEFLECTION_SCRIPT, EMERGENCY_STATEMENT):
        text = text.replace(script, " ")
    found = _LEAK.search(text)
    if found:
        return GuardResult(True, GuardCategory.CLINICAL.value, GuardLayer.RULES.value,
                           f"Reply contains clinical wording: {found.group(0)[:40]!r}")
    return ALLOWED


def build_blocked_reply(result: GuardResult) -> str:
    """What the person is told when the input gate blocks a message."""
    return f"{EMERGENCY_STATEMENT}\n{DEFLECTION_SCRIPT}" if result.band == UrgencyBand.EMERGENT else DEFLECTION_SCRIPT
