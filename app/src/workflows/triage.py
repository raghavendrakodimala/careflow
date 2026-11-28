"""Triage workflow: a fixed LangGraph pipeline around the intake parser and the agent team.

    START -> guard -> intake -> hydrate --(clinical / Emergent / intake failed)--> deflect ------+
                          |                                                             v
                          +-> eligibility -> routing -> team -> safety_review ------> decide
                                                                                        |
                                                 (auto) finalize <----------------------+--> review -> finalize -> END

The order of the steps is fixed here, not chosen by the model. The model is used in two places only:
parsing the message (intake, the Intake agent) and writing the reply (team: the Insurance and Referral
Tracker specialists from agents.py, each the agent graph with its own prompt and tools). In
`safety_review` the Clinical Safety Reviewer reads the reply and can veto it (fail closed).

Human review is "act, then pause": OP-CS-014 6.1, 8.1 and 9.6 allow no delay to the deflection script,
the emergency statement or the Clinical Routing Ticket, so those go out before the pause. The workflow
then stops at `review` (LangGraph interrupt, state saved by the checkpointer) until a staff member
records a decision. While it waits nothing is booked for the case: the assistant is not offered
book_appointment when review is already known to be needed.

`review` contains only the interrupt. LangGraph re-runs an interrupted node from its first line on
resume, so every side effect (flag, review request) happens in the node before it.

Rules the nodes keep:
- identity: a caller the channel has not verified is matched only on full name plus member number
  (find_patient, two identifiers); a verified patient_ref is kept even when the message cannot be parsed.
- band: the SOP's own trigger phrases are a floor under the model's band, and the only band when parsing
  fails. A flag raised by the assistant is brought up to the workflow's band and reasons, never the reverse.
- memory: the contact's row in patient_interactions is updated once the route is known.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..agents import (
    REVIEWER,
    Reviewer,
    TeamResult,
    build_default_reviewer,
    record_verdict,
    review_reply,
    run_specialists,
    withhold,
)
from ..config import Settings, get_settings
from ..domain import policies
from ..domain.policies import ContactTopic, EligibilityStatus, PaymentKind, ReferralStatus, Route, UrgencyBand
from ..graph import CLINICAL_TICKET_REASON, ESCALATION_ANSWER, ESCALATION_REASON, GROUNDING_REASON, AgentDependencies
from ..guardrails import check_rules
from ..intake import parse_intake
from ..memory import PRIORITY_MARK, REPEAT_WINDOW, PatientMemory
from ..prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from ..reliability import guard_model_call
from ..schemas import DEFAULT_CHANNEL, PatientRequest
from ..state import CONFIGURABLE, THREAD_ID
from ..store import FlagCategory
from ..telemetry import REVIEWER_SPAN, SpanKind
from ..tool_contracts import CONTACT_ID, ToolName
from ..tools import CareFlowTools
from .state import CaseStatus, ReviewDecision, TriageState

# What the patient is told when a case goes to staff and no other reply was written.
HANDOFF_ANSWER = "Your request has been passed to our staff, who will contact you."
# Route reasons shown to staff on the review request and the flag.
GUARD_BLOCKED = "Input guard: {reason}"
INTAKE_FAILED = "Message could not be read automatically: failed closed and routed to a clinician (OP-CS-014 9.3)"
ASSISTANT_DEFLECTED = "The assistant deflected a clinical question (OP-CS-014 5.1)"
REVIEWER_WITHHELD = "The Clinical Safety Reviewer withheld the reply (OP-CS-014 9.6)"
SYSTEM_DOWN = "A clinic system could not be reached ({what}); nothing was assumed in its place"
REVIEW_REASON = "Triage review needed"  # start of the reason on a flag the workflow raises for a review
# The whole reply for an Urgent case. It states no time figure: SOP 4.8 is a staff procedure, not a promise
# worded for patients.
URGENT_HANDOFF = ("Your request has been marked urgent and passed to our clinical staff, who will contact you. "
                  "If you think this is an emergency, call 911 now.")
CLINICAL_ISSUE = policies.CLINICAL_ISSUE
LIVE_REFERRAL = policies.LIVE_REFERRAL_STATUSES  # referral statuses that can still lead to a booking


class TriageNode(StrEnum):
    """The workflow's nodes (see the module docstring). LangGraph, the audit trail and the trace spans are
    given the plain names."""

    GUARD = "guard"
    INTAKE = "intake"
    HYDRATE = "hydrate"
    DEFLECT = "deflect"
    ELIGIBILITY = "eligibility"
    ROUTING = "routing"
    TEAM = "team"
    SAFETY_REVIEW = "safety_review"
    DECIDE = "decide"
    REVIEW = "review"
    FINALIZE = "finalize"


# Every node's name, as it appears in audit steps and trace spans.
NODES = tuple(node.value for node in TriageNode)


@dataclass
class TriageDependencies:
    """Everything the workflow needs from outside (the seam for tests and the resilience wrappers)."""

    agent: AgentDependencies                  # tools, model and memory, shared with the assistant sub-step
    parse: Callable[[dict], PatientRequest]   # intake parser (the Intake agent)
    settings: Settings | None = None
    review: Reviewer | None = None            # Clinical Safety Reviewer; None = only the output rules check replies

    @property
    def tools(self) -> CareFlowTools:
        return self.agent.tools

    @property
    def memory(self) -> PatientMemory | None:
        return self.agent.memory

    @classmethod
    def build_default(cls, tools: CareFlowTools | None = None, settings: Settings | None = None) -> TriageDependencies:
        """Production wiring: AgentDependencies.build_default plus the model-backed intake parser and reviewer."""
        agent = AgentDependencies.build_default(tools, settings)
        parse, review = (lambda record: parse_intake(record, settings)), build_default_reviewer(settings)
        breaker = getattr(agent.model, "breaker", None)
        if breaker is not None:  # the two structured calls share the model's breaker and are traced
            name = (settings or get_settings()).careflow_model
            parse, review = guard_model_call(parse, breaker, agent.tracer, name), guard_model_call(review, breaker, agent.tracer, name)
        return cls(agent=agent, parse=parse, settings=settings, review=review)


def _get_thread_id(config: RunnableConfig) -> str:
    """The case's thread_id: also the contact_id of its flags and the thread of its memory row."""
    return config[CONFIGURABLE][THREAD_ID]


def _list_assistant_reasons(agent: dict, flags: list[dict]) -> list[str]:
    """Why staff must look at a case the team answered, from the team step's summary and its flags."""
    reasons = []
    if agent.get("error"):
        reasons.append("The assistant was unavailable")
    if agent.get("stopped_early"):
        reasons.append("The assistant reached its step limit")
    if agent.get("ungrounded"):
        reasons.append("The assistant's answer was not supported by the policy documents")
    if agent.get("nothing_found"):
        reasons.append("The policy documents do not cover the question")
    if any(not flag["automatic"] for flag in flags):
        reasons.append("The assistant asked for staff help")
    return reasons


def build_triage_graph(dependencies: TriageDependencies, checkpointer: Any = None):
    """Compile the workflow. Pausing for staff needs a `checkpointer`; callers use start_triage / resume_triage
    (workflows/__init__.py) rather than invoking the graph directly."""
    tools, store = dependencies.tools, dependencies.tools.store

    def get_now() -> datetime:
        """The clock's current time with the timezone dropped (compared with stored timestamps as naive ISO strings)."""
        return tools.clock.now().replace(tzinfo=None)

    def get_timestamp() -> str:
        return get_now().isoformat(timespec="minutes")

    def make_audit_entry(step: TriageNode, **detail) -> list[dict]:
        """One audit entry, as the one-item list the state's `audit` reducer appends."""
        return [{"step": step.value, "at": get_timestamp(), **detail}]

    def raise_flag(thread: str, reason: str, urgency: str, patient_ref: str | None, clinical: bool) -> dict:
        """Raise a staff flag for this contact (idempotent per thread). An unknown patient_ref is retried without it."""
        arguments: dict = {"reason": reason, "urgency": urgency, "clinical": clinical}
        flag_for_human, context = ToolName.FLAG_FOR_HUMAN.value, {CONTACT_ID: thread}
        result = tools.call(flag_for_human, {**arguments, "patient_ref": patient_ref} if patient_ref else arguments,
                            context=context)
        if not result.get("ok") and patient_ref:
            result = tools.call(flag_for_human, arguments, context=context)
        return result

    def identify(parsed: dict) -> str | None:
        """A caller the channel did not verify: full name plus member number must match one patient
        (two identifiers, data-handling standard 7.5). A member number alone identifies nobody."""
        name, member = (parsed.get("patient_name") or "").split(), parsed.get("insurance_member_number")
        if len(name) < 2 or not member:
            return None
        for cut in {1, len(name) - 1}:  # "Mary Ann Lee": given "Mary" or "Mary Ann"
            found = tools.call(ToolName.FIND_PATIENT.value, {
                "given_name": " ".join(name[:cut]), "family_name": " ".join(name[cut:]), "member_number": member})
            if found.get("found"):
                return found["patient_ref"]
        return None

    def get_channel(state: TriageState) -> str:
        """The channel the message came in on, for the patient's contact history."""
        return state["record"].get("channel") or DEFAULT_CHANNEL

    # ---- nodes ----------------------------------------------------------------------------------
    def guard_node(state: TriageState) -> dict:
        """The input gate, rules only. The model-based check on this path is the intake parse that follows
        (seeks_clinical_advice), so a second classifier call would add cost and nothing else."""
        gate = check_rules(state["record"].get("raw_text") or "")
        return {"guard": gate.to_dict(),
                "audit": make_audit_entry(TriageNode.GUARD, blocked=gate.blocked, category=gate.category)}

    def intake_node(state: TriageState) -> dict:
        """Parse the message (the one model call before the reply), set the band and identify the caller.
        A message the gate blocked or the parser cannot read is refused here, and `deflect` runs next."""
        record = state["record"]
        patient_ref = str(record.get("patient_ref") or "").strip().upper()
        # Verified by the channel: kept whatever happens next.
        patient_ref = patient_ref if patient_ref in tools.patients else None
        floor = policies.find_trigger_band(record.get("raw_text"))  # SOP 4.6 phrases: a floor under the model's band

        def build_refusal(reason: str, **detail) -> dict:
            """No parsed intake: the trigger phrases alone give the band, and the route is already refuse."""
            return {"intake": {}, "band": policies.find_highest_band(floor), "patient_ref": patient_ref,
                    "route": Route.REFUSE.value, "reasons": [reason],
                    "audit": make_audit_entry(TriageNode.INTAKE, **detail, trigger_band=floor)}

        gate = state.get("guard") or {}
        if gate.get("blocked"):  # stopped at the gate: the message is not sent to the model at all
            return build_refusal(GUARD_BLOCKED.format(reason=gate["reason"]), skipped="input guard")
        try:
            parsed = dependencies.parse(record).model_dump()
        except Exception as error:  # fail closed: an unreadable message is treated as clinical (OP-CS-014 9.3)
            return build_refusal(INTAKE_FAILED, error=type(error).__name__)
        band = policies.find_highest_band(parsed["urgency"], floor)
        return {"intake": parsed, "band": band, "patient_ref": patient_ref or identify(parsed),
                "audit": make_audit_entry(TriageNode.INTAKE, urgency=parsed["urgency"], trigger_band=floor,
                                          specialty=parsed.get("specialty"),
                                          seeks_clinical_advice=parsed["seeks_clinical_advice"])}

    def hydrate_node(state: TriageState, config: RunnableConfig) -> dict:
        """What the contact is about, and the repeat-contact rules for a known patient: band floor
        (SOP 4.3, 7.3) and Repeat - Priority (OP-CS-014 8.2)."""
        patient_ref, band, thread = state.get("patient_ref"), state["band"], _get_thread_id(config)
        clinical = state.get("route") == Route.REFUSE or (state.get("intake") or {}).get("seeks_clinical_advice")
        topic = (ContactTopic.CLINICAL_QUESTION.value if clinical
                 else policies.classify_intake_topic(state["record"].get("raw_text")))
        if not patient_ref or dependencies.memory is None:
            return {"repeat": None, "topic": topic}
        marker = dependencies.memory.find_repeat_marker(patient_ref, topic, exclude_thread=thread)
        if marker:  # same issue within 24h: at least the band of the earlier contact
            since = (get_now() - REPEAT_WINDOW).isoformat(timespec="minutes")
            earlier = [flag["urgency"] for flag in store.list_flags_for_patient(patient_ref)
                       if flag["created_at"] >= since and flag["contact_id"] != thread]
            band = policies.find_highest_band(band, *earlier)
        if marker == PRIORITY_MARK:  # the third contact changes the callback priority (OP-CS-014 8.2)
            band = policies.find_highest_band(band, UrgencyBand.PRIORITY.value)
        return {"band": band, "repeat": marker, "topic": topic,
                "audit": make_audit_entry(TriageNode.HYDRATE, band=band, repeat=marker, topic=topic)}

    def seeks_clinical_advice(state: TriageState) -> bool:
        """What the intake parse said. With no parse the message was not read, so it counts as clinical
        (OP-CS-014 9.3)."""
        return bool((state.get("intake") or {}).get("seeks_clinical_advice", True))

    def after_hydrate(state: TriageState) -> str:
        if state.get("route") == Route.REFUSE:  # blocked at the gate, or intake failed
            return TriageNode.DEFLECT.value
        route, _ = policies.decide_triage_route(
            seeks_clinical_advice=seeks_clinical_advice(state), urgency=state["band"])
        return (TriageNode.DEFLECT if route == Route.REFUSE else TriageNode.ELIGIBILITY).value

    def deflect_node(state: TriageState, config: RunnableConfig) -> dict:
        """The scripted reply and the Clinical Routing Ticket, at once (OP-CS-014 6.1, 9.6). Insurance is not
        raised (SOP 5.1) and nothing is referred or booked (referral policy 3.3)."""
        thread, patient_ref, band = _get_thread_id(config), state.get("patient_ref"), state["band"]
        reasons = state.get("reasons") or policies.decide_triage_route(
            seeks_clinical_advice=seeks_clinical_advice(state), urgency=band)[1]
        reason = policies.mark_hcp(CLINICAL_TICKET_REASON, state["record"].get("raw_text"))  # OP-CS-014 8.3
        reason = f"{state['repeat']}: {reason}" if state.get("repeat") else reason
        result = raise_flag(thread, reason, band, patient_ref, clinical=True)
        response = (f"{EMERGENCY_STATEMENT}\n{DEFLECTION_SCRIPT}" if band == UrgencyBand.EMERGENT
                    else DEFLECTION_SCRIPT)
        if dependencies.memory is not None and result.get("ok"):
            dependencies.memory.record_outcome(patient_ref, CLINICAL_ISSUE, result, channel=get_channel(state),
                                       thread_id=thread)
        return {"route": Route.REFUSE.value, "reasons": reasons, "response": response,
                "flag_id": result.get("flag_id"),
                "audit": make_audit_entry(TriageNode.DEFLECT, flag_id=result.get("flag_id"), urgency=band)}

    def eligibility_node(state: TriageState) -> dict:
        """Coverage and how a visit would be paid. Never a reason to refuse care (SOP 5.4)."""
        patient_ref = state.get("patient_ref")
        result = tools.call(ToolName.CHECK_ELIGIBILITY.value, {"patient_ref": patient_ref}) if patient_ref else {}
        status = result.get("eligibility_status") if result.get("ok") else EligibilityStatus.UNKNOWN.value
        payment = policies.decide_booking_payment(status, False, "").payment or PaymentKind.SELF_PAY_OFFER.value
        out = {"status": status, "plan_code": result.get("plan_code"), "new_patient": result.get("new_patient"),
               "payment": payment,
               "unavailable": bool(result.get("unavailable"))}  # the EHR did not answer; staff must check
        return {"eligibility": out, "audit": make_audit_entry(TriageNode.ELIGIBILITY, status=status, payment=payment)}

    def find_relevant_referral(patient_ref: str, specialty: str) -> dict | None:
        """The patient's referral row that decides the route for this specialty: the latest live one;
        failing that, a latest referral that was rejected still matters, because staff must see the
        rejection (and a denied pre-authorisation) before anything is booked."""
        mine = sorted((referral for referral in store.list_rows_for_patient("referrals", patient_ref)
                       if referral["to_specialty"] == specialty),
                      key=lambda referral: (referral["raised_on"], referral["referral_id"]))
        live = [referral for referral in mine if referral["status"] in LIVE_REFERRAL]
        return live[-1] if live else (mine[-1] if mine and mine[-1]["status"] == ReferralStatus.REJECTED else None)

    def summarise_referral(row: dict) -> dict:
        """The referral as the plan shows it, with its SLA date."""
        due = policies.compute_referral_due_date(date.fromisoformat(row["raised_on"]), row["urgency"])
        return {"referral_id": row["referral_id"], "status": row["status"], "urgency": row["urgency"],
                "raised_on": row["raised_on"], "sla_due_on": due.isoformat() if due else None,
                # `accepted` and `scheduled` are already actioned (policy 3.2): only `open` can be late
                "overdue": bool(due and row["status"] == ReferralStatus.OPEN and due < get_now().date())}

    def routing_node(state: TriageState) -> dict:
        """The administrative route (policies.decide_triage_route) from the referral, its pre-authorisation and the
        band, and the plan staff see. A lookup that failed always sends the case to staff."""
        patient_ref, band, specialty = state.get("patient_ref"), state["band"], state["intake"].get("specialty")
        referral, preauth, rejected = None, None, False
        down = ["eligibility"] if state["eligibility"].get("unavailable") else []  # systems that did not answer
        row = find_relevant_referral(patient_ref, specialty) if patient_ref and specialty else None
        if row:
            rejected = row["status"] == ReferralStatus.REJECTED
            referral = summarise_referral(row)
            preauth_check = tools.call(ToolName.CHECK_PREAUTH.value, {"referral_id": row["referral_id"]})
            if preauth_check.get("ok"):
                preauth = {"procedure_code": preauth_check["procedure_code"], "covered": preauth_check["covered"],
                           "preauth_required": preauth_check["preauth_required"],
                           "status": preauth_check["referral"]["preauth_status"],
                           "approval_window_business_days": preauth_check["approval_window_business_days"]}
            elif preauth_check.get("unavailable"):
                down.append("pre-authorisation")
            if not rejected:  # a rejected referral's urgency no longer counts towards the band
                band = policies.find_highest_band(band, row["urgency"])
        route, reasons = policies.decide_triage_route(
            seeks_clinical_advice=False, urgency=band, patient_known=bool(patient_ref), specialty=specialty,
            has_referral=bool(referral) and not rejected, referral_status=(referral or {}).get("status"),
            referral_rejected=rejected, preauth_status=(preauth or {}).get("status"),
            referral_overdue=bool(referral and referral["overdue"]),
            repeat_priority=state.get("repeat") == PRIORITY_MARK,
        )
        if down:  # never route on a guess when a lookup failed
            route = Route.HUMAN_REVIEW.value
            reasons = [*reasons, SYSTEM_DOWN.format(what=" and ".join(down))]
        plan = {"specialty": specialty, "sign_off_required": policies.is_sign_off_required(specialty),
                "referral": referral, "preauth": preauth, "payment": state["eligibility"]["payment"]}
        return {"band": band, "plan": plan, "route": route, "reasons": reasons,
                "audit": make_audit_entry(TriageNode.ROUTING, route=route,
                                          referral_id=(referral or {}).get("referral_id"))}

    def summarise_team(team: TeamResult, withheld: set[str]) -> dict:
        """What `decide` and staff need to know about the team step (state["agent"])."""
        return {
            "agents": team.agents, "tools_used": team.tools_used,
            "tools_offered": sorted(tools.list_available_tools() - withheld), "stopped_early": team.stopped_early,
            "flags": [{"flag_id": flag["flag_id"], "category": flag.get("category"), "urgency": flag["urgency"],
                       # raised by the graph itself (step cap, withheld answer), not asked for by the model
                       "automatic": any(mark in flag["reason"] for mark in (ESCALATION_REASON, GROUNDING_REASON))}
                      for flag in team.flags],
            "ungrounded": team.ungrounded,
            "nothing_found": team.nothing_found,
        }

    def team_node(state: TriageState, config: RunnableConfig) -> dict:
        """The reply, written by the specialists the message needs (agents.list_specialists_for). Booking is
        withheld from all of them when staff must decide first."""
        if not state.get("answer", True):
            return {"agent": {}}
        if state["band"] == UrgencyBand.URGENT:
            # SOP 4.8: a clinician contacts the patient within the hour and nothing is booked before that, so
            # the assistant has nothing to add. A fixed reply: the model is not asked to triage the case
            # again, because a specialist given the raw message can deflect it as clinical and so overrule
            # the route decided here.
            return {"agent": {}, "response": URGENT_HANDOFF,
                    "audit": make_audit_entry(TriageNode.TEAM, skipped="Urgent band")}
        withheld = {ToolName.BOOK_APPOINTMENT.value} if state["route"] != Route.AUTO else set()
        try:
            team = run_specialists(state["record"]["raw_text"], dependencies.agent, patient_ref=state.get("patient_ref"),
                                   channel=get_channel(state), thread_id=_get_thread_id(config), withheld=withheld)
        except Exception as error:  # the model is down: hand to staff, never guess
            return {"agent": {"error": type(error).__name__}, "response": ESCALATION_ANSWER,
                    "audit": make_audit_entry(TriageNode.TEAM, error=type(error).__name__)}
        return {"agent": summarise_team(team, withheld), "response": team.answer,
                "audit": make_audit_entry(TriageNode.TEAM, agents=team.agents, tools=team.tools_used)}

    def safety_review_node(state: TriageState, config: RunnableConfig) -> dict:
        """The Clinical Safety Reviewer. A veto, or no verdict, replaces the reply with the deflection
        script and raises a Clinical Routing Ticket (fail closed, OP-CS-014 9.3 and 9.6)."""
        agent = state.get("agent") or {}
        if not agent or agent.get("error"):
            return {}  # nothing was written by the team
        with dependencies.agent.tracer.open_span(REVIEWER_SPAN, kind=SpanKind.GUARDRAIL, agent=REVIEWER.name) as span:
            verdict = review_reply(state["record"]["raw_text"], state.get("response") or "", dependencies.review)
            record_verdict(span, verdict)
        if verdict.safe:
            return {"verdict": verdict.model_dump(), "audit": make_audit_entry(TriageNode.SAFETY_REVIEW, safe=True)}
        ticket = withhold(dependencies.agent, state.get("patient_ref"), _get_thread_id(config), urgency=state["band"],
                          message=state["record"].get("raw_text") or "", channel=get_channel(state))
        flags = [*(agent.get("flags") or []),
                 {"flag_id": ticket.get("flag_id"), "category": FlagCategory.CLINICAL.value,
                  "urgency": ticket.get("urgency") or state["band"], "automatic": True}]
        return {"verdict": verdict.model_dump(), "response": DEFLECTION_SCRIPT,
                "agent": {**agent, "flags": flags, "vetoed": True},
                "audit": make_audit_entry(TriageNode.SAFETY_REVIEW, safe=False, category=verdict.category,
                                          flag_id=ticket.get("flag_id"))}

    def decide_node(state: TriageState, config: RunnableConfig) -> dict:
        """The final route. A non-auto route always has a staff flag and a review request before the pause,
        and the flag carries the workflow's band, repeat marker and reasons whoever raised it."""
        thread, route, reasons = _get_thread_id(config), state["route"], list(state.get("reasons") or [])
        agent, flag_id, patient_ref = state.get("agent") or {}, state.get("flag_id"), state.get("patient_ref")
        deflected = route == Route.REFUSE  # deflect already raised and recorded the clinical ticket
        flags = agent.get("flags") or []
        band = policies.find_highest_band(state["band"], *(flag["urgency"] for flag in flags))
        adopted = None
        if not deflected:
            clinical = [flag for flag in flags if flag["category"] == FlagCategory.CLINICAL]
            if clinical:  # a refusal is only a refusal: the administrative reasons no longer apply
                why = REVIEWER_WITHHELD if agent.get("vetoed") else ASSISTANT_DEFLECTED
                route, adopted, reasons = Route.REFUSE.value, clinical[-1]["flag_id"], [why]
            else:
                reasons += _list_assistant_reasons(agent, flags)
                if reasons:  # the routing step's reasons or the assistant's: either sends the case to staff
                    route = Route.HUMAN_REVIEW.value
                    adopted = flags[0]["flag_id"] if flags else None
        update: dict = {"route": route, "reasons": reasons, "band": band}
        if route == Route.AUTO:
            return {**update, "audit": make_audit_entry(TriageNode.DECIDE, route=route)}

        # One flag per case: the assistant's own is adopted, deflect's is kept, otherwise one is raised here.
        detail, marker = "; ".join(reasons), state.get("repeat")
        if adopted:  # raised by the assistant: bring it up to the workflow's band, marker and reasons
            flag_id = store.promote_flag(adopted, band, policies.URGENCY_ORDER, prefix=marker, detail=detail)["flag_id"]
        elif not flag_id:
            reason = f"{REVIEW_REASON}: {detail}"
            flag_id = raise_flag(thread, f"{marker}: {reason}" if marker else reason, band, patient_ref,
                                 clinical=False).get("flag_id")
        if not deflected and dependencies.memory is not None and flag_id:
            topic = (CLINICAL_ISSUE if route == Route.REFUSE
                     else state.get("topic") or ContactTopic.APPOINTMENT_BOOKING.value)
            dependencies.memory.record_outcome(patient_ref, topic, store.get_flag(flag_id), channel=get_channel(state),
                                       thread_id=thread)
        review = store.add_review(thread, route, reasons, get_timestamp(), flag_id=flag_id, patient_ref=patient_ref)
        update.update(flag_id=flag_id, review_id=review["review_id"], status=CaseStatus.AWAITING_HUMAN_REVIEW.value)
        if not state.get("response"):
            update["response"] = HANDOFF_ANSWER
        return {**update,
                "audit": make_audit_entry(TriageNode.DECIDE, route=route, review_id=review["review_id"],
                                          flag_id=flag_id)}

    def review_node(state: TriageState) -> dict:
        """The pause for staff. Only the interrupt and the validation of what it returns live here: the node
        is re-run from its first line on resume, so it must have no side effect."""
        raw = interrupt({
            "review_id": state.get("review_id"), "flag_id": state.get("flag_id"), "route": state["route"],
            "reasons": state.get("reasons"), "band": state["band"], "patient_ref": state.get("patient_ref"),
            "plan": state.get("plan"), "response_sent": state.get("response"),
        })
        decision = ReviewDecision.model_validate(raw)
        problem = policies.validate_reband(state["band"], decision.urgency, decision.note)
        if problem:  # also checked in resume_triage; here so a direct Command(resume=...) cannot bypass it
            raise ValueError(problem)
        return {"decision": decision.model_dump()}

    def finalize_node(state: TriageState) -> dict:
        """Close the case. After a review, the staff decision (and any re-band) is written to the review request."""
        decision = state.get("decision")
        if not decision:
            return {"status": CaseStatus.COMPLETED.value,
                    "audit": make_audit_entry(TriageNode.FINALIZE, route=state["route"])}
        store.decide_review(state["review_id"], decision["reviewer_id"], decision["action"], get_timestamp(),
                            note=decision.get("note"), urgency=decision.get("urgency"))
        return {"status": CaseStatus.COMPLETED.value, "band": decision.get("urgency") or state["band"],
                "audit": make_audit_entry(TriageNode.FINALIZE, route=state["route"],
                                          reviewer_id=decision["reviewer_id"], action=decision["action"])}

    def after_decide(state: TriageState) -> str:
        return (TriageNode.FINALIZE if state["route"] == Route.AUTO else TriageNode.REVIEW).value

    def wrap_in_span(name: str, node_function):
        """The node with a span around it. functools.wraps keeps the signature LangGraph inspects."""
        if not dependencies.agent.tracer.enabled:
            return node_function

        @functools.wraps(node_function)
        def run_in_span(state: TriageState, *args, **kwargs):
            with dependencies.agent.tracer.open_span(f"step.{name}", step=name, patient_ref=state.get("patient_ref")) as span:
                update = node_function(state, *args, **kwargs)
                merged = {**state, **(update or {})}
                span.set(route=merged.get("route") or None, band=merged.get("band"), flag_id=merged.get("flag_id"),
                         review_id=merged.get("review_id"), safe=(merged.get("verdict") or {}).get("safe"))
                return update

        return run_in_span

    graph = StateGraph(TriageState)
    # LangGraph is given every node's plain name (`.value`): the names are stored in checkpoints.
    node = TriageNode
    for step, node_function in ((node.GUARD, guard_node), (node.INTAKE, intake_node), (node.HYDRATE, hydrate_node),
                     (node.DEFLECT, deflect_node), (node.ELIGIBILITY, eligibility_node), (node.ROUTING, routing_node),
                     (node.TEAM, team_node), (node.SAFETY_REVIEW, safety_review_node), (node.DECIDE, decide_node),
                     (node.REVIEW, review_node), (node.FINALIZE, finalize_node)):
        graph.add_node(step.value, wrap_in_span(step.value, node_function))
    graph.add_edge(START, node.GUARD.value)
    graph.add_edge(node.GUARD.value, node.INTAKE.value)
    graph.add_edge(node.INTAKE.value, node.HYDRATE.value)
    graph.add_conditional_edges(node.HYDRATE.value, after_hydrate, {
        node.DEFLECT.value: node.DEFLECT.value, node.ELIGIBILITY.value: node.ELIGIBILITY.value})
    graph.add_edge(node.DEFLECT.value, node.DECIDE.value)
    graph.add_edge(node.ELIGIBILITY.value, node.ROUTING.value)
    graph.add_edge(node.ROUTING.value, node.TEAM.value)
    graph.add_edge(node.TEAM.value, node.SAFETY_REVIEW.value)
    graph.add_edge(node.SAFETY_REVIEW.value, node.DECIDE.value)
    graph.add_conditional_edges(node.DECIDE.value, after_decide, {
        node.FINALIZE.value: node.FINALIZE.value, node.REVIEW.value: node.REVIEW.value})
    graph.add_edge(node.REVIEW.value, node.FINALIZE.value)
    graph.add_edge(node.FINALIZE.value, END)
    return graph.compile(checkpointer=checkpointer)
