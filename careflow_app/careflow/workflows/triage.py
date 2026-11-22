"""M5 triage workflow: a fixed LangGraph pipeline around the M1 intake parser and the M2-M4 agent.

    START -> guard -> intake -> hydrate --(clinical / Emergent / intake failed)--> deflect ------+
                          |                                                             v
                          +-> eligibility -> routing -> team -> safety_review ------> decide
                                                                                        |
                                                 (auto) finalize <----------------------+--> review -> finalize -> END

The order of the steps is fixed here, not chosen by the model. The model is used in two places only:
parsing the message (intake, the Intake agent) and writing the reply (team: the Insurance and Referral
Tracker specialists from agents.py, each the existing agent graph with its own prompt and tools). M6
adds `safety_review`: the Clinical Safety Reviewer reads the reply and can veto it (fail closed).

Human review (owner decision 2026-11-21, "act, then pause"): OP-CS-014 6.1, 8.1 and 9.6 allow no delay
to the deflection script, the emergency statement or the Clinical Routing Ticket, so those go out
before the pause. The workflow then stops at `review` (LangGraph interrupt, state saved by the
checkpointer) until a staff member records a decision. While it waits nothing is booked for the case:
the assistant is not offered book_appointment when review is already known to be needed.

`review` contains only the interrupt. LangGraph re-runs an interrupted node from its first line on
resume, so every side effect (flag, review request) happens in the node before it.

Rules added after the architect's M5 review (feedback.md):
- identity: a caller the channel has not verified is matched only on full name plus member number
  (find_patient, two identifiers); a verified patient_ref is kept even when the message cannot be parsed.
- band: the SOP's own trigger phrases are a floor under the model's band, and the only band when parsing
  fails. A flag raised by the assistant is brought up to the workflow's band and reasons, never the reverse.
- memory: the contact's row in patient_interactions is updated once the route is known.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..agents import Reviewer, default_reviewer, review_reply, run_specialists, withhold
from ..config import Settings, get_settings
from ..domain import policies
from ..guardrails import check_rules
from ..graph import CLINICAL_TICKET_REASON, ESCALATION_ANSWER, ESCALATION_REASON, GROUNDING_REASON, Deps
from ..intake import parse_intake
from ..memory import PRIORITY_MARK, REPEAT_WINDOW, PatientMemory
from ..prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from ..reliability import guarded
from ..schemas import PatientRequest
from ..tools import CareFlowTools
from .state import ReviewDecision, TriageState

HANDOFF_ANSWER = "Your request has been passed to our staff, who will contact you."
GUARD_BLOCKED = "Input guard: {reason}"
INTAKE_FAILED = "Message could not be read automatically: failed closed and routed to a clinician (OP-CS-014 9.3)"
ASSISTANT_DEFLECTED = "The assistant deflected a clinical question (OP-CS-014 5.1)"
REVIEWER_WITHHELD = "The Clinical Safety Reviewer withheld the reply (OP-CS-014 9.6)"
REVIEW_REASON = "Triage review needed"
# The whole reply for an Urgent case. It states no time figure: SOP 4.8 is a staff procedure, not a promise
# worded for patients.
URGENT_HANDOFF = ("Your request has been marked urgent and passed to our clinical staff, who will contact you. "
                  "If you think this is an emergency, call 911 now.")
SYSTEM_DOWN = "A clinic system could not be reached ({what}); nothing was assumed in its place"
CLINICAL_ISSUE = "clinical_question"
LIVE_REFERRAL = ("open", "accepted", "scheduled")
NODES = ("guard", "intake", "hydrate", "deflect", "eligibility", "routing", "team", "safety_review", "decide", "review",
         "finalize")


@dataclass
class TriageDeps:
    """Everything the workflow needs from outside (the seam for tests and M7 hardening)."""

    agent: Deps                               # tools, model and memory, shared with the assistant sub-step
    parse: Callable[[dict], PatientRequest]   # M1 intake parser (the Intake agent)
    settings: Settings | None = None
    review: Reviewer | None = None            # M6 Clinical Safety Reviewer; None = replies are not reviewed

    @property
    def tools(self) -> CareFlowTools:
        return self.agent.tools

    @property
    def memory(self) -> PatientMemory | None:
        return self.agent.memory

    @classmethod
    def default(cls, tools: CareFlowTools | None = None, settings: Settings | None = None) -> TriageDeps:
        agent = Deps.default(tools, settings)
        parse, review = (lambda record: parse_intake(record, settings)), default_reviewer(settings)
        breaker = getattr(agent.model, "breaker", None)
        if breaker is not None:  # M7-1: the two structured calls share the model's breaker and are traced
            name = (settings or get_settings()).careflow_model
            parse, review = guarded(parse, breaker, agent.tracer, name), guarded(review, breaker, agent.tracer, name)
        return cls(agent=agent, parse=parse, settings=settings, review=review)


def _thread(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def build_triage_graph(deps: TriageDeps, checkpointer: Any = None):
    tools, store = deps.tools, deps.tools.store

    def now() -> datetime:
        return tools.clock.now().replace(tzinfo=None)

    def stamp() -> str:
        return now().isoformat(timespec="minutes")

    def audit(step: str, **detail) -> list[dict]:
        return [{"step": step, "at": stamp(), **detail}]

    def flag(thread: str, reason: str, urgency: str, patient_ref: str | None, clinical: bool) -> dict:
        """Raise a staff flag for this contact (idempotent per thread). An unknown patient_ref is retried without it."""
        args: dict = {"reason": reason, "urgency": urgency, "clinical": clinical}
        result = tools.call("flag_for_human", {**args, "patient_ref": patient_ref} if patient_ref else args,
                            context={"contact_id": thread})
        if not result.get("ok") and patient_ref:
            result = tools.call("flag_for_human", args, context={"contact_id": thread})
        return result

    def identify(parsed: dict) -> str | None:
        """A caller the channel did not verify: full name plus member number must match one patient
        (two identifiers, data-handling standard 7.5). A member number alone identifies nobody."""
        name, member = (parsed.get("patient_name") or "").split(), parsed.get("insurance_member_number")
        if len(name) < 2 or not member:
            return None
        for cut in {1, len(name) - 1}:  # "Mary Ann Lee": given "Mary" or "Mary Ann"
            found = tools.call("find_patient", {"given_name": " ".join(name[:cut]), "family_name": " ".join(name[cut:]),
                                                "member_number": member})
            if found.get("found"):
                return found["patient_ref"]
        return None

    def channel(state: TriageState) -> str:
        return state["record"].get("channel") or "web_form"

    # ---- nodes ----------------------------------------------------------------------------------
    def guard_node(state: TriageState) -> dict:
        """M8 input gate, rules only. The model-based check on this path is the intake parse that follows
        (seeks_clinical_advice), so a second classifier call would add cost and nothing else."""
        gate = check_rules(state["record"].get("raw_text") or "")
        return {"guard": gate.as_dict(), "audit": audit("guard", blocked=gate.blocked, category=gate.category)}

    def intake_node(state: TriageState) -> dict:
        record = state["record"]
        ref = str(record.get("patient_ref") or "").strip().upper()
        ref = ref if ref in tools.patients else None  # verified by the channel: kept whatever happens next
        floor = policies.trigger_band(record.get("raw_text"))  # SOP 4.6 phrases: a floor under the model's band
        gate = state.get("guard") or {}
        if gate.get("blocked"):  # stopped at the gate: the message is not sent to the model at all
            return {"intake": {}, "band": policies.highest_band(floor), "patient_ref": ref,
                    "route": policies.ROUTE_REFUSE, "reasons": [GUARD_BLOCKED.format(reason=gate["reason"])],
                    "audit": audit("intake", skipped="input guard", trigger_band=floor)}
        try:
            parsed = deps.parse(record).model_dump()
        except Exception as exc:  # fail closed: an unreadable message is treated as clinical (OP-CS-014 9.3)
            return {"intake": {}, "band": policies.highest_band(floor), "patient_ref": ref,
                    "route": policies.ROUTE_REFUSE, "reasons": [INTAKE_FAILED],
                    "audit": audit("intake", error=type(exc).__name__, trigger_band=floor)}
        band = policies.highest_band(parsed["urgency"], floor)
        return {"intake": parsed, "band": band, "patient_ref": ref or identify(parsed),
                "audit": audit("intake", urgency=parsed["urgency"], trigger_band=floor, specialty=parsed.get("specialty"),
                               seeks_clinical_advice=parsed["seeks_clinical_advice"])}

    def hydrate_node(state: TriageState, config: RunnableConfig) -> dict:
        """What the contact is about, and the repeat-contact rules for a known patient: band floor
        (SOP 4.3, 7.3) and Repeat - Priority (OP-CS-014 8.2)."""
        ref, band, thread = state.get("patient_ref"), state["band"], _thread(config)
        clinical = state.get("route") == policies.ROUTE_REFUSE or (state.get("intake") or {}).get("seeks_clinical_advice")
        topic = CLINICAL_ISSUE if clinical else policies.intake_topic(state["record"].get("raw_text"))
        if not ref or deps.memory is None:
            return {"repeat": None, "topic": topic}
        marker = deps.memory.repeat_marker(ref, topic, exclude_thread=thread)
        if marker:  # same issue within 24h: at least the band of the earlier contact
            since = (now() - REPEAT_WINDOW).isoformat(timespec="minutes")
            earlier = [f["urgency"] for f in store.flags_for_patient(ref)
                       if f["created_at"] >= since and f["contact_id"] != thread]
            band = policies.highest_band(band, *earlier)
        if marker == PRIORITY_MARK:  # the third contact changes the callback priority (OP-CS-014 8.2)
            band = policies.highest_band(band, "Priority")
        return {"band": band, "repeat": marker, "topic": topic,
                "audit": audit("hydrate", band=band, repeat=marker, topic=topic)}

    def after_hydrate(state: TriageState) -> str:
        if state.get("route") == policies.ROUTE_REFUSE:  # intake failed
            return "deflect"
        route, _ = policies.triage_route(
            seeks_clinical_advice=state["intake"]["seeks_clinical_advice"], urgency=state["band"])
        return "deflect" if route == policies.ROUTE_REFUSE else "eligibility"

    def deflect_node(state: TriageState, config: RunnableConfig) -> dict:
        """The scripted reply and the Clinical Routing Ticket, at once (OP-CS-014 6.1, 9.6). Insurance is not
        raised (SOP 5.1) and nothing is referred or booked (referral policy 3.3)."""
        thread, ref, band = _thread(config), state.get("patient_ref"), state["band"]
        reasons = state.get("reasons") or policies.triage_route(
            seeks_clinical_advice=state["intake"]["seeks_clinical_advice"], urgency=band)[1]
        reason = policies.mark_hcp(CLINICAL_TICKET_REASON, state["record"].get("raw_text"))  # OP-CS-014 8.3
        reason = f"{state['repeat']}: {reason}" if state.get("repeat") else reason
        result = flag(thread, reason, band, ref, clinical=True)
        response = f"{EMERGENCY_STATEMENT}\n{DEFLECTION_SCRIPT}" if band == "Emergent" else DEFLECTION_SCRIPT
        if deps.memory is not None and result.get("ok"):
            deps.memory.record_outcome(ref, CLINICAL_ISSUE, result, channel=channel(state), thread_id=thread)
        return {"route": policies.ROUTE_REFUSE, "reasons": reasons, "response": response,
                "flag_id": result.get("flag_id"),
                "audit": audit("deflect", flag_id=result.get("flag_id"), urgency=band)}

    def eligibility_node(state: TriageState) -> dict:
        """Coverage and how a visit would be paid. Never a reason to refuse care (SOP 5.4)."""
        ref = state.get("patient_ref")
        r = tools.call("check_eligibility", {"patient_ref": ref}) if ref else {}
        status = r.get("eligibility_status") if r.get("ok") else "unknown"
        payment = policies.booking_payment(status, False, "").payment or "self_pay_offer"
        out = {"status": status, "plan_code": r.get("plan_code"), "new_patient": r.get("new_patient"), "payment": payment,
               "unavailable": bool(r.get("unavailable"))}  # M7: the EHR did not answer; staff must check
        return {"eligibility": out, "audit": audit("eligibility", status=status, payment=payment)}

    def routing_node(state: TriageState) -> dict:
        ref, band, specialty = state.get("patient_ref"), state["band"], state["intake"].get("specialty")
        referral, preauth, rejected = None, None, False
        down = ["eligibility"] if state["eligibility"].get("unavailable") else []
        if ref and specialty:
            mine = sorted((r for r in store.rows_for_patient("referrals", ref) if r["to_specialty"] == specialty),
                          key=lambda r: (r["raised_on"], r["referral_id"]))
            live = [r for r in mine if r["status"] in LIVE_REFERRAL]
            # The latest live referral; failing that, a latest referral that was rejected still matters:
            # staff must see the rejection (and a denied pre-authorisation) before anything is booked.
            row = live[-1] if live else (mine[-1] if mine and mine[-1]["status"] == "rejected" else None)
            if row:
                rejected = row["status"] == "rejected"
                due = policies.referral_due(date.fromisoformat(row["raised_on"]), row["urgency"])
                referral = {"referral_id": row["referral_id"], "status": row["status"], "urgency": row["urgency"],
                            "raised_on": row["raised_on"], "sla_due_on": due.isoformat() if due else None,
                            # `accepted` and `scheduled` are already actioned (policy 3.2): only `open` can be late
                            "overdue": bool(due and row["status"] == "open" and due < now().date())}
                p = tools.call("check_preauth", {"referral_id": row["referral_id"]})
                if p.get("ok"):
                    preauth = {"procedure_code": p["procedure_code"], "covered": p["covered"],
                               "preauth_required": p["preauth_required"], "status": p["referral"]["preauth_status"],
                               "approval_window_business_days": p["approval_window_business_days"]}
                elif p.get("unavailable"):
                    down.append("pre-authorisation")
                if not rejected:
                    band = policies.highest_band(band, row["urgency"])
        route, reasons = policies.triage_route(
            seeks_clinical_advice=False, urgency=band, patient_known=bool(ref), specialty=specialty,
            has_referral=bool(referral) and not rejected, referral_status=(referral or {}).get("status"),
            referral_rejected=rejected, preauth_status=(preauth or {}).get("status"),
            referral_overdue=bool(referral and referral["overdue"]),
            repeat_priority=state.get("repeat") == PRIORITY_MARK,
        )
        if down:  # M7: never route on a guess when a lookup failed
            route = policies.ROUTE_HUMAN
            reasons = [*reasons, SYSTEM_DOWN.format(what=" and ".join(down))]
        plan = {"specialty": specialty, "sign_off_required": policies.sign_off_required(specialty),
                "referral": referral, "preauth": preauth, "payment": state["eligibility"]["payment"]}
        return {"band": band, "plan": plan, "route": route, "reasons": reasons,
                "audit": audit("routing", route=route, referral_id=(referral or {}).get("referral_id"))}

    def team_node(state: TriageState, config: RunnableConfig) -> dict:
        """The reply, written by the specialists the message needs (agents.specialists_for). Booking is
        withheld from all of them when staff must decide first."""
        if not state.get("answer", True):
            return {"agent": {}}
        if state["band"] == "Urgent":
            # SOP 4.8: a clinician contacts the patient within the hour and nothing is booked before that, so
            # the assistant has nothing to add. A fixed reply: the model is not asked to triage the case again
            # (live M5 demo, 2026-11-22: a specialist given the raw message deflected it as clinical, and that
            # overruled the route decided here).
            return {"agent": {}, "response": URGENT_HANDOFF, "audit": audit("team", skipped="Urgent band")}
        withheld = {"book_appointment"} if state["route"] != policies.ROUTE_AUTO else set()
        try:
            team = run_specialists(state["record"]["raw_text"], deps.agent, patient_ref=state.get("patient_ref"),
                                   channel=channel(state), thread_id=_thread(config), withheld=withheld)
        except Exception as exc:  # the model is down: hand to staff, never guess
            return {"agent": {"error": type(exc).__name__}, "response": ESCALATION_ANSWER,
                    "audit": audit("team", error=type(exc).__name__)}
        flags = [c["result"] for c in team.tool_calls if c["name"] == "flag_for_human" and c["result"].get("ok")]
        searches = [c["result"] for c in team.tool_calls if c["name"] == "search_policies"]
        summary = {
            "agents": team.agents, "tools_used": team.tools_used,
            "tools_offered": sorted(tools.available_tools() - withheld), "stopped_early": team.stopped_early,
            "flags": [{"flag_id": f["flag_id"], "category": f.get("category"), "urgency": f["urgency"],
                       # raised by the graph itself (step cap, withheld answer), not asked for by the model
                       "automatic": any(mark in f["reason"] for mark in (ESCALATION_REASON, GROUNDING_REASON))}
                      for f in flags],
            "ungrounded": team.ungrounded,
            "nothing_found": bool(searches) and all(s.get("abstained") or not s.get("count") for s in searches),
        }
        return {"agent": summary, "response": team.answer,
                "audit": audit("team", agents=team.agents, tools=team.tools_used)}

    def safety_review_node(state: TriageState, config: RunnableConfig) -> dict:
        """The Clinical Safety Reviewer. A veto, or no verdict, replaces the reply with the deflection
        script and raises a Clinical Routing Ticket (fail closed, OP-CS-014 9.3 and 9.6)."""
        agent = state.get("agent") or {}
        if not agent or agent.get("error"):
            return {}  # nothing was written by the team
        with deps.agent.tracer.span("agent.clinical_safety_reviewer", kind="guardrail",
                                    agent="clinical_safety_reviewer") as span:
            verdict = review_reply(state["record"]["raw_text"], state.get("response") or "", deps.review)
            span.set(safe=verdict.safe, category=verdict.category)
        if verdict.safe:
            return {"verdict": verdict.model_dump(), "audit": audit("safety_review", safe=True)}
        ticket = withhold(deps.agent, state.get("patient_ref"), _thread(config), urgency=state["band"],
                          message=state["record"].get("raw_text") or "", channel=channel(state))
        flags = [*(agent.get("flags") or []),
                 {"flag_id": ticket.get("flag_id"), "category": "clinical", "urgency": ticket.get("urgency") or state["band"],
                  "automatic": True}]
        return {"verdict": verdict.model_dump(), "response": DEFLECTION_SCRIPT,
                "agent": {**agent, "flags": flags, "vetoed": True},
                "audit": audit("safety_review", safe=False, category=verdict.category, flag_id=ticket.get("flag_id"))}

    def decide_node(state: TriageState, config: RunnableConfig) -> dict:
        """The final route. A non-auto route always has a staff flag and a review request before the pause,
        and the flag carries the workflow's band, repeat marker and reasons whoever raised it."""
        thread, route, reasons = _thread(config), state["route"], list(state.get("reasons") or [])
        agent, flag_id, ref = state.get("agent") or {}, state.get("flag_id"), state.get("patient_ref")
        deflected = route == policies.ROUTE_REFUSE  # deflect already raised and recorded the clinical ticket
        flags = agent.get("flags") or []
        band = policies.highest_band(state["band"], *(f["urgency"] for f in flags))
        adopted = None
        if not deflected:
            clinical = [f for f in flags if f["category"] == "clinical"]
            if clinical:  # a refusal is only a refusal: the administrative reasons no longer apply
                why = REVIEWER_WITHHELD if agent.get("vetoed") else ASSISTANT_DEFLECTED
                route, adopted, reasons = policies.ROUTE_REFUSE, clinical[-1]["flag_id"], [why]
            else:
                if agent.get("error"):
                    reasons.append("The assistant was unavailable")
                if agent.get("stopped_early"):
                    reasons.append("The assistant reached its step limit")
                if agent.get("ungrounded"):
                    reasons.append("The assistant's answer was not supported by the policy documents")
                if agent.get("nothing_found"):
                    reasons.append("The policy documents do not cover the question")
                if any(not f["automatic"] for f in flags):
                    reasons.append("The assistant asked for staff help")
                if reasons:
                    route = policies.ROUTE_HUMAN
                    adopted = flags[0]["flag_id"] if flags else None
        update: dict = {"route": route, "reasons": reasons, "band": band}
        if route == policies.ROUTE_AUTO:
            return {**update, "audit": audit("decide", route=route)}

        detail, marker = "; ".join(reasons), state.get("repeat")
        if adopted:  # raised by the assistant: bring it up to the workflow's band, marker and reasons
            flag_id = store.promote_flag(adopted, band, policies.URGENCY_ORDER, prefix=marker, detail=detail)["flag_id"]
        elif not flag_id:
            reason = f"{REVIEW_REASON}: {detail}"
            flag_id = flag(thread, f"{marker}: {reason}" if marker else reason, band, ref, clinical=False).get("flag_id")
        if not deflected and deps.memory is not None and flag_id:
            topic = CLINICAL_ISSUE if route == policies.ROUTE_REFUSE else state.get("topic") or "appointment_booking"
            deps.memory.record_outcome(ref, topic, store.flag(flag_id), channel=channel(state), thread_id=thread)
        review = store.add_review(thread, route, reasons, stamp(), flag_id=flag_id, patient_ref=ref)
        update.update(flag_id=flag_id, review_id=review["review_id"], status="awaiting_human_review")
        if not state.get("response"):
            update["response"] = HANDOFF_ANSWER
        return {**update, "audit": audit("decide", route=route, review_id=review["review_id"], flag_id=flag_id)}

    def review_node(state: TriageState) -> dict:
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
        decision = state.get("decision")
        if not decision:
            return {"status": "completed", "audit": audit("finalize", route=state["route"])}
        store.decide_review(state["review_id"], decision["reviewer_id"], decision["action"], stamp(),
                            note=decision.get("note"), urgency=decision.get("urgency"))
        return {"status": "completed", "band": decision.get("urgency") or state["band"],
                "audit": audit("finalize", route=state["route"], reviewer_id=decision["reviewer_id"],
                               action=decision["action"])}

    def after_decide(state: TriageState) -> str:
        return "finalize" if state["route"] == policies.ROUTE_AUTO else "review"

    def traced(name: str, fn):
        """The node with a span around it. functools.wraps keeps the signature LangGraph inspects."""
        if not deps.agent.tracer.enabled:
            return fn

        @functools.wraps(fn)
        def node(state: TriageState, *args, **kwargs):
            with deps.agent.tracer.span(f"step.{name}", step=name, patient_ref=state.get("patient_ref")) as span:
                update = fn(state, *args, **kwargs)
                merged = {**state, **(update or {})}
                span.set(route=merged.get("route") or None, band=merged.get("band"), flag_id=merged.get("flag_id"),
                         review_id=merged.get("review_id"), safe=(merged.get("verdict") or {}).get("safe"))
                return update

        return node

    g = StateGraph(TriageState)
    for name, fn in (("guard", guard_node), ("intake", intake_node), ("hydrate", hydrate_node), ("deflect", deflect_node),
                     ("eligibility", eligibility_node), ("routing", routing_node), ("team", team_node),
                     ("safety_review", safety_review_node), ("decide", decide_node), ("review", review_node), ("finalize", finalize_node)):
        g.add_node(name, traced(name, fn))
    g.add_edge(START, "guard")
    g.add_edge("guard", "intake")
    g.add_edge("intake", "hydrate")
    g.add_conditional_edges("hydrate", after_hydrate, {"deflect": "deflect", "eligibility": "eligibility"})
    g.add_edge("deflect", "decide")
    g.add_edge("eligibility", "routing")
    g.add_edge("routing", "team")
    g.add_edge("team", "safety_review")
    g.add_edge("safety_review", "decide")
    g.add_conditional_edges("decide", after_decide, {"finalize": "finalize", "review": "review"})
    g.add_edge("review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
