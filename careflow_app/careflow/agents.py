"""M6: the agent team. Four agents with separate jobs, prompts and tools:

    Intake                    reads the message into a PatientRequest and identifies the caller. It is the
                              structured-output step from M1 (intake.parse_intake); in the triage workflow
                              it is the `intake` node.
    Insurance                 coverage, pre-authorisation, costs and policy questions.
    Referral Tracker          referral status, appointment slots and booking (the EHR tools, over MCP when
                              the tools object is an McpRoutedTools).
    Clinical Safety Reviewer  reads every reply before it is sent and vetoes clinical content.

Insurance and Referral Tracker are the existing agent graph (graph.build_agent_graph) with their own
prompt and tool subset, so each keeps the memory note, the grounding check and the clinical-ticket rule.
Which specialist handles a message is decided in code from its wording (ADR-04: the model does not choose
the route). A message that needs both runs Insurance first and gives its answer to the Referral Tracker.

The reviewer returns a structured verdict. A veto, or any failure to get a verdict, replaces the reply
with the deflection script and raises a Clinical Routing Ticket: fail closed (OP-CS-014 9.3, 9.6).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .agent import AgentResult, PatientMismatch, redact_answers, run_agent, thread_patient
from .config import Settings
from .domain import policies
from .graph import DEFLECTION_MARK, Deps
from .guardrails import Classifier, GuardResult, blocked_reply, check_input, check_output
from .intake import parse_intake
from .memory import PRIORITY_MARK
from .llm_client import complete_structured
from .prompts import DEFLECTION_SCRIPT, INSURANCE_PROMPT, REFERRAL_PROMPT, REVIEWER_PROMPT

REVIEW_TICKET_REASON = "Reply withheld by the Clinical Safety Reviewer; Clinical Routing Ticket created (OP-CS-014 9.6)"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    job: str
    prompt: str
    tools: frozenset[str]


INTAKE = AgentSpec("intake", "Read the message into a PatientRequest and identify the caller.", "", frozenset())
INSURANCE = AgentSpec(
    "insurance", "Coverage, pre-authorisation, costs and policy questions.", INSURANCE_PROMPT,
    frozenset({"check_eligibility", "check_preauth", "calculate_copay", "search_policies", "find_patient",
               "flag_for_human"}),
)
REFERRAL = AgentSpec(
    "referral_tracker", "Referral status, appointment slots and booking.", REFERRAL_PROMPT,
    # check_eligibility: the shared rules say "check eligibility before booking"; it gives no cost figure.
    frozenset({"check_eligibility", "check_preauth", "find_appointment_slots", "book_appointment", "search_policies",
               "find_patient", "flag_for_human"}),
)
REVIEWER = AgentSpec("clinical_safety_reviewer", "Veto any reply with clinical content.", REVIEWER_PROMPT, frozenset())
TEAM = (INTAKE, INSURANCE, REFERRAL, REVIEWER)
SPECIALISTS = {a.name: a for a in (INSURANCE, REFERRAL)}

# Wording that points to each specialist. Insurance is the default: it holds the policy search, so a
# general question about a rule or a timing goes there.
_SCHEDULING = re.compile(
    r"\b(?:book|booking|appointment|appt|slot|reschedul\w*|schedule|referr\w*|see (?:a|an|the|my) \w+|"
    r"earliest|next available|telehealth visit)\b|MHP-R-\d", re.I)
_COVERAGE = re.compile(
    r"\b(?:co-?pay\w*|cost\w*|price|owe|bill\w*|statement|deductible|coinsurance|eligib\w*|cover\w*|insur\w*|"
    r"pre-?auth\w*|plan|self-?pay|how much)\b", re.I)


def specialists_for(message: str) -> list[AgentSpec]:
    """The specialists a message needs, in the order they run."""
    scheduling, coverage = bool(_SCHEDULING.search(message or "")), bool(_COVERAGE.search(message or ""))
    if scheduling and coverage:
        return [INSURANCE, REFERRAL]
    return [REFERRAL] if scheduling else [INSURANCE]


def intake_agent(record: dict, settings: Settings | None = None):
    """The Intake agent: one structured-output call (M1)."""
    return parse_intake(record, settings)


# ---- Clinical Safety Reviewer ---------------------------------------------------------------------
class SafetyVerdict(BaseModel):
    safe: bool
    category: Literal["none", "diagnosis", "medication", "symptom_advice", "test_result", "other_clinical"] = "none"
    reason: str = Field(default="", max_length=300)


Reviewer = Callable[[str, str], SafetyVerdict]  # (patient message, draft reply) -> verdict


def default_reviewer(settings: Settings | None = None) -> Reviewer:
    def review(message: str, reply: str) -> SafetyVerdict:
        messages = [
            {"role": "system", "content": REVIEWER_PROMPT},
            {"role": "user", "content": f"Patient message:\n{message}\n\nDraft reply:\n{reply}"},
        ]
        return complete_structured(messages, SafetyVerdict, settings)

    return review


def review_reply(message: str, reply: str, reviewer: Reviewer | None) -> SafetyVerdict:
    """The verdict on a draft. First the output rules (M8, no model call): clinical wording is a veto
    whatever the reviewer would say. Then the reviewer, if one is configured. A reply that is already the
    deflection script needs no review. Any failure to get a verdict is a veto (fail closed)."""
    leak = check_output(reply)
    if leak.blocked:
        return SafetyVerdict(safe=False, category="other_clinical", reason=leak.reason[:300])
    if reviewer is None or not reply.strip() or DEFLECTION_MARK in " ".join(reply.lower().split()):
        return SafetyVerdict(safe=True, reason="not reviewed")
    try:
        return reviewer(message, reply)
    except Exception as exc:
        return SafetyVerdict(safe=False, category="other_clinical", reason=f"reviewer unavailable ({type(exc).__name__})")


CLINICAL_ISSUE = "clinical_question"
INPUT_TICKET_REASON = "Message stopped by the input guard; Clinical Routing Ticket created (OP-CS-014 9.6)"


def _clinical_ticket(deps: Deps, patient_ref: str | None, contact_id: str | None, reason: str, urgency: str,
                     message: str, channel: str) -> dict:
    """A Clinical Routing Ticket raised by the team itself, completed from the records like the graph's own:
    the "Caller states HCP" mark (OP-CS-014 8.3), the repeat marks and the Priority band on a third contact
    (8.2), and the contact recorded as escalated so later contacts count it."""
    reason = policies.mark_hcp(reason, message)
    marker = None
    if patient_ref and deps.memory is not None:
        marker = deps.memory.repeat_marker(patient_ref, CLINICAL_ISSUE, exclude_thread=contact_id)
    if marker:
        reason = f"{marker}: {reason}"
        if marker == PRIORITY_MARK:
            urgency = policies.highest_band(urgency, "Priority")
    args: dict = {"reason": reason, "urgency": urgency, "clinical": True}
    context = {"contact_id": contact_id} if contact_id else None
    result = deps.tools.call("flag_for_human", {**args, "patient_ref": patient_ref} if patient_ref else args, context=context)
    if not result.get("ok") and patient_ref:
        result = deps.tools.call("flag_for_human", args, context=context)
    if patient_ref and deps.memory is not None and result.get("ok"):
        deps.memory.record_outcome(patient_ref, CLINICAL_ISSUE, result, channel=channel, thread_id=contact_id)
    return result


def withhold(deps: Deps, patient_ref: str | None, contact_id: str | None, urgency: str = "Routine",
             message: str = "", channel: str = "web_form") -> dict:
    """Raise the clinical ticket for a vetoed reply and return the flag_for_human result."""
    return _clinical_ticket(deps, patient_ref, contact_id, REVIEW_TICKET_REASON, urgency, message, channel)


def withhold_input(deps: Deps, patient_ref: str | None, contact_id: str | None, gate: GuardResult,
                   message: str = "", channel: str = "web_form") -> dict:
    """The clinical ticket for a message the input gate blocked."""
    return _clinical_ticket(deps, patient_ref, contact_id, f"{INPUT_TICKET_REASON}: {gate.category}",
                            gate.band or "Routine", message, channel)


# ---- the team --------------------------------------------------------------------------------------
@dataclass
class TeamResult:
    answer: str
    agents: list[str] = field(default_factory=list)       # specialists that ran, in order
    tool_calls: list[dict] = field(default_factory=list)  # {name, arguments, result, agent}
    stopped_early: bool = False
    grounding: list[dict] = field(default_factory=list)   # one per specialist
    patient_ref: str | None = None
    verdict: dict | None = None                           # the reviewer's verdict, once reviewed
    vetoed: bool = False
    guard: dict | None = None                             # the input gate's result (M8)
    searched_nothing: bool = False

    @property
    def flags(self) -> list[dict]:
        return [c["result"] for c in self.tool_calls if c["name"] == "flag_for_human" and c["result"].get("ok")]

    @property
    def route(self) -> str:
        """auto | human_review | refuse, in the golden set's terms."""
        deflected = DEFLECTION_MARK in " ".join(self.answer.lower().split())
        if deflected or self.vetoed or any(f.get("category") == "clinical" for f in self.flags):
            return "refuse"
        searches = [c["result"] for c in self.tool_calls if c["name"] == "search_policies"]
        nothing = bool(searches) and all(s.get("abstained") or not s.get("count") for s in searches)
        if self.flags or self.stopped_early or self.ungrounded or nothing:
            return "human_review"
        return "auto"

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]

    @property
    def ungrounded(self) -> bool:
        return any(g.get("checked") and not g.get("grounded") for g in self.grounding)


def run_specialists(message: str, deps: Deps, patient_ref: str | None = None, channel: str = "web_form",
                    thread_id: str | None = None, withheld: frozenset[str] | set[str] = frozenset(),
                    checkpointer=None) -> TeamResult:
    """Run the specialists the message needs. `withheld` tools are offered to nobody (the workflow
    withholds booking while a case waits for staff)."""
    team = TeamResult(answer="", patient_ref=patient_ref)
    available = deps.tools.available_tools() - set(withheld)
    answers: list[str] = []
    for spec in specialists_for(message):
        text = message
        if answers and checkpointer is not None:
            # On a stored conversation the first answer is already in the thread, as an assistant message.
            # Copying it into this message would store it a second time as a *user* message, where a veto's
            # redaction (assistant messages only) cannot reach it.
            text += f"\n\n(Already answered by the {team.agents[-1]} specialist: see the reply above.)"
        elif answers:  # nothing is stored: the second specialist is told what the patient already heard
            text += f"\n\n(Already answered by the {team.agents[-1]} specialist: {answers[-1]})"
        with deps.tracer.span(f"agent.{spec.name}", kind="agent", agent=spec.name, patient_ref=team.patient_ref) as span:
            r: AgentResult = run_agent(
                text, deps=deps, patient_ref=team.patient_ref, channel=channel, thread_id=thread_id,
                tool_names=set(spec.tools) & available, system_prompt=spec.prompt,
                # Both specialists of a two-topic turn run on the conversation's thread, so the second one
                # keeps the verified patient and the history, and the turn is there for the next one.
                checkpointer=checkpointer,
            )
            span.set(tool_calls=r.tools_used, case_status="stopped_early" if r.stopped_early else "answered")
        team.agents.append(spec.name)
        team.tool_calls += [{**c, "agent": spec.name} for c in r.tool_calls]
        team.stopped_early = team.stopped_early or r.stopped_early
        team.grounding.append(r.grounding)
        team.patient_ref = r.patient_ref or team.patient_ref
        answers.append(r.answer)
        if DEFLECTION_MARK in " ".join(r.answer.lower().split()):
            break  # a clinical question: one deflection is the whole reply (OP-CS-014 6.5)
    team.answer = "\n\n".join(a for a in answers if a) if len(answers) > 1 and not team_deflected(answers) else answers[-1]
    return team


def team_deflected(answers: list[str]) -> bool:
    return any(DEFLECTION_MARK in " ".join(a.lower().split()) for a in answers)


def run_team(message: str, deps: Deps, reviewer: Reviewer | None = None, patient_ref: str | None = None,
             channel: str = "web_form", thread_id: str | None = None, withheld: frozenset[str] | set[str] = frozenset(),
             checkpointer=None, guard: bool = False, classifier: Classifier | None = None,
             guard_time_limit_s: float = 2.0) -> TeamResult:
    """The input gate (M8, when `guard` is set), the specialists, then the output check and the Clinical
    Safety Reviewer. The chat path uses this; the triage workflow runs the same steps as separate nodes."""
    # A reference the records do not hold identifies nobody (as the workflow's intake step treats it).
    patient_ref = str(patient_ref).strip().upper() if patient_ref else None
    patient_ref = patient_ref if patient_ref in deps.tools.patients else None
    known = thread_patient(deps, checkpointer, thread_id)  # verified on an earlier turn of this conversation
    if patient_ref and known and patient_ref != known:
        raise PatientMismatch(thread_id or "default")
    with deps.tracer.trace(thread_id or uuid.uuid4().hex, "team", channel=channel, patient_ref=patient_ref) as root:
        if guard:
            with deps.tracer.span("guard.input", kind="guardrail") as span:
                gate: GuardResult = check_input(message, classifier, guard_time_limit_s)
                span.set(safe=not gate.blocked, category=gate.category)
            if gate.blocked:  # no agent sees the message: the script and a clinical ticket, at once
                flag = withhold_input(deps, patient_ref or known, thread_id, gate, message, channel)
                root.set(vetoed=True, category=gate.category)
                return TeamResult(
                    answer=blocked_reply(gate), patient_ref=patient_ref or known, vetoed=True, guard=gate.as_dict(),
                    tool_calls=[{"name": "flag_for_human", "arguments": {"clinical": True}, "result": flag,
                                 "agent": "input_guard"}],
                )
        team = run_specialists(message, deps, patient_ref, channel, thread_id, withheld, checkpointer)
        with deps.tracer.span(f"agent.{REVIEWER.name}", kind="guardrail", agent=REVIEWER.name) as span:
            verdict = review_reply(message, team.answer, reviewer)
            span.set(safe=verdict.safe, category=verdict.category)
        team.verdict = verdict.model_dump()
        if not verdict.safe:
            flag = withhold(deps, team.patient_ref, thread_id, message=message, channel=channel)
            team.vetoed, team.answer = True, DEFLECTION_SCRIPT
            # The agent stored its reply before the review: replace it, so the next turn never sees it.
            redact_answers(deps, checkpointer, thread_id, DEFLECTION_SCRIPT, count=len(team.agents))
            team.tool_calls.append({"name": "flag_for_human", "arguments": {"clinical": True}, "result": flag,
                                    "agent": REVIEWER.name})
        if guard:
            team.guard = gate.as_dict()
        root.set(agents=team.agents, vetoed=team.vetoed)
    return team
