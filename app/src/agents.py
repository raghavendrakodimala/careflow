"""The agent team. Four agents with separate jobs, prompts and tools:

    Intake                    reads the message into a PatientRequest and identifies the caller. It is the
                              structured-output step (intake.parse_intake); in the triage workflow
                              it is the `intake` node.
    Insurance                 coverage, pre-authorisation, costs and policy questions.
    Referral Tracker          referral status, appointment slots and booking (the EHR tools, over MCP when
                              the tools object is an McpRoutedTools).
    Clinical Safety Reviewer  reads every reply before it is sent and vetoes clinical content.

Insurance and Referral Tracker are the agent graph (graph.build_agent_graph), each with its own
prompt and tool subset, so each keeps the memory note, the grounding check and the clinical-ticket rule.
Which specialist handles a message is decided in code from its wording (the model does not choose the
route). A message that needs both runs Insurance first and gives its answer to the Referral Tracker.

The reviewer returns a structured verdict. A veto, or any failure to get a verdict, replaces the reply
with the deflection script and raises a Clinical Routing Ticket: fail closed (OP-CS-014 9.3, 9.6).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .agent import DEFAULT_THREAD, AgentResult, PatientMismatchError, get_thread_patient, redact_answers, run_agent
from .config import Settings, get_settings
from .domain import policies
from .domain.policies import Route, UrgencyBand
from .graph import AgentDependencies
from .guardrails import TIME_LIMIT_S, Classifier, GuardResult, build_blocked_reply, check_input, check_output
from .intake import parse_intake
from .llm_client import complete_structured
from .memory import PRIORITY_MARK
from .prompts import DEFLECTION_SCRIPT, INSURANCE_PROMPT, REFERRAL_PROMPT, REVIEWER_PROMPT, is_deflection
from .schemas import DEFAULT_CHANNEL
from .state import Role
from .store import FlagCategory
from .telemetry import AGENT_SPAN_PREFIX, INPUT_GUARD_SPAN, REVIEWER_SPAN, SpanKind, SpanStatus
from .tool_contracts import CONTACT_ID, ToolName

# Staff-flag reasons for the Clinical Routing Tickets the team raises itself.
REVIEW_TICKET_REASON = "Reply withheld by the Clinical Safety Reviewer; Clinical Routing Ticket created (OP-CS-014 9.6)"
INPUT_TICKET_REASON = "Message stopped by the input guard; Clinical Routing Ticket created (OP-CS-014 9.6)"
CLINICAL_ISSUE = policies.CLINICAL_ISSUE
_REASON_MAX = 300  # characters of a SafetyVerdict reason


@dataclass(frozen=True)
class AgentSpec:
    """One member of the team: its name (also its trace label), its job, and the prompt and tools it runs with."""

    name: str
    job: str
    prompt: str
    tools: frozenset[str]


INTAKE = AgentSpec("intake", "Read the message into a PatientRequest and identify the caller.", "", frozenset())
INSURANCE = AgentSpec(
    "insurance", "Coverage, pre-authorisation, costs and policy questions.", INSURANCE_PROMPT,
    frozenset(tool.value for tool in (
        ToolName.CHECK_ELIGIBILITY, ToolName.CHECK_PREAUTH, ToolName.CALCULATE_COPAY, ToolName.SEARCH_POLICIES,
        ToolName.FIND_PATIENT, ToolName.FLAG_FOR_HUMAN)),
)
REFERRAL = AgentSpec(
    "referral_tracker", "Referral status, appointment slots and booking.", REFERRAL_PROMPT,
    # check_eligibility: the shared rules say "check eligibility before booking"; it gives no cost figure.
    frozenset(tool.value for tool in (
        ToolName.CHECK_ELIGIBILITY, ToolName.CHECK_PREAUTH, ToolName.FIND_APPOINTMENT_SLOTS,
        ToolName.BOOK_APPOINTMENT, ToolName.SEARCH_POLICIES, ToolName.FIND_PATIENT, ToolName.FLAG_FOR_HUMAN)),
)
REVIEWER = AgentSpec("clinical_safety_reviewer", "Veto any reply with clinical content.", REVIEWER_PROMPT, frozenset())
TEAM = (INTAKE, INSURANCE, REFERRAL, REVIEWER)
INPUT_GUARD = "input_guard"  # the `agent` label of the ticket raised for a message the input gate blocked
SPECIALISTS = {spec.name: spec for spec in (INSURANCE, REFERRAL)}

# Wording that points to each specialist. Insurance is the default: it holds the policy search, so a
# general question about a rule or a timing goes there.
_SCHEDULING = re.compile(
    r"\b(?:book|booking|appointment|appt|slot|reschedul\w*|schedule|referr\w*|see (?:a|an|the|my) \w+|"
    r"earliest|next available|telehealth visit)\b|MHP-R-\d", re.I)
_COVERAGE = re.compile(
    r"\b(?:co-?pay\w*|cost\w*|price|owe|bill\w*|statement|deductible|coinsurance|eligib\w*|cover\w*|insur\w*|"
    r"pre-?auth\w*|plan|self-?pay|how much)\b", re.I)


def list_specialists_for(message: str) -> list[AgentSpec]:
    """The specialists a message needs, in the order they run."""
    scheduling, coverage = bool(_SCHEDULING.search(message or "")), bool(_COVERAGE.search(message or ""))
    if scheduling and coverage:
        return [INSURANCE, REFERRAL]
    return [REFERRAL] if scheduling else [INSURANCE]


def run_intake_agent(record: dict, settings: Settings | None = None):
    """The Intake agent: one structured-output call."""
    return parse_intake(record, settings)


# ---- Clinical Safety Reviewer ---------------------------------------------------------------------
class SafetyCategory(StrEnum):
    """What kind of clinical content the reviewer found (the values of SafetyVerdict.category, which stays
    a Literal so the schema the model is given does not change). A verdict holds the plain string."""

    NONE = "none"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    SYMPTOM_ADVICE = "symptom_advice"
    TEST_RESULT = "test_result"
    OTHER_CLINICAL = "other_clinical"


# No docstring on this class: it would become the `description` of the JSON schema the model is given.
class SafetyVerdict(BaseModel):
    safe: bool
    category: Literal["none", "diagnosis", "medication", "symptom_advice", "test_result", "other_clinical"] = "none"
    reason: str = Field(default="", max_length=_REASON_MAX)


Reviewer = Callable[[str, str], SafetyVerdict]  # (patient message, draft reply) -> verdict
# Seconds one reviewer request may take before it counts as failed (and so as a veto).
REVIEWER_TIMEOUT_S = 15
# Retries per model for the reviewer. One: with none, a single rate-limit error vetoed a correct reply
# (3 golden cases in one live run); the general three retries took 184 s to fail during an outage.
REVIEWER_RETRIES = 1
# How a verdict that could not be obtained starts its reason, so it can be told apart from a clinical veto.
REVIEWER_UNAVAILABLE = "reviewer unavailable"


def build_default_reviewer(settings: Settings | None = None) -> Reviewer:
    """The model-backed reviewer: one structured-output call per reply, with REVIEWER_PROMPT.

    The call has its own limits. A review that fails is a veto (OP-CS-014 9.3), so during a provider outage
    the patient gets the script either way; with the general limits (a 90-second timeout and three retries)
    they waited minutes for it. One retry rides out a passing rate-limit error, and the fallback models are
    kept: a second model can still give a verdict.
    """
    settings = (settings or get_settings()).model_copy(
        update={"request_timeout_s": REVIEWER_TIMEOUT_S, "transport_retries": REVIEWER_RETRIES})

    def review(message: str, reply: str) -> SafetyVerdict:
        messages = [
            {"role": Role.SYSTEM.value, "content": REVIEWER_PROMPT},
            {"role": Role.USER.value, "content": f"Patient message:\n{message}\n\nDraft reply:\n{reply}"},
        ]
        return complete_structured(messages, SafetyVerdict, settings)

    return review


def review_reply(message: str, reply: str, reviewer: Reviewer | None) -> SafetyVerdict:
    """The verdict on a draft. First the output rules (no model call): clinical wording is a veto
    whatever the reviewer would say. Then the reviewer, if one is configured. A reply that is already the
    deflection script needs no review. Any failure to get a verdict is a veto (fail closed)."""
    leak = check_output(reply)
    if leak.blocked:
        return SafetyVerdict(safe=False, category=SafetyCategory.OTHER_CLINICAL.value, reason=leak.reason[:_REASON_MAX])
    if reviewer is None or not reply.strip() or is_deflection(reply):
        return SafetyVerdict(safe=True, reason="not reviewed")
    try:
        return reviewer(message, reply)
    except Exception as error:
        return SafetyVerdict(safe=False, category=SafetyCategory.OTHER_CLINICAL.value,
                             reason=f"{REVIEWER_UNAVAILABLE} ({type(error).__name__})")


def record_verdict(span, verdict: SafetyVerdict) -> None:
    """Record a verdict on the reviewer's span. A review that could not be done is recorded as an error
    with the failure's type, so the traces tell an outage from a clinical veto (both still withhold the reply)."""
    span.set(safe=verdict.safe, category=verdict.category)
    if verdict.reason.startswith(REVIEWER_UNAVAILABLE):
        span.status = SpanStatus.ERROR.value
        span.set(error_type=verdict.reason[len(REVIEWER_UNAVAILABLE):].strip(" ()") or "unknown")


def _raise_clinical_ticket(dependencies: AgentDependencies, patient_ref: str | None, contact_id: str | None, reason: str, urgency: str,
                           message: str, channel: str) -> dict:
    """A Clinical Routing Ticket raised by the team itself, completed from the records like the graph's own:
    the "Caller states HCP" mark (OP-CS-014 8.3), the repeat marks and the Priority band on a third contact
    (8.2), and the contact recorded as escalated so later contacts count it. `contact_id` is the conversation's
    thread, so the flag de-duplicates within it. Returns the flag_for_human result; a flag the tool refuses
    with a patient_ref is retried without it, so the ticket exists either way."""
    reason = policies.mark_hcp(reason, message)
    marker = None
    if patient_ref and dependencies.memory is not None:
        marker = dependencies.memory.find_repeat_marker(patient_ref, CLINICAL_ISSUE, exclude_thread=contact_id)
    if marker:
        reason = f"{marker}: {reason}"
        if marker == PRIORITY_MARK:
            urgency = policies.find_highest_band(urgency, UrgencyBand.PRIORITY.value)
    arguments: dict = {"reason": reason, "urgency": urgency, "clinical": True}
    context = {CONTACT_ID: contact_id} if contact_id else None
    flag_for_human = ToolName.FLAG_FOR_HUMAN.value
    result = dependencies.tools.call(flag_for_human, {**arguments, "patient_ref": patient_ref} if patient_ref else arguments,
                             context=context)
    if not result.get("ok") and patient_ref:
        result = dependencies.tools.call(flag_for_human, arguments, context=context)
    if patient_ref and dependencies.memory is not None and result.get("ok"):
        dependencies.memory.record_outcome(patient_ref, CLINICAL_ISSUE, result, channel=channel, thread_id=contact_id)
    return result


def withhold(dependencies: AgentDependencies, patient_ref: str | None, contact_id: str | None, urgency: str = policies.DEFAULT_BAND,
             message: str = "", channel: str = DEFAULT_CHANNEL) -> dict:
    """Raise the clinical ticket for a vetoed reply and return the flag_for_human result."""
    return _raise_clinical_ticket(dependencies, patient_ref, contact_id, REVIEW_TICKET_REASON, urgency, message, channel)


def withhold_input(dependencies: AgentDependencies, patient_ref: str | None, contact_id: str | None, gate: GuardResult,
                   message: str = "", channel: str = DEFAULT_CHANNEL) -> dict:
    """The clinical ticket for a message the input gate blocked, at the band the gate gave it."""
    return _raise_clinical_ticket(dependencies, patient_ref, contact_id, f"{INPUT_TICKET_REASON}: {gate.category}",
                            gate.band or policies.DEFAULT_BAND, message, channel)


# ---- the team --------------------------------------------------------------------------------------
@dataclass
class TeamResult:
    """One turn of the team: the reply after review, and what each specialist did."""

    answer: str
    agents: list[str] = field(default_factory=list)       # specialists that ran, in order
    tool_calls: list[dict] = field(default_factory=list)  # {name, arguments, result, agent}
    stopped_early: bool = False                           # a specialist hit its step cap
    grounding: list[dict] = field(default_factory=list)   # one per specialist
    patient_ref: str | None = None
    verdict: dict | None = None                           # the reviewer's verdict, once reviewed
    vetoed: bool = False                                  # the input gate or the reviewer withheld the reply
    guard: dict | None = None                             # the input gate's result

    @property
    def flags(self) -> list[dict]:
        """The staff flags raised in this turn (flag_for_human results that succeeded)."""
        return [call["result"] for call in self.tool_calls
                if call["name"] == ToolName.FLAG_FOR_HUMAN and call["result"].get("ok")]

    @property
    def booking_flag_ids(self) -> list[str]:
        """Staff flags a booking raised by itself (for example coverage still to be verified). They are in
        the booking's result, not in a flag_for_human call."""
        return [call["result"]["flag_id"] for call in self.tool_calls
                if call["name"] == ToolName.BOOK_APPOINTMENT and call["result"].get("ok")
                and call["result"].get("flag_id")]

    @property
    def nothing_found(self) -> bool:
        """True when policy search was used and every search abstained or came back empty: the documents do
        not cover the question."""
        searches = [call["result"] for call in self.tool_calls if call["name"] == ToolName.SEARCH_POLICIES]
        return bool(searches) and all(search.get("abstained") or not search.get("count") for search in searches)

    @property
    def route(self) -> str:
        """auto | human_review | refuse, in the golden set's terms: refuse for anything clinical; human_review
        when staff were flagged (by the agent or by a booking), the step cap was hit, the answer was
        ungrounded or every policy search came back empty."""
        clinical = any(flag.get("category") == FlagCategory.CLINICAL for flag in self.flags)
        if is_deflection(self.answer) or self.vetoed or clinical:
            return Route.REFUSE.value
        if (self.flags or self.booking_flag_ids or self.stopped_early or self.ungrounded
                or self.nothing_found):
            return Route.HUMAN_REVIEW.value
        return Route.AUTO.value

    @property
    def tools_used(self) -> list[str]:
        return [call["name"] for call in self.tool_calls]

    @property
    def ungrounded(self) -> bool:
        """A specialist's answer was checked against the policy documents and withheld."""
        return any(check.get("checked") and not check.get("grounded") for check in self.grounding)


def run_specialists(message: str, dependencies: AgentDependencies, patient_ref: str | None = None, channel: str = DEFAULT_CHANNEL,
                    thread_id: str | None = None, withheld: frozenset[str] | set[str] = frozenset(),
                    checkpointer=None) -> TeamResult:
    """Run the specialists the message needs. `withheld` tools are offered to nobody (the workflow
    withholds booking while a case waits for staff). The reply is not reviewed here: the caller runs
    review_reply on it (run_team, or the workflow's safety_review step)."""
    team = TeamResult(answer="", patient_ref=patient_ref)
    available = dependencies.tools.list_available_tools() - set(withheld)
    answers: list[str] = []
    for spec in list_specialists_for(message):
        text = message
        if answers and checkpointer is not None:
            # On a stored conversation the first answer is already in the thread, as an assistant message.
            # Copying it into this message would store it a second time as a *user* message, where a veto's
            # redaction (assistant messages only) cannot reach it.
            text += f"\n\n(Already answered by the {team.agents[-1]} specialist: see the reply above.)"
        elif answers:  # nothing is stored: the second specialist is told what the patient already heard
            text += f"\n\n(Already answered by the {team.agents[-1]} specialist: {answers[-1]})"
        with dependencies.tracer.open_span(f"{AGENT_SPAN_PREFIX}{spec.name}", kind=SpanKind.AGENT, agent=spec.name,
                              patient_ref=team.patient_ref) as span:
            result: AgentResult = run_agent(
                text, dependencies=dependencies, patient_ref=team.patient_ref, channel=channel, thread_id=thread_id,
                tool_names=set(spec.tools) & available, system_prompt=spec.prompt,
                # Both specialists of a two-topic turn run on the conversation's thread, so the second one
                # keeps the verified patient and the history, and the turn is there for the next one.
                checkpointer=checkpointer,
            )
            span.set(tool_calls=result.tools_used, case_status="stopped_early" if result.stopped_early else "answered")
        team.agents.append(spec.name)
        team.tool_calls += [{**call, "agent": spec.name} for call in result.tool_calls]
        team.stopped_early = team.stopped_early or result.stopped_early
        team.grounding.append(result.grounding)
        team.patient_ref = result.patient_ref or team.patient_ref
        answers.append(result.answer)
        if is_deflection(result.answer):
            break  # a clinical question: one deflection is the whole reply (OP-CS-014 6.5)
    # Both answers, unless one of them deflected: then the last one alone is the reply.
    joined = len(answers) > 1 and not has_deflection(answers)
    team.answer = "\n\n".join(answer for answer in answers if answer) if joined else answers[-1]
    return team


def has_deflection(answers: list[str]) -> bool:
    """Whether any specialist's answer is the deflection script."""
    return any(is_deflection(answer) for answer in answers)


def run_team(message: str, dependencies: AgentDependencies, reviewer: Reviewer | None = None, patient_ref: str | None = None,
             channel: str = DEFAULT_CHANNEL, thread_id: str | None = None,
             withheld: frozenset[str] | set[str] = frozenset(), checkpointer=None, guard: bool = False,
             classifier: Classifier | None = None, guard_time_limit_s: float = TIME_LIMIT_S) -> TeamResult:
    """The input gate (when `guard` is set), the specialists, then the output check and the Clinical
    Safety Reviewer. The chat path uses this; the triage workflow runs the same steps as separate nodes.
    Raises PatientMismatchError when `patient_ref` is not the patient the stored conversation is verified for."""
    # A reference the records do not hold identifies nobody (as the workflow's intake step treats it).
    patient_ref = str(patient_ref).strip().upper() if patient_ref else None
    patient_ref = patient_ref if patient_ref in dependencies.tools.patients else None
    known = get_thread_patient(dependencies, checkpointer, thread_id)  # verified on an earlier turn of this conversation
    if patient_ref and known and patient_ref != known:
        raise PatientMismatchError(thread_id or DEFAULT_THREAD)
    with dependencies.tracer.open_trace(thread_id or uuid.uuid4().hex, "team", channel=channel, patient_ref=patient_ref) as root:
        if guard:
            with dependencies.tracer.open_span(INPUT_GUARD_SPAN, kind=SpanKind.GUARDRAIL) as span:
                gate: GuardResult = check_input(message, classifier, guard_time_limit_s)
                span.set(safe=not gate.blocked, category=gate.category)
            if gate.blocked:  # no agent sees the message: the script and a clinical ticket, at once
                flag = withhold_input(dependencies, patient_ref or known, thread_id, gate, message, channel)
                root.set(vetoed=True, category=gate.category)
                return TeamResult(
                    answer=build_blocked_reply(gate), patient_ref=patient_ref or known, vetoed=True, guard=gate.to_dict(),
                    tool_calls=[{"name": ToolName.FLAG_FOR_HUMAN.value, "arguments": {"clinical": True},
                                 "result": flag, "agent": INPUT_GUARD}],
                )
        team = run_specialists(message, dependencies, patient_ref, channel, thread_id, withheld, checkpointer)
        with dependencies.tracer.open_span(REVIEWER_SPAN, kind=SpanKind.GUARDRAIL, agent=REVIEWER.name) as span:
            verdict = review_reply(message, team.answer, reviewer)
            record_verdict(span, verdict)
        team.verdict = verdict.model_dump()
        if not verdict.safe:
            flag = withhold(dependencies, team.patient_ref, thread_id, message=message, channel=channel)
            team.vetoed, team.answer = True, DEFLECTION_SCRIPT
            # The agent stored its reply before the review: replace it, so the next turn never sees it.
            redact_answers(dependencies, checkpointer, thread_id, DEFLECTION_SCRIPT, count=len(team.agents))
            team.tool_calls.append({"name": ToolName.FLAG_FOR_HUMAN.value, "arguments": {"clinical": True},
                                    "result": flag, "agent": REVIEWER.name})
        if guard:
            team.guard = gate.to_dict()
        root.set(agents=team.agents, vetoed=team.vetoed)
    return team
