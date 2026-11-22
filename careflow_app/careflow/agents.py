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
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .agent import AgentResult, run_agent
from .config import Settings
from .graph import DEFLECTION_MARK, Deps
from .intake import parse_intake
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
    frozenset({"check_preauth", "find_appointment_slots", "book_appointment", "search_policies", "find_patient",
               "flag_for_human"}),
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
    """The reviewer's verdict on a draft. No reviewer configured: passed. A reply that is already the
    deflection script needs no review. Any failure to get a verdict is a veto (fail closed)."""
    if reviewer is None or not reply.strip() or DEFLECTION_MARK in " ".join(reply.lower().split()):
        return SafetyVerdict(safe=True, reason="not reviewed")
    try:
        return reviewer(message, reply)
    except Exception as exc:
        return SafetyVerdict(safe=False, category="other_clinical", reason=f"reviewer unavailable ({type(exc).__name__})")


def withhold(deps: Deps, patient_ref: str | None, contact_id: str | None, urgency: str = "Routine") -> dict:
    """Raise the clinical ticket for a vetoed reply and return the flag_for_human result."""
    args: dict = {"reason": REVIEW_TICKET_REASON, "urgency": urgency, "clinical": True}
    context = {"contact_id": contact_id} if contact_id else None
    result = deps.tools.call("flag_for_human", {**args, "patient_ref": patient_ref} if patient_ref else args, context)
    if not result.get("ok") and patient_ref:
        result = deps.tools.call("flag_for_human", args, context)
    return result


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
        if answers:  # the second specialist sees what the patient has already been told
            text += f"\n\n(Already answered by the {team.agents[-1]} specialist: {answers[-1]})"
        r: AgentResult = run_agent(
            text, deps=deps, patient_ref=team.patient_ref, channel=channel, thread_id=thread_id,
            tool_names=set(spec.tools) & available, system_prompt=spec.prompt,
            checkpointer=checkpointer if len(specialists_for(message)) == 1 else None,
        )
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
             checkpointer=None) -> TeamResult:
    """Specialists, then the Clinical Safety Reviewer. The chat path uses this; the triage workflow runs
    the same two steps as separate nodes."""
    team = run_specialists(message, deps, patient_ref, channel, thread_id, withheld, checkpointer)
    verdict = review_reply(message, team.answer, reviewer)
    team.verdict = verdict.model_dump()
    if not verdict.safe:
        flag = withhold(deps, team.patient_ref, thread_id)
        team.vetoed, team.answer = True, DEFLECTION_SCRIPT
        team.tool_calls.append({"name": "flag_for_human", "arguments": {"clinical": True}, "result": flag,
                                "agent": REVIEWER.name})
    return team
