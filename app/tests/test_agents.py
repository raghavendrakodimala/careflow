"""The agent team. Offline: the model and the reviewer are fakes."""

import pytest

from src.agents import (
    INSURANCE,
    INTAKE,
    REFERRAL,
    REVIEW_TICKET_REASON,
    REVIEWER,
    TEAM,
    SafetyVerdict,
    list_specialists_for,
    review_reply,
    run_specialists,
    run_team,
)
from src.graph import AgentDependencies
from src.memory import PatientMemory
from src.prompts import DEFLECTION_SCRIPT, INSURANCE_PROMPT, REFERRAL_PROMPT, SYSTEM_PROMPT
from src.tool_contracts import TOOLS_BY_NAME
from src.tools import CareFlowTools

from .test_workflow import Harness, _find_patient, _make_call, _make_message

SAFE = SafetyVerdict(safe=True)
VETO = SafetyVerdict(safe=False, category="medication", reason="mentions a dose")


class Team:
    """AgentDependencies with a scripted model that records, per call, the prompt and the tools it was offered."""

    def __init__(self, steps=None, tools=None):
        self.tools = tools or CareFlowTools()
        self.steps = list(steps or [_make_message("Here is your answer.")])
        self.calls: list[dict] = []
        self.dependencies = AgentDependencies(tools=self.tools, model=self._call_model, memory=PatientMemory(self.tools.store, self.tools.clock))

    def _call_model(self, messages, specs):
        self.calls.append({"system": messages[0]["content"], "user": messages[-1]["content"],
                           "tools": {spec["function"]["name"] for spec in specs}})
        return self.steps[min(len(self.calls), len(self.steps)) - 1]


# ---- the four agents -------------------------------------------------------------------------------
def test_the_team_has_four_agents_with_separate_tools():
    assert [agent.name for agent in TEAM] == ["intake", "insurance", "referral_tracker", "clinical_safety_reviewer"]
    assert INTAKE.tools == REVIEWER.tools == frozenset()
    assert "calculate_copay" in INSURANCE.tools and "calculate_copay" not in REFERRAL.tools
    assert {"find_appointment_slots", "book_appointment"} <= REFERRAL.tools
    assert not {"find_appointment_slots", "book_appointment"} & INSURANCE.tools
    assert (INSURANCE.tools | REFERRAL.tools) <= set(TOOLS_BY_NAME)


def test_every_specialist_keeps_the_shared_safety_rules():
    for prompt in (INSURANCE_PROMPT, REFERRAL_PROMPT):
        assert prompt.startswith(SYSTEM_PROMPT) and DEFLECTION_SCRIPT in prompt


@pytest.mark.parametrize("message, expected", [
    ("How long is the standard new-patient consultation for Cardiology?", ["insurance"]),
    ("Is my plan active?", ["insurance"]),
    ("Book me a dermatology appointment next week", ["referral_tracker"]),
    ("Any update on referral MHP-R-00012?", ["referral_tracker"]),
    ("What will I owe for a colonoscopy, and can you book it?", ["insurance", "referral_tracker"]),
    ("", ["insurance"]),
])
def test_specialist_is_chosen_from_the_wording(message, expected):
    assert [agent.name for agent in list_specialists_for(message)] == expected


def test_each_specialist_is_offered_only_its_own_tools():
    team = Team()
    run_specialists("Is my plan active?", team.dependencies)
    assert team.calls[0]["system"] == INSURANCE_PROMPT
    assert team.calls[0]["tools"] == set(INSURANCE.tools) - {"search_policies"}  # no index in this test
    team = Team()
    run_specialists("Book me a dermatology appointment", team.dependencies)
    assert team.calls[0]["system"] == REFERRAL_PROMPT and team.calls[0]["tools"] == set(REFERRAL.tools) - {"search_policies"}


def test_withheld_tools_are_offered_to_nobody():
    team = Team()
    run_specialists("What will I owe, and can you book it?", team.dependencies, withheld={"book_appointment"})
    assert len(team.calls) == 2 and all("book_appointment" not in call["tools"] for call in team.calls)


def test_two_topic_message_runs_insurance_then_the_referral_tracker_with_its_answer():
    team = Team(steps=[_make_message("You would owe $40."), _make_message("I found three slots.")])
    result = run_specialists("What will I owe for a dermatology visit, and can you book it?", team.dependencies)
    assert result.agents == ["insurance", "referral_tracker"]
    assert "You would owe $40." in team.calls[1]["user"]
    assert result.answer == "You would owe $40.\n\nI found three slots."


def test_a_deflection_ends_the_turn_with_the_script_alone():
    steps = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Dose question", urgency="Routine", clinical=True)]),
             _make_message(DEFLECTION_SCRIPT), _make_message("should never be asked")]
    team = Team(steps=steps)
    result = run_specialists("What will my tablets cost and should I book to change the dose?", team.dependencies)
    assert result.agents == ["insurance"] and result.answer == DEFLECTION_SCRIPT and len(team.calls) == 2


def test_tool_calls_record_which_agent_made_them():
    team = Team(steps=[_make_message(calls=[_make_call("c1", "check_eligibility", patient_ref="MHP-P-00001")]), _make_message("Active.")])
    result = run_specialists("Is my plan active?", team.dependencies, patient_ref="MHP-P-00001")
    assert [(call["agent"], call["name"]) for call in result.tool_calls] == [("insurance", "check_eligibility")]


# ---- Clinical Safety Reviewer ----------------------------------------------------------------------
def test_reviewer_passes_a_normal_answer():
    team = Team()
    result = run_team("Is my plan active?", team.dependencies, reviewer=lambda message, reply: SAFE)
    assert not result.vetoed and result.answer == "Here is your answer." and team.tools.flags == []


def test_reviewer_veto_replaces_the_reply_and_raises_a_clinical_ticket():
    team = Team(steps=[_make_message("Those tablets are mainly used for blood pressure.")])  # past the output rules; the reviewer catches it
    ref = _find_patient(team.tools)
    result = run_team("What does my plan cover?", team.dependencies, reviewer=lambda message, reply: VETO, patient_ref=ref, thread_id="t1")
    assert result.vetoed and result.answer == DEFLECTION_SCRIPT and "tablets" not in result.answer
    (flag,) = team.tools.flags
    assert (flag["category"], flag["patient_ref"], flag["reason"]) == ("clinical", ref, REVIEW_TICKET_REASON)
    assert result.tool_calls[-1]["agent"] == "clinical_safety_reviewer" and result.verdict["category"] == "medication"


def test_reviewer_failure_fails_closed():
    def raise_timeout(message, reply):
        raise TimeoutError("model timed out")

    verdict = review_reply("q", "Your co-pay is $40.", raise_timeout)
    assert verdict.safe is False and "unavailable" in verdict.reason


def test_reviewer_sees_the_message_and_the_draft():
    seen = []
    team = Team(steps=[_make_message("Your co-pay is $40.")])
    run_team("What is my co-pay?", team.dependencies, reviewer=lambda message, reply: seen.append((message, reply)) or SAFE)
    assert seen == [("What is my co-pay?", "Your co-pay is $40.")]


def test_the_script_itself_is_not_sent_for_review():
    called = []
    assert review_reply("q", DEFLECTION_SCRIPT, lambda message, reply: called.append(1) or VETO).safe is True
    assert review_reply("q", "anything", None).safe is True and called == []


# ---- the team inside the triage workflow -------------------------------------------------------------
def test_workflow_runs_the_specialist_then_the_reviewer():
    harness = Harness()
    harness.dependencies.review = lambda message, reply: SAFE
    result = harness.start(_find_patient(harness.tools))
    assert result.steps == ["guard", "intake", "hydrate", "eligibility", "routing", "team", "safety_review", "decide", "finalize"]
    assert result.agent["agents"] == ["referral_tracker"] and result.verdict["safe"] is True
    assert (result.route, result.status) == ("auto", "completed")


def test_workflow_veto_becomes_a_refusal_that_waits_for_staff():
    harness = Harness(steps=[_make_message("That sounds like eczema; try a steroid cream.")], parsed={"urgency": "Priority"})
    harness.dependencies.review = lambda message, reply: SafetyVerdict(safe=False, category="diagnosis", reason="names a condition")
    ref = _find_patient(harness.tools)
    result = harness.start(ref)
    assert (result.route, result.status) == ("refuse", "awaiting_human_review")
    assert result.response == DEFLECTION_SCRIPT and "eczema" not in result.response
    assert result.reasons == ["The Clinical Safety Reviewer withheld the reply (OP-CS-014 9.6)"]
    (flag,) = harness.tools.flags
    assert (flag["category"], flag["urgency"]) == ("clinical", "Priority")  # the ticket carries the workflow band
    assert harness.tools.store.get_review(result.review_id)["flag_id"] == flag["flag_id"]


def test_workflow_reviewer_error_fails_closed():
    harness = Harness()

    def raise_provider_error(message, reply):
        raise RuntimeError("provider down")

    harness.dependencies.review = raise_provider_error
    result = harness.start(_find_patient(harness.tools))
    assert result.route == "refuse" and result.response == DEFLECTION_SCRIPT and harness.tools.flags[0]["category"] == "clinical"


def test_workflow_does_not_review_when_the_team_did_not_run():
    called = []
    harness = Harness(parsed={"seeks_clinical_advice": True})
    harness.dependencies.review = lambda message, reply: called.append(1) or VETO
    assert harness.start(_find_patient(harness.tools)).route == "refuse" and called == []  # deflected before the team
    harness = Harness()
    harness.dependencies.review = lambda message, reply: called.append(1) or VETO
    assert harness.start(_find_patient(harness.tools), answer=False).route == "auto" and called == []


def test_referral_tracker_books_through_mcp(tmp_path):
    from src.mcp import McpRoutedTools
    from src.store import RuntimeStore

    local = CareFlowTools(store=RuntimeStore(tmp_path / "runtime.db"))
    routed = McpRoutedTools(local)
    try:
        established = local.established
        ref = next(patient_ref for patient_ref, patient in local.patients.items() if patient["eligibility_status"] == "active" and patient_ref in established)
        slot = local.find_appointment_slots("ENT", patient_ref=ref)["slots"][0]["slot_id"]
        steps = [_make_message(calls=[_make_call("c1", "book_appointment", patient_ref=ref, slot_id=slot)]), _make_message("Booked.")]
        team = Team(steps=steps, tools=routed)
        result = run_specialists("Please book that ENT appointment", team.dependencies, patient_ref=ref)
        assert result.agents == ["referral_tracker"] and result.tool_calls[0]["result"]["ok"] is True
        assert routed.remote_calls == ["book_appointment"] and local.booked[slot] == ref
    finally:
        routed.close()
        local.store.close()
