"""M6: the agent team. Offline: the model and the reviewer are fakes."""

import pytest

from careflow.agents import (INSURANCE, INTAKE, REFERRAL, REVIEW_TICKET_REASON, REVIEWER, TEAM, SafetyVerdict,
                             review_reply, run_specialists, run_team, specialists_for)
from careflow.graph import Deps
from careflow.memory import PatientMemory
from careflow.prompts import DEFLECTION_SCRIPT, INSURANCE_PROMPT, REFERRAL_PROMPT, SYSTEM_PROMPT
from careflow.tool_contracts import TOOLS_BY_NAME
from careflow.tools import CareFlowTools

from .test_workflow import Harness, _call, _msg, _patient

SAFE = SafetyVerdict(safe=True)
VETO = SafetyVerdict(safe=False, category="medication", reason="mentions a dose")


class Team:
    """Deps with a scripted model that records, per call, the prompt and the tools it was offered."""

    def __init__(self, steps=None, tools=None):
        self.tools = tools or CareFlowTools()
        self.steps = list(steps or [_msg("Here is your answer.")])
        self.calls: list[dict] = []
        self.deps = Deps(tools=self.tools, model=self._model, memory=PatientMemory(self.tools.store, self.tools.clock))

    def _model(self, messages, specs):
        self.calls.append({"system": messages[0]["content"], "user": messages[-1]["content"],
                           "tools": {s["function"]["name"] for s in specs}})
        return self.steps[min(len(self.calls), len(self.steps)) - 1]


# ---- the four agents -------------------------------------------------------------------------------
def test_the_team_has_four_agents_with_separate_tools():
    assert [a.name for a in TEAM] == ["intake", "insurance", "referral_tracker", "clinical_safety_reviewer"]
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
    assert [a.name for a in specialists_for(message)] == expected


def test_each_specialist_is_offered_only_its_own_tools():
    t = Team()
    run_specialists("Is my plan active?", t.deps)
    assert t.calls[0]["system"] == INSURANCE_PROMPT
    assert t.calls[0]["tools"] == set(INSURANCE.tools) - {"search_policies"}  # no index in this test
    t = Team()
    run_specialists("Book me a dermatology appointment", t.deps)
    assert t.calls[0]["system"] == REFERRAL_PROMPT and t.calls[0]["tools"] == set(REFERRAL.tools) - {"search_policies"}


def test_withheld_tools_are_offered_to_nobody():
    t = Team()
    run_specialists("What will I owe, and can you book it?", t.deps, withheld={"book_appointment"})
    assert len(t.calls) == 2 and all("book_appointment" not in c["tools"] for c in t.calls)


def test_two_topic_message_runs_insurance_then_the_referral_tracker_with_its_answer():
    t = Team(steps=[_msg("You would owe $40."), _msg("I found three slots.")])
    r = run_specialists("What will I owe for a dermatology visit, and can you book it?", t.deps)
    assert r.agents == ["insurance", "referral_tracker"]
    assert "You would owe $40." in t.calls[1]["user"]
    assert r.answer == "You would owe $40.\n\nI found three slots."


def test_a_deflection_ends_the_turn_with_the_script_alone():
    steps = [_msg(calls=[_call("c1", "flag_for_human", reason="Dose question", urgency="Routine", clinical=True)]),
             _msg(DEFLECTION_SCRIPT), _msg("should never be asked")]
    t = Team(steps=steps)
    r = run_specialists("What will my tablets cost and should I book to change the dose?", t.deps)
    assert r.agents == ["insurance"] and r.answer == DEFLECTION_SCRIPT and len(t.calls) == 2


def test_tool_calls_record_which_agent_made_them():
    t = Team(steps=[_msg(calls=[_call("c1", "check_eligibility", patient_ref="MHP-P-00001")]), _msg("Active.")])
    r = run_specialists("Is my plan active?", t.deps, patient_ref="MHP-P-00001")
    assert [(c["agent"], c["name"]) for c in r.tool_calls] == [("insurance", "check_eligibility")]


# ---- Clinical Safety Reviewer ----------------------------------------------------------------------
def test_reviewer_passes_a_normal_answer():
    t = Team()
    r = run_team("Is my plan active?", t.deps, reviewer=lambda message, reply: SAFE)
    assert not r.vetoed and r.answer == "Here is your answer." and t.tools.flags == []


def test_reviewer_veto_replaces_the_reply_and_raises_a_clinical_ticket():
    t = Team(steps=[_msg("Those tablets are mainly used for blood pressure.")])  # past the output rules; the reviewer catches it
    ref = _patient(t.tools)
    r = run_team("What does my plan cover?", t.deps, reviewer=lambda message, reply: VETO, patient_ref=ref, thread_id="t1")
    assert r.vetoed and r.answer == DEFLECTION_SCRIPT and "tablets" not in r.answer
    (flag,) = t.tools.flags
    assert (flag["category"], flag["patient_ref"], flag["reason"]) == ("clinical", ref, REVIEW_TICKET_REASON)
    assert r.tool_calls[-1]["agent"] == "clinical_safety_reviewer" and r.verdict["category"] == "medication"


def test_reviewer_failure_fails_closed():
    def broken(message, reply):
        raise TimeoutError("model timed out")

    verdict = review_reply("q", "Your co-pay is $40.", broken)
    assert verdict.safe is False and "unavailable" in verdict.reason


def test_reviewer_sees_the_message_and_the_draft():
    seen = []
    Team_ = Team(steps=[_msg("Your co-pay is $40.")])
    run_team("What is my co-pay?", Team_.deps, reviewer=lambda m, r: seen.append((m, r)) or SAFE)
    assert seen == [("What is my co-pay?", "Your co-pay is $40.")]


def test_the_script_itself_is_not_sent_for_review():
    called = []
    assert review_reply("q", DEFLECTION_SCRIPT, lambda m, r: called.append(1) or VETO).safe is True
    assert review_reply("q", "anything", None).safe is True and called == []


# ---- the team inside the triage workflow -------------------------------------------------------------
def test_workflow_runs_the_specialist_then_the_reviewer():
    h = Harness()
    h.deps.review = lambda message, reply: SAFE
    r = h.start(_patient(h.tools))
    assert r.steps == ["guard", "intake", "hydrate", "eligibility", "routing", "team", "safety_review", "decide", "finalize"]
    assert r.agent["agents"] == ["referral_tracker"] and r.verdict["safe"] is True
    assert (r.route, r.status) == ("auto", "completed")


def test_workflow_veto_becomes_a_refusal_that_waits_for_staff():
    h = Harness(steps=[_msg("That sounds like eczema; try a steroid cream.")], parsed={"urgency": "Priority"})
    h.deps.review = lambda message, reply: SafetyVerdict(safe=False, category="diagnosis", reason="names a condition")
    ref = _patient(h.tools)
    r = h.start(ref)
    assert (r.route, r.status) == ("refuse", "awaiting_human_review")
    assert r.response == DEFLECTION_SCRIPT and "eczema" not in r.response
    assert r.reasons == ["The Clinical Safety Reviewer withheld the reply (OP-CS-014 9.6)"]
    (flag,) = h.tools.flags
    assert (flag["category"], flag["urgency"]) == ("clinical", "Priority")  # the ticket carries the workflow band
    assert h.tools.store.review(r.review_id)["flag_id"] == flag["flag_id"]


def test_workflow_reviewer_error_fails_closed():
    h = Harness()

    def broken(message, reply):
        raise RuntimeError("provider down")

    h.deps.review = broken
    r = h.start(_patient(h.tools))
    assert r.route == "refuse" and r.response == DEFLECTION_SCRIPT and h.tools.flags[0]["category"] == "clinical"


def test_workflow_does_not_review_when_the_team_did_not_run():
    called = []
    h = Harness(parsed={"seeks_clinical_advice": True})
    h.deps.review = lambda message, reply: called.append(1) or VETO
    assert h.start(_patient(h.tools)).route == "refuse" and called == []  # deflected before the team
    h = Harness()
    h.deps.review = lambda message, reply: called.append(1) or VETO
    assert h.start(_patient(h.tools), answer=False).route == "auto" and called == []


def test_referral_tracker_books_through_mcp(tmp_path):
    from careflow.mcp import McpRoutedTools
    from careflow.store import RuntimeStore

    local = CareFlowTools(store=RuntimeStore(tmp_path / "runtime.db"))
    routed = McpRoutedTools(local)
    try:
        est = local.established
        ref = next(r for r, p in local.patients.items() if p["eligibility_status"] == "active" and r in est)
        slot = local.find_appointment_slots("ENT", patient_ref=ref)["slots"][0]["slot_id"]
        steps = [_msg(calls=[_call("c1", "book_appointment", patient_ref=ref, slot_id=slot)]), _msg("Booked.")]
        t = Team(steps=steps, tools=routed)
        r = run_specialists("Please book that ENT appointment", t.deps, patient_ref=ref)
        assert r.agents == ["referral_tracker"] and r.tool_calls[0]["result"]["ok"] is True
        assert routed.remote_calls == ["book_appointment"] and local.booked[slot] == ref
    finally:
        routed.close()
        local.store.close()
