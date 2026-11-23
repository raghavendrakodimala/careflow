"""Offline tests for the M8 evaluation script's own logic (the live run needs the model and the index)."""

from types import SimpleNamespace

from careflow.agents import TeamResult
from careflow.guardrails import check_rules
from careflow.prompts import DEFLECTION_SCRIPT, EMERGENCY_STATEMENT
from careflow.tools import CareFlowTools
from scripts.run_m8_eval import CASE_KEYS, load_cases, merged_grounding, score, summarise


def _flag(tools, clinical=True):
    reason = "clinical question" if clinical else "needs a callback"  # distinct, or the store de-duplicates them
    return {"name": "flag_for_human", "arguments": {}, "result": tools.flag_for_human(reason, clinical=clinical)}


def test_extra_cases_use_the_golden_format_and_are_all_refusals():
    golden, extra = load_cases()
    assert len(golden) == 20 and len(extra) == 6
    for case in extra:
        assert set(case) == CASE_KEYS == set(golden[0])
        assert case["expected_route"] == "refuse" and case["category"] in ("guardrail", "injection")
        assert case["id"].startswith("CAREFLOW-EX-") and case["must_not_contain"]
    assert len({c["id"] for c in golden + extra}) == 26
    # With the extra cases the set leans toward refusals, as the brief asks: 10 of 26, up from 4 of 20.
    assert sum(c["expected_route"] == "refuse" for c in golden + extra) == 10


def test_some_extra_cases_get_past_the_rules_on_purpose():
    # They are there to test the model layers (classifier, prompt, reviewer), not the regular expressions.
    _, extra = load_cases()
    assert 0 < sum(not check_rules(c["question"]).blocked for c in extra) < len(extra)


def test_route_of_a_team_answer():
    tools = CareFlowTools()
    assert TeamResult(answer="Your plan is active.").route == "auto"
    assert TeamResult(answer=DEFLECTION_SCRIPT).route == "refuse"
    assert TeamResult(answer="x", vetoed=True).route == "refuse"
    assert TeamResult(answer="x", tool_calls=[_flag(tools)]).route == "refuse"
    assert TeamResult(answer="Passed to staff.", tool_calls=[_flag(tools, clinical=False)]).route == "human_review"
    assert TeamResult(answer="x", stopped_early=True).route == "human_review"
    nothing = {"name": "search_policies", "arguments": {}, "result": {"ok": True, "count": 0, "abstained": True}}
    assert TeamResult(answer="I do not have that information.", tool_calls=[nothing]).route == "human_review"
    assert TeamResult(answer="x", grounding=[{"checked": True, "grounded": False}]).route == "human_review"


def test_merged_grounding():
    team = SimpleNamespace(grounding=[{"checked": True, "grounded": True}, {}, {"checked": True, "grounded": False,
                                                                                "unsupported": ["$5"]}])
    assert merged_grounding(team) == {"checked": True, "grounded": False, "unsupported": ["$5"], "uncited": []}
    assert merged_grounding(SimpleNamespace(grounding=[{}]))["checked"] is False


def test_score_a_refusal_case():
    _, extra = load_cases()
    case = next(c for c in extra if c["id"] == "CAREFLOW-EX-005")  # emergency wording
    tools = CareFlowTools()
    good = TeamResult(answer=f"{EMERGENCY_STATEMENT}\n{DEFLECTION_SCRIPT}", tool_calls=[_flag(tools)], vetoed=True)
    assert score(case, good, tools) == []
    no_911 = TeamResult(answer=DEFLECTION_SCRIPT, tool_calls=[_flag(tools)], vetoed=True)
    assert score(case, no_911, tools) == ["no emergency statement"]
    leaked = TeamResult(answer="A doctor would say it is a heart attack.")
    problems = score(case, leaked, CareFlowTools())
    assert any("heart attack" in p for p in problems) and "no deflection script" in problems
    assert "no clinical flag" in problems and "route auto, expected refuse" in problems


def test_summarise():
    rows = [{"category": "guardrail", "route": "refuse", "expected_route": "refuse", "problems": []},
            {"category": "guardrail", "route": "auto", "expected_route": "refuse", "problems": ["x"]},
            {"category": "factual", "route": "auto", "expected_route": "auto", "problems": []}]
    assert summarise(rows) == {"cases": 3, "passed": 2, "route_match": 2,
                               "by_category": {"factual": "1/1", "guardrail": "1/2"}}
