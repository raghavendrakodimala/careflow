"""LIVE system evaluation: the golden set and six extra refusal cases through the full system.

Run from app/ after building the index (python -m scripts.build_policy_index):
  python -m scripts.run_system_eval                      # all 26 cases
  python -m scripts.run_system_eval --cases EV-901 EX-003
  python -m scripts.run_system_eval --classifier hosted  # the chat model as the input-gate classifier
  python -m scripts.run_system_eval --classifier none    # input gate on rules only

Each question goes through the team entry point the API's /v1/chat uses: the input gate (rules, then
the classifier within its time limit; by default the local one in src/guard_model.py), the specialist
agents with policy search, the output rules, the Clinical Safety Reviewer and the grounding check. About
5 chat calls per answerable case and 1 or 2 per refusal: roughly 110 to 140 calls in all, plus a few
embedding calls. Per-case results are written to out/system_eval.json.

Scored per case, with the same checks as the retrieval evaluation (scripts/run_retrieval_eval.check_case) plus the route:
- no must_not_contain phrase; key figures present; a cited document named; groundedness passed;
- a refusal gives the deflection script and raises a clinical ticket (and the emergency statement where
  the message is an emergency); an unanswerable question states no figure and offers staff;
- route: the system's route (auto, human_review, refuse) against the case's expected_route.

The two sets are reported separately: the toolkit's 20 cases, and the 6 hand-written refusal cases in
eval/refusal_cases.json (the toolkit's data/ folder is not changed).
Known mismatch in the toolkit set: EV-020 expects `refuse` for an unanswerable question while EV-905
expects `human_review` for the same situation; the system sends both to `human_review`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from scripts._common import get_result_tag, load_golden_set
from src.config import OUT_DIR, get_settings
from src.graph import EMERGENCY_MARK

EXTRA_CASES = Path(__file__).resolve().parents[1] / "eval" / "refusal_cases.json"
EMERGENCY_CASES = {"CAREFLOW-EV-901", "CAREFLOW-EX-005"}  # the message itself describes an emergency
# The fields every case has, in both sets (the extra cases follow the golden set's schema).
CASE_KEYS = {"id", "question", "expected", "category", "must_cite", "must_not_contain", "expected_route"}


def load_cases() -> tuple[list[dict], list[dict]]:
    """(the toolkit's golden cases, the hand-written extra refusal cases)."""
    extra = json.loads(EXTRA_CASES.read_text(encoding="utf-8"))
    return load_golden_set(), extra


def merge_grounding(team) -> dict:
    """One grounding verdict for a team answer: checked if any specialist was, grounded only if all were."""
    checked = [verdict for verdict in team.grounding if verdict.get("checked")]
    return {"checked": bool(checked), "grounded": all(verdict.get("grounded") for verdict in checked),
            "unsupported": [claim for verdict in checked for claim in verdict.get("unsupported") or []],
            "uncited": [claim for verdict in checked for claim in verdict.get("uncited") or []]}


def score_case(case: dict, team, tools) -> list[str]:
    """Problems with one answer (empty = pass): the retrieval evaluation's checks, the emergency statement, and the route."""
    from scripts.run_retrieval_eval import (
        check_case,  # imported here: run_retrieval_eval reads the golden set at import
    )

    # check_case expects a single agent's result; present the team's answer in that shape.
    view = SimpleNamespace(answer=team.answer, grounding=merge_grounding(team), tools_used=team.tools_used)
    problems = check_case(case, view, tools)
    # The deflection script itself mentions 911, so the test is the emergency statement's own wording.
    if case["id"] in EMERGENCY_CASES and EMERGENCY_MARK not in " ".join(team.answer.lower().split()) \
            and "no emergency statement" not in problems:
        problems.append("no emergency statement")
    if team.route != case["expected_route"]:
        problems.append(f"route {team.route}, expected {case['expected_route']}")
    return problems


def summarise(rows: list[dict]) -> dict:
    """Totals for one set of result rows: cases, passed, routes as expected, and passed/total per category."""
    by_category: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(not row["problems"])
    return {
        "cases": len(rows), "passed": sum(not row["problems"] for row in rows),
        "route_match": sum(row["route"] == row["expected_route"] for row in rows),
        "by_category": {category: f"{sum(passes)}/{len(passes)}" for category, passes in sorted(by_category.items())},
    }


def main() -> None:
    # Imported here so importing this module (the tests do, for the helpers above) stays light.
    from src.agents import build_default_reviewer, run_team
    from src.graph import AgentDependencies
    from src.guard_model import build_classifier
    from src.rag.retriever import open_policy_search
    from src.tools import CareFlowTools

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", help="case ids (or their endings, e.g. EV-901) to run")
    parser.add_argument("--classifier", choices=("local", "hosted", "none"),
                    help="input-gate classifier (default: the CAREFLOW_GUARD_CLASSIFIER setting)")
    parser.add_argument("--no-classifier", action="store_true", help="same as --classifier none")
    args = parser.parse_args()
    settings = get_settings()
    golden, extra = load_cases()
    wanted = lambda candidate: not args.cases or any(candidate["id"].endswith(ending) for ending in args.cases)  # noqa: E731

    retriever = open_policy_search(settings)
    if retriever.count() == 0:
        raise SystemExit("policy index is empty: run python -m scripts.build_policy_index first")
    reviewer = build_default_reviewer(settings)
    mode = "none" if args.no_classifier else args.classifier or settings.careflow_guard_classifier
    classifier = build_classifier(settings.model_copy(update={"careflow_guard_classifier": mode}))
    if hasattr(classifier, "warm"):
        classifier.warm()  # as the API does at start-up
    print(f"System evaluation: index {retriever.count()} chunks; input-gate classifier {mode} "
          f"(time limit {settings.careflow_guard_timeout_s:g}s); model {settings.careflow_model}\n")

    results: dict[str, list[dict]] = {}
    try:
        for label, cases in (("Toolkit golden set", golden), ("Extra refusal cases", extra)):
            rows = []
            for case in filter(wanted, cases):
                tools = CareFlowTools(policy_index=retriever)
                dependencies = AgentDependencies.build_default(tools, settings)
                try:
                    team = run_team(case["question"], dependencies, reviewer, guard=True, classifier=classifier,
                                    guard_time_limit_s=settings.careflow_guard_timeout_s)
                except Exception as error:
                    print(f"[ERR ] {case['id']}: {type(error).__name__}: {str(error)[:110]}")
                    rows.append({"id": case["id"], "category": case["category"], "route": "error",
                                 "expected_route": case["expected_route"], "problems": [f"error {type(error).__name__}"]})
                    continue
                problems = score_case(case, team, tools)
                gate = team.guard or {}
                rows.append({"id": case["id"], "category": case["category"], "route": team.route,
                             "expected_route": case["expected_route"], "problems": problems, "agents": team.agents,
                             "guard": gate, "vetoed": team.vetoed, "answer": team.answer})
                print(f"{get_result_tag(problems)} {case['id']} ({case['category']}, {case['expected_route']}) "
                      f"route={team.route} agents={team.agents} guard={gate.get('layer')}:{gate.get('category')} "
                      f"{gate.get('latency_ms', 0):.0f}ms{'' if not problems else '  ' + '; '.join(problems)}"
                      f"\n        -> {team.answer[:240]}")
                time.sleep(1)  # be gentle with free-tier per-minute limits
            results[label] = rows
    finally:
        retriever.close()

    print()
    for label, rows in results.items():
        if not rows:
            continue
        totals = summarise(rows)
        print(f"{label}: {totals['passed']}/{totals['cases']} passed; route as expected {totals['route_match']}/{totals['cases']}; "
              f"by category {totals['by_category']}")
    gates = [row["guard"] for rows in results.values() for row in rows if row.get("guard")]
    slow = [gate_result for gate_result in gates if gate_result.get("layer") == "fail_closed"]
    if gates:
        times = sorted(gate_result.get("latency_ms", 0) for gate_result in gates)
        print(f"Input gate: median {times[len(times) // 2]:.0f} ms, max {times[-1]:.0f} ms; "
              f"blocked for no answer in time: {len(slow)}/{len(gates)}")
    out = OUT_DIR / "system_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"Per-case results written to out/{out.name}")


if __name__ == "__main__":
    main()
