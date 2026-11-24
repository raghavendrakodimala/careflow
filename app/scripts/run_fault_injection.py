"""Fault injection over the intake records. Offline: no model calls.

Run from app/:
  python -m scripts.run_fault_injection            # first 60 records
  python -m scripts.run_fault_injection --all      # all 200

Each record is routed by the triage workflow (parses from out/intake_predictions.jsonl, assistant off except
in the last scenario) under four conditions:

  healthy      nothing fails
  flaky EHR    the first two EHR calls of every record fail, then it recovers
  EHR down     every EHR call fails
  model down   the model's circuit breaker is open (the assistant is switched on for this one)

What must hold: a flaky EHR changes no route (retries absorb it); with the EHR or the model down no
case is handled automatically and no refusal is lost; nothing is booked on a guess.
"""

from __future__ import annotations

import argparse
from collections import Counter

from scripts._common import load_intake_records, load_parses
from src.config import DEFAULT_BREAKER_COOLDOWN_S, DEFAULT_BREAKER_FAILURES
from src.domain.policies import Route
from src.graph import AgentDependencies
from src.memory import PatientMemory
from src.reliability import BACKENDS, MODEL_BREAKER, CircuitBreaker, ResilientTools, RetryPolicy, make_resilient_model
from src.schemas import PatientRequest
from src.tool_contracts import Backend
from src.tools import CareFlowTools
from src.workflows import TriageDependencies, start_triage

EHR = {name for name, backend in BACKENDS.items() if backend == Backend.EHR}  # the tools the EHR backend serves


class FaultyTools:
    """Tools whose EHR calls raise: the first `failures` per record, or all of them with None."""

    def __init__(self, inner: CareFlowTools, failures: int | None):
        self.inner, self.failures, self.left, self.raised = inner, failures, failures, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def new_record(self) -> None:
        self.left = self.failures

    def call(self, name, arguments=None, context=None):
        if name in EHR and (self.left is None or self.left > 0):
            if self.left is not None:
                self.left -= 1
            self.raised += 1
            raise ConnectionError("EHR did not answer")
        return self.inner.call(name, arguments, context)


def _no_model(messages, specs):
    """Stands in for the chat model: reaching it means a scenario was not as offline as it claims."""
    raise AssertionError("this scenario must not call the model")


def run_scenario(records: list[dict], parses: dict[str, dict], failures: int | bool | None, model_down: bool = False) -> dict:
    """Route every record under one condition and count what happened.

    failures: False = healthy EHR; an int = that many EHR calls fail at the start of each record;
    None = every EHR call fails. model_down: switch the assistant on behind an open circuit breaker.
    Returns the route per record and totals of bookings, retries (sleeps requested) and injected faults."""
    local = CareFlowTools()
    faulty = FaultyTools(local, failures) if failures is not False else None
    sleeps: list[float] = []  # the retry policy "sleeps" into this list, so the run does not wait
    breakers = {backend: CircuitBreaker(backend, failure_threshold=DEFAULT_BREAKER_FAILURES, cooldown_s=DEFAULT_BREAKER_COOLDOWN_S)
                for backend in set(BACKENDS.values())}
    tools = ResilientTools(faulty or local, retry=RetryPolicy(sleep=sleeps.append), breakers=breakers)
    model = _no_model
    if model_down:
        breaker = CircuitBreaker(MODEL_BREAKER, failure_threshold=1)
        breaker.record_failure()
        model = make_resilient_model(_no_model, breaker)
    dependencies = TriageDependencies(agent=AgentDependencies(tools=tools, model=model, memory=PatientMemory(local.store, local.clock)),
                                              parse=lambda record: PatientRequest(**parses[record["record_id"]]))
    routes, booked = {}, 0
    for record in records:
        local.store.reset()
        for breaker in breakers.values():  # each record starts with closed breakers and a fresh fault budget
            breaker.record_success()
        if faulty:
            faulty.new_record()
        result = start_triage(record, dependencies, answer=model_down)
        routes[record["record_id"]] = result.route
        booked += len(local.booked)
    return {"routes": routes, "booked": booked, "retries": len(sleeps), "faults": faulty.raised if faulty else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="all 200 records (default: the first 60)")
    args = parser.parse_args()
    parses = load_parses()
    records = [record for record in load_intake_records() if record["record_id"] in parses][: None if args.all else 60]

    scenarios = {
        "healthy": run_scenario(records, parses, failures=False),
        "flaky EHR": run_scenario(records, parses, failures=2),
        "EHR down": run_scenario(records, parses, failures=None),
        "model down": run_scenario(records, parses, failures=False, model_down=True),
    }
    base = scenarios["healthy"]["routes"]
    print(f"Fault injection over {len(records)} intake records (no model calls)\n")
    print(f"{'scenario':12}{'auto':>6}{'human':>7}{'refuse':>8}{'faults':>8}{'retries':>9}{'booked':>8}  same route as healthy")
    for name, outcome in scenarios.items():
        mix = Counter(outcome["routes"].values())
        same = sum(1 for record_id, route in outcome["routes"].items() if route == base[record_id])
        print(f"{name:12}{mix[Route.AUTO]:>6}{mix[Route.HUMAN_REVIEW]:>7}{mix[Route.REFUSE]:>8}{outcome['faults']:>8}{outcome['retries']:>9}"
              f"{outcome['booked']:>8}  {same}/{len(base)}")

    refusals = {record_id for record_id, route in base.items() if route == Route.REFUSE}
    problems = []
    if scenarios["flaky EHR"]["routes"] != base:
        problems.append("a flaky EHR changed a route")
    for name in ("EHR down", "model down"):
        routes = scenarios[name]["routes"]
        # "No automatic case" is tested here for the model only; for the EHR it is wrongly_auto below.
        if name == "model down" and any(route == Route.AUTO for route in routes.values()):
            problems.append("model down: a case was handled automatically with no assistant")
        if {record_id for record_id, route in routes.items() if route == Route.REFUSE} != refusals:
            problems.append(f"{name}: the set of refusals changed")
    down = scenarios["EHR down"]
    wrongly_auto = [record_id for record_id, route in down["routes"].items() if route == Route.AUTO and base[record_id] != Route.REFUSE]
    if wrongly_auto:
        problems.append(f"EHR down: {len(wrongly_auto)} cases still automatic, e.g. {wrongly_auto[:3]}")
    if any(outcome["booked"] for outcome in scenarios.values()):
        problems.append("something was booked")
    print("\n" + ("All checks hold." if not problems else "PROBLEMS: " + "; ".join(problems)))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
