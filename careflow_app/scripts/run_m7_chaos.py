"""M7 fault injection over the intake records. Offline: no model calls.

Run from careflow_app/:
  python -m scripts.run_m7_chaos            # first 60 records
  python -m scripts.run_m7_chaos --all      # all 200

Each record is routed by the triage workflow (parses from out/m1_predictions.jsonl, assistant off except
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
import json
from collections import Counter

from careflow.config import DATA_DIR, OUT_DIR
from careflow.graph import Deps
from careflow.memory import PatientMemory
from careflow.reliability import BACKENDS, CircuitBreaker, ResilientTools, RetryPolicy, resilient_model
from careflow.schemas import PatientRequest
from careflow.tools import CareFlowTools
from careflow.workflows import TriageDeps, start_triage

EHR = {name for name, backend in BACKENDS.items() if backend == "ehr"}


class Faulty:
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
    raise AssertionError("this scenario must not call the model")


def run(records: list[dict], parses: dict[str, dict], failures: int | None | bool, model_down: bool = False) -> dict:
    local = CareFlowTools()
    faulty = Faulty(local, failures) if failures is not False else None
    sleeps: list[float] = []
    breakers = {b: CircuitBreaker(b, failure_threshold=3, cooldown_s=30) for b in set(BACKENDS.values())}
    tools = ResilientTools(faulty or local, retry=RetryPolicy(sleep=sleeps.append), breakers=breakers)
    model = _no_model
    if model_down:
        breaker = CircuitBreaker("model", failure_threshold=1)
        breaker.record_failure()
        model = resilient_model(_no_model, breaker)
    deps = TriageDeps(agent=Deps(tools=tools, model=model, memory=PatientMemory(local.store, local.clock)),
                      parse=lambda record: PatientRequest(**parses[record["record_id"]]))
    routes, booked = {}, 0
    for record in records:
        local.store.reset()
        for breaker in breakers.values():  # each record starts with closed breakers and a fresh fault budget
            breaker.record_success()
        if faulty:
            faulty.new_record()
        r = start_triage(record, deps, answer=model_down)
        routes[record["record_id"]] = r.route
        booked += len(local.booked)
    return {"routes": routes, "booked": booked, "retries": len(sleeps), "faults": faulty.raised if faulty else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="all 200 records (default: the first 60)")
    args = parser.parse_args()
    lines = (DATA_DIR / "intake" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    predictions = (OUT_DIR / "m1_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    parses = {p["record_id"]: p["parsed"] for p in map(json.loads, predictions) if p.get("parsed")}
    records = [r for r in records if r["record_id"] in parses][: None if args.all else 60]

    scenarios = {
        "healthy": run(records, parses, failures=False),
        "flaky EHR": run(records, parses, failures=2),
        "EHR down": run(records, parses, failures=None),
        "model down": run(records, parses, failures=False, model_down=True),
    }
    base = scenarios["healthy"]["routes"]
    print(f"M7 fault injection over {len(records)} intake records (no model calls)\n")
    print(f"{'scenario':12}{'auto':>6}{'human':>7}{'refuse':>8}{'faults':>8}{'retries':>9}{'booked':>8}  same route as healthy")
    for name, s in scenarios.items():
        mix = Counter(s["routes"].values())
        same = sum(1 for k, v in s["routes"].items() if v == base[k])
        print(f"{name:12}{mix['auto']:>6}{mix['human_review']:>7}{mix['refuse']:>8}{s['faults']:>8}{s['retries']:>9}"
              f"{s['booked']:>8}  {same}/{len(base)}")

    refusals = {k for k, v in base.items() if v == "refuse"}
    problems = []
    if scenarios["flaky EHR"]["routes"] != base:
        problems.append("a flaky EHR changed a route")
    for name in ("EHR down", "model down"):
        routes = scenarios[name]["routes"]
        # A request with no known patient never asks the EHR for anything, so it can still be automatic.
        if name == "model down" and any(v == "auto" for v in routes.values()):
            problems.append("model down: a case was handled automatically with no assistant")
        if {k for k, v in routes.items() if v == "refuse"} != refusals:
            problems.append(f"{name}: the set of refusals changed")
    down = scenarios["EHR down"]
    wrongly_auto = [k for k, v in down["routes"].items() if v == "auto" and base[k] != "refuse"]
    if wrongly_auto:
        problems.append(f"EHR down: {len(wrongly_auto)} cases still automatic, e.g. {wrongly_auto[:3]}")
    if any(s["booked"] for s in scenarios.values()):
        problems.append("something was booked")
    print("\n" + ("All checks hold." if not problems else "PROBLEMS: " + "; ".join(problems)))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
