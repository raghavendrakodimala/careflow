"""Routing check over the 200 intake records. Offline: no model calls.

Run from app/:
  python -m scripts.run_routing_eval

Each record goes through the triage workflow twice with the assistant switched off (answer=False):
once with the parse the intake model produced (cached in out/intake_predictions.jsonl by run_intake_eval) and
once with the toolkit's ground_truth labels as the parse. The toolkit provides those labels as routing
labels, so the second run is the expected route and the first is what the system would do.

The comparison shows how intake mistakes change the route. It is reported, not gated: the urgency
labels are noisy (measured in the intake evaluation). Every record starts from a freshly seeded store, so one
record's flags never change the next record's route.
"""

from __future__ import annotations

import json
from collections import Counter

from scripts._common import INTAKE_PREDICTIONS, load_intake_records, load_parses
from src.config import OUT_DIR
from src.domain import policies
from src.graph import AgentDependencies
from src.memory import PatientMemory
from src.schemas import PatientRequest
from src.tools import CareFlowTools
from src.workflows import TriageDependencies, start_triage

ROUTES = tuple(route.value for route in policies.Route)
REFUSE, HUMAN_REVIEW = policies.Route.REFUSE, policies.Route.HUMAN_REVIEW


def _no_model(messages, specs):
    """Stands in for the chat model: with answer=False the workflow must never reach it."""
    raise AssertionError("the routing check must not call the model")


def route_all(records: list[dict], parses: dict[str, dict]) -> dict[str, dict]:
    """record_id -> {route, reasons, band} using `parses` as the intake step."""
    tools = CareFlowTools()
    dependencies = TriageDependencies(
        agent=AgentDependencies(tools=tools, model=_no_model, memory=PatientMemory(tools.store, tools.clock)),
        parse=lambda record: PatientRequest(**parses[record["record_id"]]),
    )
    out = {}
    for record in records:
        if record["record_id"] not in parses:
            continue
        tools.store.reset()
        result = start_triage(record, dependencies, answer=False)
        out[record["record_id"]] = {"route": result.route, "reasons": result.reasons, "band": result.band}
    return out


def _print_route_counts(title: str, rows: dict[str, dict]) -> None:
    """One line: how many records took each route."""
    mix = Counter(row["route"] for row in rows.values())
    print(f"{title}: {len(rows)} records  " + "  ".join(f"{route}={mix[route]}" for route in ROUTES))


def main() -> None:
    records = load_intake_records()
    if not INTAKE_PREDICTIONS.exists():
        raise SystemExit(f"{INTAKE_PREDICTIONS} not found: run `python -m scripts.run_intake_eval` first (live).")
    predicted = load_parses()
    # The labels cover urgency, specialty and clinical advice. Who the caller says they are is not a
    # label, so it is taken from the model's parse in both runs: the comparison then shows only what the
    # labelled fields change.
    identity = ("patient_name", "insurance_member_number")
    labelled = {
        record["record_id"]: {
            **{field: (predicted.get(record["record_id"]) or {}).get(field) for field in identity},
            **{field: record["ground_truth"].get(field) for field in ("urgency", "specialty", "seeks_clinical_advice")},
        }
        for record in records if record.get("ground_truth")
    }

    print("Routing check (workflow with the assistant off; no model calls)\n")
    floors = Counter(policies.find_trigger_band(record["raw_text"]) for record in records)
    print(f"SOP trigger phrases found in the messages: Emergent={floors['Emergent']} Urgent={floors['Urgent']}\n")
    got, want = route_all(records, predicted), route_all(records, labelled)
    _print_route_counts("Routes from the model parses ", got)
    _print_route_counts("Routes from the ground truth ", want)

    both = sorted(set(got) & set(want))
    agree = [record_id for record_id in both if got[record_id]["route"] == want[record_id]["route"]]
    print(f"\nSame route: {len(agree)}/{len(both)} ({len(agree) / len(both):.0%})")
    print("\nConfusion (rows = expected from labels, columns = from model parses)")
    print(f"{'':14}" + "".join(f"{got_route:>14}" for got_route in ROUTES))
    for wanted_route in ROUTES:
        print(f"{wanted_route:14}" + "".join(f"{sum(1 for record_id in both if want[record_id]['route'] == wanted_route and got[record_id]['route'] == got_route):>14}" for got_route in ROUTES))

    # The safety-relevant number: a request that should be refused (clinical or Emergent) but was not.
    missed = [record_id for record_id in both if want[record_id]["route"] == REFUSE and got[record_id]["route"] != REFUSE]
    extra = [record_id for record_id in both if want[record_id]["route"] != REFUSE and got[record_id]["route"] == REFUSE]
    print(f"\nShould be refused but was not: {len(missed)}  {missed[:10]}")
    print(f"Refused but labels say otherwise: {len(extra)}  {extra[:10]}")

    # Group by the reason's leading text: what follows ":" or " (" is per-record detail.
    reasons = Counter(text.split(":")[0].split(" (")[0] for record_id in got for text in got[record_id]["reasons"] if got[record_id]["route"] == HUMAN_REVIEW)
    print("\nWhy cases went to human review (from the model parses):")
    for reason, count in reasons.most_common():
        print(f"  {count:>4}  {reason}")

    out = OUT_DIR / "routing_routes.json"
    out.write_text(json.dumps({"from_intake_parses": got, "from_ground_truth": want}, indent=1), encoding="utf-8")
    print(f"\nPer-record routes written to {out.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
