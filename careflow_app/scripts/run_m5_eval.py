"""M5 routing check over the 200 intake records. Offline: no model calls.

Run from careflow_app/:
  python -m scripts.run_m5_eval

Each record goes through the triage workflow twice with the assistant switched off (answer=False):
once with the parse the M1 model produced (cached in out/m1_predictions.jsonl by run_m1_eval) and once
with the toolkit's ground_truth labels as the parse. The toolkit calls those labels "your M5 routing
labels", so the second run is the expected route and the first is what the system would do.

The comparison shows how intake mistakes change the route. It is reported, not gated: the urgency
labels are noisy (see notes/handoff.md, M1). Every record starts from a freshly seeded store, so one
record's flags never change the next record's route.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from careflow.config import DATA_DIR, OUT_DIR
from careflow.domain import policies
from careflow.graph import Deps
from careflow.memory import PatientMemory
from careflow.schemas import PatientRequest
from careflow.tools import CareFlowTools
from careflow.workflows import TriageDeps, start_triage

ROUTES = ("auto", "human_review", "refuse")
PREDICTIONS = OUT_DIR / "m1_predictions.jsonl"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _no_model(messages, specs):
    raise AssertionError("the routing check must not call the model")


def route_all(records: list[dict], parses: dict[str, dict]) -> dict[str, dict]:
    """record_id -> {route, reasons, band} using `parses` as the intake step."""
    tools = CareFlowTools()
    deps = TriageDeps(
        agent=Deps(tools=tools, model=_no_model, memory=PatientMemory(tools.store, tools.clock)),
        parse=lambda record: PatientRequest(**parses[record["record_id"]]),
    )
    out = {}
    for record in records:
        if record["record_id"] not in parses:
            continue
        tools.store.reset()
        r = start_triage(record, deps, answer=False)
        out[record["record_id"]] = {"route": r.route, "reasons": r.reasons, "band": r.band}
    return out


def _table(title: str, rows: dict[str, dict]) -> None:
    mix = Counter(r["route"] for r in rows.values())
    print(f"{title}: {len(rows)} records  " + "  ".join(f"{k}={mix[k]}" for k in ROUTES))


def main() -> None:
    records = _jsonl(DATA_DIR / "intake" / "records.jsonl")
    if not PREDICTIONS.exists():
        raise SystemExit(f"{PREDICTIONS} not found: run `python -m scripts.run_m1_eval` first (live).")
    predicted = {p["record_id"]: p["parsed"] for p in _jsonl(PREDICTIONS) if p.get("parsed")}
    # The labels cover urgency, specialty and clinical advice. Who the caller says they are is not a
    # label, so it is taken from the M1 parse in both runs: the comparison then shows only what the
    # labelled fields change (feedback M5-6e).
    identity = ("patient_name", "insurance_member_number")
    labelled = {
        r["record_id"]: {
            **{k: (predicted.get(r["record_id"]) or {}).get(k) for k in identity},
            **{k: r["ground_truth"].get(k) for k in ("urgency", "specialty", "seeks_clinical_advice")},
        }
        for r in records if r.get("ground_truth")
    }

    print("M5 routing check (workflow with the assistant off; no model calls)\n")
    floors = Counter(policies.trigger_band(r["raw_text"]) for r in records)
    print(f"SOP trigger phrases found in the messages: Emergent={floors['Emergent']} Urgent={floors['Urgent']}\n")
    got, want = route_all(records, predicted), route_all(records, labelled)
    _table("Routes from the M1 parses    ", got)
    _table("Routes from the ground truth ", want)

    both = sorted(set(got) & set(want))
    agree = [i for i in both if got[i]["route"] == want[i]["route"]]
    print(f"\nSame route: {len(agree)}/{len(both)} ({len(agree) / len(both):.0%})")
    print("\nConfusion (rows = expected from labels, columns = from M1 parses)")
    print(f"{'':14}" + "".join(f"{c:>14}" for c in ROUTES))
    for w in ROUTES:
        print(f"{w:14}" + "".join(f"{sum(1 for i in both if want[i]['route'] == w and got[i]['route'] == c):>14}" for c in ROUTES))

    # The safety-relevant number: a request that should be refused (clinical or Emergent) but was not.
    missed = [i for i in both if want[i]["route"] == "refuse" and got[i]["route"] != "refuse"]
    extra = [i for i in both if want[i]["route"] != "refuse" and got[i]["route"] == "refuse"]
    print(f"\nShould be refused but was not: {len(missed)}  {missed[:10]}")
    print(f"Refused but labels say otherwise: {len(extra)}  {extra[:10]}")

    reasons = Counter(r.split(":")[0].split(" (")[0] for i in got for r in got[i]["reasons"] if got[i]["route"] == "human_review")
    print("\nWhy cases went to human review (from the M1 parses):")
    for reason, n in reasons.most_common():
        print(f"  {n:>4}  {reason}")

    out = OUT_DIR / "m5_routes.json"
    out.write_text(json.dumps({"from_m1_parses": got, "from_ground_truth": want}, indent=1), encoding="utf-8")
    print(f"\nPer-record routes written to {out.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
