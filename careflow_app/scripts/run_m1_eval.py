"""Parse intake records with the LLM and score against ground_truth.

Run from careflow_app/:  python -m scripts.run_m1_eval [--limit N]
Predictions are cached in out/m1_predictions.jsonl, so a rerun only does what is missing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from careflow.config import DATA_DIR, REPO_ROOT, get_settings
from careflow.intake import parse_intake
from careflow.llm_client import LLMClientError

OUT = REPO_ROOT / "careflow_app" / "out"
CACHE = OUT / "m1_predictions.jsonl"

# Label vocabulary in the generated data is loose; map variants to one name.
_LABEL_ALIASES = {"insurance_id_stated": "insurance_id", "plan_code_format": "insurance_id"}
_KEY_FIELDS = {"insurance_id", "specialty"}


def load_records() -> list[dict]:
    path = DATA_DIR / "intake" / "records.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    rows = (json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["record_id"]: r for r in rows}


def norm_missing(fields) -> set[str]:
    return {_LABEL_ALIASES.get(f, f) for f in fields}


def score(records: list[dict], cache: dict[str, dict]) -> None:
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # name -> [correct, total]
    by_channel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    mismatches = []
    failed = [r["record_id"] for r in records if r["record_id"] in cache and "error" in cache[r["record_id"]]]

    for rec in records:
        gt, pred = rec.get("ground_truth"), cache.get(rec["record_id"])
        if gt is None or pred is None or "error" in pred:
            continue
        p = pred["parsed"]
        checks = {
            "urgency": p["urgency"] == gt["urgency"],
            "urgent_flag (Urgent/Emergent vs not)": (p["urgency"] in ("Urgent", "Emergent"))
            == (gt["urgency"] in ("Urgent", "Emergent")),
            "specialty": p["specialty"] == gt["specialty"],
            "seeks_clinical_advice": p["seeks_clinical_advice"] == gt["seeks_clinical_advice"],
            "missing_fields (exact)": norm_missing(p["missing_fields"]) == norm_missing(gt["missing_fields"]),
            "missing_fields (insurance/specialty only)": (
                norm_missing(p["missing_fields"]) & _KEY_FIELDS == norm_missing(gt["missing_fields"]) & _KEY_FIELDS
            ),
        }
        stated = rec.get("insurance_id_stated")
        checks["insurance_id vs stated"] = (pred["insurance_id"] or None) == (stated or None)
        for name, ok in checks.items():
            tally[name][0] += ok
            tally[name][1] += 1
        by_channel[rec["channel"]][0] += all(checks[k] for k in ("urgency", "specialty", "seeks_clinical_advice"))
        by_channel[rec["channel"]][1] += 1
        if not all(checks[k] for k in ("urgency", "specialty", "seeks_clinical_advice")):
            mismatches.append((rec["record_id"], gt, p))

    if not tally:
        print("\nNothing scored (no successful predictions); out/m1_mismatches.json left unchanged.")
        return
    print("\nPer-field accuracy (records with ground_truth)")
    for name, (ok, n) in tally.items():
        print(f"  {name:<44}{ok:>4}/{n:<4} {ok / n:6.1%}" if n else f"  {name}: n/a")
    print("\nCore three right at once, by channel")
    for ch, (ok, n) in sorted(by_channel.items()):
        print(f"  {ch:<20}{ok:>4}/{n:<4} {ok / n:6.1%}")
    no_gt = [r["record_id"] for r in records if "ground_truth" not in r]
    print(f"\nNo ground_truth (not scored): {no_gt}")
    print(f"Failed to parse: {len(failed)} {failed[:10]}")
    print(f"Mismatches on core fields: {len(mismatches)} (see out/m1_mismatches.json)")
    (OUT / "m1_mismatches.json").write_text(
        json.dumps(
            [{"record_id": i, "truth": g, "predicted": {k: p[k] for k in g}} for i, g, p in mismatches], indent=1
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N records")
    ap.add_argument("--score-only", action="store_true", help="no LLM calls, score the cache")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()[: args.limit]
    cache = load_cache()
    settings = get_settings()
    if args.score_only and not any("parsed" in cache.get(r["record_id"], {}) for r in records):
        # Never overwrite the last mismatches file with an empty one because the cache is missing.
        raise SystemExit(f"--score-only: no cached predictions in {CACHE}. Run without --score-only first.")
    if not args.score_only:
        print(f"model order: {settings.models}")
        todo = [r for r in records if r["record_id"] not in cache or "error" in cache[r["record_id"]]]
        print(f"{len(records)} records, {len(todo)} to parse")
        with CACHE.open("a", encoding="utf-8") as fh:
            for i, rec in enumerate(todo, 1):
                try:
                    req = parse_intake(rec, settings)
                    row = {
                        "record_id": rec["record_id"],
                        "parsed": req.model_dump(),
                        "insurance_id": req.insurance_id,
                    }
                except LLMClientError as exc:
                    row = {"record_id": rec["record_id"], "error": str(exc)[:300]}
                cache[rec["record_id"]] = row
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  [{i}/{len(todo)}] {rec['record_id']} {'ERR' if 'error' in row else 'ok'}")
    score(records, cache)


if __name__ == "__main__":
    main()
