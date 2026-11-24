"""Intake evaluation: parse the intake records with the model and score the parses against ground_truth.

Run from app/:
  python -m scripts.run_intake_eval [--limit N]    # LIVE: about one chat call per record not yet parsed (200 records)
  python -m scripts.run_intake_eval --score-only   # offline: score what is already cached

Predictions are cached in out/intake_predictions.jsonl, so a rerun only parses what is missing or failed.
That file is also the intake step of the offline routing and fault-injection checks. Core-field
mismatches are written to out/intake_mismatches.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from scripts._common import load_intake_records, read_jsonl
from src.config import OUT_DIR, get_settings
from src.domain.policies import SAME_DAY_BANDS
from src.intake import parse_intake
from src.llm_client import LLMClientError

OUT = OUT_DIR
CACHE = OUT / "intake_predictions.jsonl"  # one row per attempt, appended; the last row for a record wins

# Label vocabulary in the generated data is loose; map variants to one name.
_LABEL_ALIASES = {"insurance_id_stated": "insurance_id", "plan_code_format": "insurance_id"}
# The missing fields that change what the system does next; the looser missing_fields score uses only these.
_KEY_FIELDS = {"insurance_id", "specialty"}
# A record counts as right overall (per channel, and for the mismatches file) when these three match.
_CORE_FIELDS = ("urgency", "specialty", "seeks_clinical_advice")


def load_records() -> list[dict]:
    return load_intake_records()


def load_cache() -> dict[str, dict]:
    """record_id -> its latest cached row: {"parsed": ..., "insurance_id": ...} or {"error": ...}."""
    if not CACHE.exists():
        return {}
    return {row["record_id"]: row for row in read_jsonl(CACHE)}


def normalise_missing(fields) -> set[str]:
    return {_LABEL_ALIASES.get(field, field) for field in fields}


def score_predictions(records: list[dict], cache: dict[str, dict]) -> None:
    """Print per-field and per-channel accuracy and write out/intake_mismatches.json.
    Writes nothing when no record could be scored, so an earlier mismatches file is kept."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # name -> [correct, total]
    by_channel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    mismatches = []
    failed = [record["record_id"] for record in records if record["record_id"] in cache and "error" in cache[record["record_id"]]]

    for record in records:
        truth, prediction = record.get("ground_truth"), cache.get(record["record_id"])
        if truth is None or prediction is None or "error" in prediction:
            continue
        parsed = prediction["parsed"]
        checks = {
            "urgency": parsed["urgency"] == truth["urgency"],
            "urgent_flag (Urgent/Emergent vs not)": (parsed["urgency"] in SAME_DAY_BANDS)
            == (truth["urgency"] in SAME_DAY_BANDS),
            "specialty": parsed["specialty"] == truth["specialty"],
            "seeks_clinical_advice": parsed["seeks_clinical_advice"] == truth["seeks_clinical_advice"],
            "missing_fields (exact)": normalise_missing(parsed["missing_fields"]) == normalise_missing(truth["missing_fields"]),
            "missing_fields (insurance/specialty only)": (
                normalise_missing(parsed["missing_fields"]) & _KEY_FIELDS == normalise_missing(truth["missing_fields"]) & _KEY_FIELDS
            ),
        }
        stated = record.get("insurance_id_stated")
        checks["insurance_id vs stated"] = (prediction["insurance_id"] or None) == (stated or None)
        for name, correct in checks.items():
            tally[name][0] += correct
            tally[name][1] += 1
        core_right = all(checks[field] for field in _CORE_FIELDS)
        by_channel[record["channel"]][0] += core_right
        by_channel[record["channel"]][1] += 1
        if not core_right:
            mismatches.append((record["record_id"], truth, parsed))

    if not tally:
        print("\nNothing scored (no successful predictions); out/intake_mismatches.json left unchanged.")
        return
    print("\nPer-field accuracy (records with ground_truth)")
    for name, (correct, total) in tally.items():
        print(f"  {name:<44}{correct:>4}/{total:<4} {correct / total:6.1%}" if total else f"  {name}: n/a")
    print("\nCore three right at once, by channel")
    for channel, (correct, total) in sorted(by_channel.items()):
        print(f"  {channel:<20}{correct:>4}/{total:<4} {correct / total:6.1%}")
    unlabelled = [record["record_id"] for record in records if "ground_truth" not in record]
    print(f"\nNo ground_truth (not scored): {unlabelled}")
    print(f"Failed to parse: {len(failed)} {failed[:10]}")
    print(f"Mismatches on core fields: {len(mismatches)} (see out/intake_mismatches.json)")
    (OUT / "intake_mismatches.json").write_text(
        json.dumps(
            [{"record_id": record_id, "truth": labels, "predicted": {field: parsed[field] for field in labels}} for record_id, labels, parsed in mismatches], indent=1
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N records")
    parser.add_argument("--score-only", action="store_true", help="no LLM calls, score the cache")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()[: args.limit]
    cache = load_cache()
    settings = get_settings()
    if args.score_only and not any("parsed" in cache.get(record["record_id"], {}) for record in records):
        # Never overwrite the last mismatches file with an empty one because the cache is missing.
        raise SystemExit(f"--score-only: no cached predictions in {CACHE}. Run without --score-only first.")
    if not args.score_only:
        print(f"model order: {settings.models}")
        todo = [record for record in records if record["record_id"] not in cache or "error" in cache[record["record_id"]]]
        print(f"{len(records)} records, {len(todo)} to parse")
        with CACHE.open("a", encoding="utf-8") as cache_file:
            for number, record in enumerate(todo, 1):
                try:
                    request = parse_intake(record, settings)
                    row = {
                        "record_id": record["record_id"],
                        "parsed": request.model_dump(),
                        "insurance_id": request.insurance_id,
                    }
                except LLMClientError as error:
                    row = {"record_id": record["record_id"], "error": str(error)[:300]}
                cache[record["record_id"]] = row
                cache_file.write(json.dumps(row) + "\n")
                cache_file.flush()
                print(f"  [{number}/{len(todo)}] {record['record_id']} {'ERR' if 'error' in row else 'ok'}")
    score_predictions(records, cache)


if __name__ == "__main__":
    main()
