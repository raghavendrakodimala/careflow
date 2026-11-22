"""Every offline check in one command. No model calls, no key needed.

Run from careflow_app/:
  python -m scripts.run_all_checks

  1. the test suite (pytest tests)
  2. the toolkit's data validator (must say READY; data/ must be unchanged)
  3. the M5 routing check over the 200 intake records
  4. the M7 fault-injection run

Steps 3 and 4 read out/m1_predictions.jsonl, the cached output of the live M1 run. If that file is
missing they are skipped and reported as such, never counted as passed.
Exits 0 only if every step that ran passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
TOOLKIT = APP.parent / "capstone-data-toolkit"
PREDICTIONS = APP / "out" / "m1_predictions.jsonl"
STEP_TIMEOUT_S = 600


def run(name: str, args: list[str], cwd: Path, needs_predictions: bool = False) -> tuple[str, str, float]:
    if needs_predictions and not PREDICTIONS.exists():
        return name, "SKIPPED (out/m1_predictions.jsonl not found)", 0.0
    started = time.time()
    try:
        proc = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=STEP_TIMEOUT_S,
                              stdin=subprocess.DEVNULL,
                              env={**os.environ, "CAREFLOW_TRACING": "off", "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:  # a stuck step is a failure, not a reason to wait for ever
        return name, f"FAIL: no result after {STEP_TIMEOUT_S // 60} minutes", time.time() - started
    lines = [line for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    if name == "data validator":
        ready = any(line.startswith("READY") for line in lines)
        return name, "PASS" if proc.returncode == 0 and ready else f"FAIL: {tail}", time.time() - started
    return name, f"PASS ({tail.strip()[:70]})" if proc.returncode == 0 else f"FAIL: {tail}", time.time() - started


def main() -> None:
    steps = [
        ("tests", ["-m", "pytest", "tests", "-q"], APP, False),
        ("data validator", ["-m", "tests.validate_data", "--domain", "careflow"], TOOLKIT, False),
        ("M5 routing check", ["-m", "scripts.run_m5_eval"], APP, True),
        ("M7 fault injection", ["-m", "scripts.run_m7_chaos"], APP, True),
    ]
    results = []
    for step in steps:
        name, outcome, seconds = run(*step)
        results.append(outcome)
        print(f"{name:20} {outcome}  [{seconds:.0f}s]", flush=True)
    failed = sum(o.startswith("FAIL") for o in results)
    skipped = sum(o.startswith("SKIPPED") for o in results)
    print(f"\n{len(results) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
