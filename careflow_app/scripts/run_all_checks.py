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

Each step's output is kept in out/checks_<step>.log. A step is stopped after 10 minutes. Note that a
laptop going to sleep stops the steps too: the one unexplained 30-minute "stall" of this runner was
the machine in standby (Windows system log, 2026-11-22), not the code.
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
LOGS = APP / "out"  # one log per step: out/checks_<step>.log


def run(name: str, args: list[str], cwd: Path, needs_predictions: bool = False) -> tuple[str, str, float]:
    if needs_predictions and not PREDICTIONS.exists():
        return name, "SKIPPED (out/m1_predictions.jsonl not found)", 0.0
    started = time.time()
    # Output goes to a file, not a pipe: a step that hangs or is killed leaves what it printed, and the
    # runner cannot be left waiting on a pipe a descendant still holds. With the stack dump below, a stuck
    # step writes every thread's stack to its log every two minutes.
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"checks_{name.lower().replace(' ', '_')}.log"
    env = {**os.environ, "CAREFLOW_TRACING": "off", "PYTHONIOENCODING": "utf-8", "PYTHONFAULTHANDLER": "1",
           # Offline checks open no sockets: litellm otherwise fetches its price map at import, and the
           # model hub may be probed for the cached encoders.
           "LITELLM_LOCAL_MODEL_COST_MAP": "True", "HF_HUB_OFFLINE": "1"}
    module, rest = args[1], args[2:]
    boot = ("import faulthandler, runpy, sys; faulthandler.dump_traceback_later(120, repeat=True); "
            f"sys.argv = {[module, *rest]!r}; runpy.run_module({module!r}, run_name='__main__', alter_sys=True)")
    with open(log, "w", encoding="utf-8") as out:
        try:
            code = subprocess.run([sys.executable, "-c", boot], cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, timeout=STEP_TIMEOUT_S, env=env).returncode
        except subprocess.TimeoutExpired:  # a stuck step is a failure, not a reason to wait for ever
            return name, f"FAIL: no result after {STEP_TIMEOUT_S // 60} minutes (see out/{log.name})", time.time() - started
    proc = subprocess.CompletedProcess(args, code, log.read_text(encoding="utf-8", errors="replace"), "")
    lines = [line for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    if name == "data validator":
        ready = any(line.startswith("READY") for line in lines)
        return name, "PASS" if proc.returncode == 0 and ready else f"FAIL: {tail}", time.time() - started
    return name, f"PASS ({tail.strip()[:70]})" if proc.returncode == 0 else f"FAIL: {tail}", time.time() - started


def main() -> None:
    steps = [
        # pytest captures stderr while a test runs, so the bootstrap's stack dump would not show a hung test
        ("tests", ["-m", "pytest", "tests", "-q", "-o", "faulthandler_timeout=120"], APP, False),
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
