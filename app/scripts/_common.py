"""Helpers shared by the scripts in this folder: data loading, demo patients and the check log.

Imported by the run_* scripts, never run on its own. Nothing here calls a model or the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.config import DATA_DIR, OUT_DIR
from src.domain.policies import LIVE_REFERRAL_STATUSES, EligibilityStatus
from src.llm_client import LLMClientError
from src.reliability import ModelUnavailableError
from src.tool_contracts import ToolName

INTAKE_RECORDS = DATA_DIR / "intake" / "records.jsonl"
GOLDEN_SET = DATA_DIR / "eval" / "golden_set.json"
# Written by the live run_intake_eval; the offline routing and fault-injection checks use it as the intake step.
INTAKE_PREDICTIONS = OUT_DIR / "intake_predictions.jsonl"

# A phrase only the OP-CS-014 6.3 deflection script contains; its presence in a reply shows the script was given.
DEFLECTION = "4 business hours"
# received_at for the demo intake records: the instant the tools' frozen clock is set to.
DEMO_RECEIVED_AT = "2026-08-01T09:00:00Z"


def read_jsonl(path: Path) -> list[dict]:
    """The rows of a JSON-lines file; blank lines are skipped."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_intake_records() -> list[dict]:
    """The toolkit's intake messages (most carry ground_truth labels)."""
    return read_jsonl(INTAKE_RECORDS)


def load_golden_set() -> list[dict]:
    """The toolkit's golden evaluation cases."""
    return json.loads(GOLDEN_SET.read_text(encoding="utf-8"))


def load_parses() -> dict[str, dict]:
    """record_id -> the parse the intake model produced; records whose parse failed are left out."""
    return {prediction["record_id"]: prediction["parsed"] for prediction in read_jsonl(INTAKE_PREDICTIONS) if prediction.get("parsed")}


def get_successful_bookings(result) -> list[dict]:
    """The results of the book_appointment calls that succeeded in one agent run, in call order."""
    return [call["result"] for call in result.tool_calls if call["name"] == ToolName.BOOK_APPOINTMENT and call["result"].get("ok")]


def find_quiet_patients(tools) -> list[str]:
    """Patient refs a demo can use without side conditions: active cover, established with the clinic and
    no live referral, so nothing but the message itself decides the route."""
    live = {referral["patient_ref"] for referral in tools.store.list_rows("referrals") if referral["status"] in LIVE_REFERRAL_STATUSES}
    return [ref for ref, patient in tools.patients.items()
            if patient["eligibility_status"] == EligibilityStatus.ACTIVE and ref not in live and ref in tools.established]


def get_result_tag(problems) -> str:
    """The tag that starts a result line. Other tooling greps for it, so the format is fixed."""
    return "[PASS]" if not problems else "[FAIL]"


class VerificationReport:
    """Named checks of a demo run: prints one [PASS]/[FAIL] line per check and keeps them for the summary."""

    def __init__(self) -> None:
        self.results: list[tuple[str, list[str]]] = []  # (check name, problems; empty = passed)

    def record_check(self, name: str, problems: list[str]) -> None:
        self.results.append((name, problems))
        print(f"{get_result_tag(problems)} {name} {problems or ''}")

    @contextmanager
    def attempt(self, name: str) -> Iterator[None]:
        """Run one scenario. If the model cannot be reached the scenario is recorded as a failed check and
        the script goes on to the next one."""
        try:
            yield
        except (LLMClientError, ModelUnavailableError) as error:
            self.record_check(name, [f"not completed: the model could not be reached ({type(error).__name__})"])

    def build_summary(self, list_failed: bool = True) -> str:
        """The "N/M checks passed" line, followed by the names of the failed checks unless list_failed is False."""
        failed = [name for name, problems in self.results if problems]
        line = f"{len(self.results) - len(failed)}/{len(self.results)} checks passed"
        return line + (f"; failed: {failed}" if failed and list_failed else "")
