"""LIVE triage check: the triage workflow end to end with the real intake parser and model.

Run from app/:
  python -m scripts.run_triage_demo

About 15 chat calls. Four messages: one handled automatically, one clinical question, one Urgent
request that is paused, picked up after a "restart" (new store connection, new checkpointer, new graph)
and released by a staff decision, and one request to a sign-off specialty with no referral on file.
The runtime store and the checkpoints are SQLite files in a temporary folder; data/ is never touched.
Policy search is not needed for these messages, so the Qdrant index is not opened.
Checks are on behaviour (route, steps, flags, review rows); reply text is checked only for the scripts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts._common import DEMO_RECEIVED_AT, VerificationReport, find_quiet_patients
from src.agent import close_checkpointer, open_session_checkpointer
from src.config import get_settings
from src.domain.policies import Route
from src.prompts import DEFLECTION_SCRIPT
from src.schemas import DEFAULT_CHANNEL
from src.store import FlagCategory, ReviewStatus, RuntimeStore
from src.tool_contracts import ToolName
from src.tools import CareFlowTools
from src.workflows import NODES, TriageDependencies, get_pending_review, resume_triage, start_triage
from src.workflows.state import CaseStatus, ReviewAction
from src.workflows.triage import TriageNode

# The workflow's steps for a request handled without staff, in order.
AUTO_STEPS = [step for step in NODES if step not in (TriageNode.DEFLECT, TriageNode.REVIEW)]


class TriageDemo(VerificationReport):
    """The temporary runtime and session databases, the objects opened on them, and the check log."""

    def __init__(self, folder: Path, settings):
        super().__init__()
        self.db, self.sessions, self.settings = folder / "runtime.db", folder / "sessions.db", settings
        self.open()

    def open(self) -> None:  # new objects every time, as after a restart
        self.tools = CareFlowTools(store=RuntimeStore(self.db))
        self.checkpointer = open_session_checkpointer(self.sessions)
        self.dependencies = TriageDependencies.build_default(self.tools, self.settings)

    def restart(self) -> None:
        close_checkpointer(self.checkpointer)
        self.tools.store.close()
        self.open()

    def start(self, name: str, text: str, patient_ref: str, thread_id: str):
        """Run one web-form message through the triage workflow and print what it did."""
        record = {"channel": DEFAULT_CHANNEL, "received_at": DEMO_RECEIVED_AT, "raw_text": text,
                  "patient_ref": patient_ref}
        result = start_triage(record, self.dependencies, self.checkpointer, thread_id=thread_id)
        print(f"\n       {name}: route={result.route} status={result.status} band={result.band}\n        steps={result.steps}"
              f"\n        reasons={result.reasons}\n        tools={result.agent.get('tools_used')}\n        -> {result.response[:220]}")
        return result


def pick_patients() -> dict:
    """One quiet patient per scenario, from an in-memory copy of the seed data."""
    seed = CareFlowTools()
    quiet = find_quiet_patients(seed)
    seed.store.close()
    return dict(zip(("auto", "clinical", "urgent", "sign_off"), quiet, strict=False))  # the first four


def main() -> None:
    settings, patients = get_settings(), pick_patients()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        demo = TriageDemo(Path(temp_dir), settings)

        # 1. Handled automatically: a direct-referral specialty, active cover, Routine.
        result = demo.start("auto", "Hello, I'd like to book a dermatology appointment for a routine skin check "
                    "sometime in the next few weeks. In person is fine.", patients["auto"], "demo-auto")
        problems = [] if (result.route, result.status) == (Route.AUTO, CaseStatus.COMPLETED) else [f"route {result.route}, status {result.status}"]
        if result.steps != AUTO_STEPS:
            problems.append(f"steps {result.steps}")
        if demo.tools.store.list_pending_reviews():
            problems.append("a review was requested")
        demo.record_check("auto: completes without staff", problems)

        # 2. Clinical question: the script and the ticket at once, then the pause.
        result = demo.start("clinical", "My blood pressure tablets make me dizzy. Should I cut the dose in half or "
                    "stop taking them?", patients["clinical"], "demo-clinical")
        flags = [flag for flag in demo.tools.flags if flag["contact_id"] == "demo-clinical"]
        problems = [] if (result.route, result.status) == (Route.REFUSE, CaseStatus.AWAITING_HUMAN_REVIEW) else [f"route {result.route}, status {result.status}"]
        if DEFLECTION_SCRIPT not in result.response:
            problems.append("reply is not the deflection script")
        if not (len(flags) == 1 and flags[0]["category"] == FlagCategory.CLINICAL):
            problems.append(f"clinical flags: {flags}")
        if result.agent:
            problems.append("the assistant was run on a clinical question")
        demo.record_check("clinical: script and ticket, then pause", problems)
        done = resume_triage("demo-clinical", {"reviewer_id": "RN-0042", "action": ReviewAction.APPROVE.value,
                                               "note": "Nurse called the patient back."}, demo.dependencies, demo.checkpointer)
        demo.record_check("clinical: nurse decision recorded",
                       [] if done.status == CaseStatus.COMPLETED and demo.tools.store.get_review(result.review_id)["reviewer_id"] == "RN-0042"
                       else [f"status {done.status}"])

        # 3. Urgent: paused, survives a restart, released by a staff decision. Nothing booked meanwhile.
        result = demo.start("urgent", "I had a mole removed last week and the wound has opened up, two of the stitches "
                    "have come out. I need a dermatology appointment as soon as possible.", patients["urgent"], "demo-urgent")
        problems = [] if (result.route, result.status) == (Route.HUMAN_REVIEW, CaseStatus.AWAITING_HUMAN_REVIEW) else [f"route {result.route}, status {result.status}"]
        if result.agent:
            problems.append("the assistant was run on an Urgent case")
        if ToolName.BOOK_APPOINTMENT in (result.agent.get("tools_offered") or []):
            problems.append("booking was offered before the review")
        if demo.tools.booked:
            problems.append(f"booked before the review: {demo.tools.booked}")
        demo.record_check("urgent: paused, nothing booked", problems)

        demo.restart()
        payload = get_pending_review("demo-urgent", demo.dependencies, demo.checkpointer)
        problems = [] if payload and payload["review_id"] == result.review_id else ["the paused case was not found after the restart"]
        if payload:
            done = resume_triage("demo-urgent", {"reviewer_id": "DCL-0007", "action": ReviewAction.APPROVE.value}, demo.dependencies, demo.checkpointer)
            row = demo.tools.store.get_review(result.review_id)
            if done.status != CaseStatus.COMPLETED or done.response != result.response:
                problems.append("state was not preserved across the restart")
            if (row["status"], row["reviewer_id"]) != (ReviewStatus.DECIDED, "DCL-0007"):
                problems.append(f"review row {row}")
            if len([flag for flag in demo.tools.flags if flag["contact_id"] == "demo-urgent"]) != 1:
                problems.append("a second flag was raised on resume")
        demo.record_check("urgent: resumed from the SQLite checkpoint after a restart", problems)

        # 4. Sign-off specialty with no referral on file (referral policy 3.5).
        result = demo.start("sign-off", "Please book me in with a cardiologist for a routine check-up.", patients["sign_off"], "demo-signoff")
        problems = [] if (result.route, result.status) == (Route.HUMAN_REVIEW, CaseStatus.AWAITING_HUMAN_REVIEW) else [f"route {result.route}, status {result.status}"]
        if not any("sign-off" in reason for reason in result.reasons):
            problems.append(f"reasons {result.reasons}")
        if demo.tools.booked:
            problems.append("booked without sign-off")
        demo.record_check("sign-off: held for staff", problems)

        print("\nStaff queue at the end:", [(review["review_id"], review["route"]) for review in demo.tools.store.list_pending_reviews()])
        close_checkpointer(demo.checkpointer)
        demo.tools.store.close()

    print("\n" + demo.build_summary())


if __name__ == "__main__":
    main()
