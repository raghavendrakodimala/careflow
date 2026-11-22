"""LIVE M5 check: the triage workflow end to end with the real intake parser and model.

Run from careflow_app/:
  python -m scripts.run_m5_demo

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

from careflow.agent import close_checkpointer, session_checkpointer
from careflow.config import get_settings
from careflow.prompts import DEFLECTION_SCRIPT
from careflow.store import RuntimeStore
from careflow.tools import CareFlowTools
from careflow.workflows import TriageDeps, pending_review, resume_triage, start_triage

RECEIVED = "2026-08-01T09:00:00Z"


class Demo:
    def __init__(self, tmp: Path, settings):
        self.db, self.sessions, self.settings = tmp / "runtime.db", tmp / "sessions.db", settings
        self.results: list[tuple[str, list[str]]] = []
        self.open()

    def open(self) -> None:  # new objects every time, as after a restart
        self.tools = CareFlowTools(store=RuntimeStore(self.db))
        self.checkpointer = session_checkpointer(self.sessions)
        self.deps = TriageDeps.default(self.tools, self.settings)

    def restart(self) -> None:
        close_checkpointer(self.checkpointer)
        self.tools.store.close()
        self.open()

    def start(self, name: str, text: str, patient_ref: str, thread_id: str):
        record = {"channel": "web_form", "received_at": RECEIVED, "raw_text": text, "patient_ref": patient_ref}
        r = start_triage(record, self.deps, self.checkpointer, thread_id=thread_id)
        print(f"\n       {name}: route={r.route} status={r.status} band={r.band}\n        steps={r.steps}"
              f"\n        reasons={r.reasons}\n        tools={r.agent.get('tools_used')}\n        -> {r.response[:220]}")
        return r

    def check(self, name: str, problems: list[str]) -> None:
        self.results.append((name, problems))
        print(f"[{'PASS' if not problems else 'FAIL'}] {name} {problems or ''}")


def pick_patients() -> dict:
    seed = CareFlowTools()
    live = {r["patient_ref"] for r in seed.store.rows("referrals") if r["status"] in ("open", "accepted", "scheduled")}
    quiet = [ref for ref, p in seed.patients.items()
             if p["eligibility_status"] == "active" and ref not in live and ref in seed.established]
    seed.store.close()
    return dict(zip(("auto", "clinical", "urgent", "sign_off"), quiet))


def main() -> None:
    settings, p = get_settings(), pick_patients()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        d = Demo(Path(tmp), settings)

        # 1. Handled automatically: a direct-referral specialty, active cover, Routine.
        r = d.start("auto", "Hello, I'd like to book a dermatology appointment for a routine skin check "
                    "sometime in the next few weeks. In person is fine.", p["auto"], "demo-auto")
        probs = [] if (r.route, r.status) == ("auto", "completed") else [f"route {r.route}, status {r.status}"]
        if r.steps != ["guard", "intake", "hydrate", "eligibility", "routing", "team", "safety_review", "decide", "finalize"]:
            probs.append(f"steps {r.steps}")
        if d.tools.store.pending_reviews():
            probs.append("a review was requested")
        d.check("auto: completes without staff", probs)

        # 2. Clinical question: the script and the ticket at once, then the pause.
        r = d.start("clinical", "My blood pressure tablets make me dizzy. Should I cut the dose in half or "
                    "stop taking them?", p["clinical"], "demo-clinical")
        flags = [f for f in d.tools.flags if f["contact_id"] == "demo-clinical"]
        probs = [] if (r.route, r.status) == ("refuse", "awaiting_human_review") else [f"route {r.route}, status {r.status}"]
        if DEFLECTION_SCRIPT not in r.response:
            probs.append("reply is not the deflection script")
        if not (len(flags) == 1 and flags[0]["category"] == "clinical"):
            probs.append(f"clinical flags: {flags}")
        if r.agent:
            probs.append("the assistant was run on a clinical question")
        d.check("clinical: script and ticket, then pause", probs)
        done = resume_triage("demo-clinical", {"reviewer_id": "RN-0042", "action": "approve",
                                               "note": "Nurse called the patient back."}, d.deps, d.checkpointer)
        d.check("clinical: nurse decision recorded",
                [] if done.status == "completed" and d.tools.store.review(r.review_id)["reviewer_id"] == "RN-0042"
                else [f"status {done.status}"])

        # 3. Urgent: paused, survives a restart, released by a staff decision. Nothing booked meanwhile.
        r = d.start("urgent", "I had a mole removed last week and the wound has opened up, two of the stitches "
                    "have come out. I need a dermatology appointment as soon as possible.", p["urgent"], "demo-urgent")
        probs = [] if (r.route, r.status) == ("human_review", "awaiting_human_review") else [f"route {r.route}, status {r.status}"]
        if "book_appointment" in (r.agent.get("tools_offered") or []):
            probs.append("booking was offered before the review")
        if d.tools.booked:
            probs.append(f"booked before the review: {d.tools.booked}")
        d.check("urgent: paused, nothing booked", probs)

        d.restart()
        payload = pending_review("demo-urgent", d.deps, d.checkpointer)
        probs = [] if payload and payload["review_id"] == r.review_id else ["the paused case was not found after the restart"]
        if payload:
            done = resume_triage("demo-urgent", {"reviewer_id": "DCL-0007", "action": "approve"}, d.deps, d.checkpointer)
            row = d.tools.store.review(r.review_id)
            if done.status != "completed" or done.response != r.response:
                probs.append("state was not preserved across the restart")
            if (row["status"], row["reviewer_id"]) != ("decided", "DCL-0007"):
                probs.append(f"review row {row}")
            if len([f for f in d.tools.flags if f["contact_id"] == "demo-urgent"]) != 1:
                probs.append("a second flag was raised on resume")
        d.check("urgent: resumed from the SQLite checkpoint after a restart", probs)

        # 4. Sign-off specialty with no referral on file (referral policy 3.5).
        r = d.start("sign-off", "Please book me in with a cardiologist for a routine check-up.", p["sign_off"], "demo-signoff")
        probs = [] if (r.route, r.status) == ("human_review", "awaiting_human_review") else [f"route {r.route}, status {r.status}"]
        if not any("sign-off" in x for x in r.reasons):
            probs.append(f"reasons {r.reasons}")
        if d.tools.booked:
            probs.append("booked without sign-off")
        d.check("sign-off: held for staff", probs)

        print("\nStaff queue at the end:", [(x["review_id"], x["route"]) for x in d.tools.store.pending_reviews()])
        close_checkpointer(d.checkpointer)
        d.tools.store.close()

    failed = [name for name, problems in d.results if problems]
    print(f"\n{len(d.results) - len(failed)}/{len(d.results)} checks passed" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
