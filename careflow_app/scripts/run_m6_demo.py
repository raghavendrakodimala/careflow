"""LIVE M6 check: the agent team, with the EHR tools served by the MCP server in a separate process.

Run from careflow_app/:
  python -m scripts.run_m6_demo

About 25 chat calls. Five messages go through the team (specialists chosen in code, then the Clinical
Safety Reviewer): a coverage question, a booking, a message needing both specialists, a clinical
question, and a triage-workflow run. The runtime store is a SQLite file in a temporary folder shared by
this process and the MCP server; data/ is never touched. Policy search is not opened.
Checks are on behaviour (which agents ran, which calls went over MCP, flags, routes), not on wording.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from careflow.agents import default_reviewer, run_team
from careflow.config import get_settings
from careflow.graph import Deps
from careflow.mcp import EHR_TOOLS, McpRoutedTools
from careflow.prompts import DEFLECTION_SCRIPT
from careflow.store import RuntimeStore
from careflow.tools import CareFlowTools
from careflow.workflows import TriageDeps, start_triage

results: list[tuple[str, list[str]]] = []


def check(name: str, problems: list[str]) -> None:
    results.append((name, problems))
    print(f"[{'PASS' if not problems else 'FAIL'}] {name} {problems or ''}")


def show(name: str, r) -> None:
    print(f"\n       {name}: agents={r.agents} tools={r.tools_used} vetoed={r.vetoed}\n        -> {r.answer[:240]}")


def main() -> None:
    settings = get_settings()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        local = CareFlowTools(store=RuntimeStore(Path(tmp) / "runtime.db"))
        tools = McpRoutedTools(local)
        print(f"MCP server started; tools served: {sorted(tools.client.tool_names)}")
        check("mcp: serves the five EHR tools", [] if tools.client.tool_names == set(EHR_TOOLS) else ["tool list differs"])
        deps, reviewer = Deps.default(tools, settings), default_reviewer(settings)
        live = {r["patient_ref"] for r in local.store.rows("referrals") if r["status"] in ("open", "accepted", "scheduled")}
        quiet = [ref for ref, p in local.patients.items()
                 if p["eligibility_status"] == "active" and ref not in live and ref in local.established]

        # 1. Insurance specialist only; its EHR lookup goes over MCP.
        before = len(tools.remote_calls)
        r = run_team("Is my insurance active, and what is my deductible?", deps, reviewer, patient_ref=quiet[0])
        show("coverage", r)
        probs = [] if r.agents == ["insurance"] else [f"agents {r.agents}"]
        if "check_eligibility" not in tools.remote_calls[before:]:
            probs.append("eligibility was not checked through MCP")
        check("insurance: coverage question", probs + (["vetoed"] if r.vetoed else []))

        # 2. Referral Tracker only: slots found and one booked, both over MCP.
        before = len(tools.remote_calls)
        r = run_team("Please book me the earliest in-person dermatology appointment you have.", deps, reviewer,
                     patient_ref=quiet[1])
        show("booking", r)
        probs = [] if r.agents == ["referral_tracker"] else [f"agents {r.agents}"]
        sent = tools.remote_calls[before:]
        if "find_appointment_slots" not in sent:
            probs.append("slots were not searched through MCP")
        if "book_appointment" in sent and quiet[1] not in local.booked.values():
            probs.append("a booking made through MCP is not visible to this process")
        check("referral tracker: slots and booking through MCP", probs)

        # 3. Both specialists, in order.
        r = run_team("What would I owe for a dermatology consultation, and can you find me an appointment slot?",
                     deps, reviewer, patient_ref=quiet[2])
        show("both", r)
        check("team: insurance then referral tracker", [] if r.agents == ["insurance", "referral_tracker"] else [f"agents {r.agents}"])

        # 4. Clinical question: the script, a clinical ticket, nothing clinical in the reply.
        r = run_team("My knee has been swollen for a week. What do you think it is, and should I take ibuprofen?",
                     deps, reviewer, patient_ref=quiet[3], thread_id="demo-clinical")
        show("clinical", r)
        flags = [f for f in local.flags if f["patient_ref"] == quiet[3]]
        probs = [] if DEFLECTION_SCRIPT in r.answer else ["reply is not the deflection script"]
        if not any(f["category"] == "clinical" for f in flags):
            probs.append("no clinical ticket")
        check("clinical: deflected with a ticket", probs)

        # 5. The triage workflow with the team inside it.
        record = {"channel": "web_form", "received_at": "2026-08-01T09:00:00Z", "patient_ref": quiet[4],
                  "raw_text": "Hello, I'd like to book a dermatology appointment for a routine skin check."}
        w = start_triage(record, TriageDeps.default(tools, settings))
        print(f"\n       workflow: route={w.route} status={w.status} steps={w.steps}\n        agents={w.agent.get('agents')} "
              f"verdict={w.verdict}\n        -> {w.response[:240]}")
        probs = [] if "team" in w.steps and "safety_review" in w.steps else [f"steps {w.steps}"]
        if w.route != "auto":
            probs.append(f"route {w.route}: {w.reasons}")
        check("workflow: team and safety review ran", probs)

        print(f"\nCalls sent to the MCP server: {len(tools.remote_calls)} {sorted(set(tools.remote_calls))}")
        tools.close()
        local.store.close()

    failed = [name for name, problems in results if problems]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
