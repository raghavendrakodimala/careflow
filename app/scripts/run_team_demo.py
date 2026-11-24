"""LIVE team check: the agent team, with the EHR tools served by the MCP server in a separate process.

Run from app/:
  python -m scripts.run_team_demo

About 25 chat calls. Five messages go through the team (specialists chosen in code, then the Clinical
Safety Reviewer): a coverage question, a booking, a message needing both specialists, a clinical
question, and a triage-workflow run. The runtime store is a SQLite file in a temporary folder shared by
this process and the MCP server; data/ is never touched. Policy search is not opened.
Checks are on behaviour (which agents ran, which calls went over MCP, flags, routes), not on wording.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts._common import DEMO_RECEIVED_AT, VerificationReport, find_quiet_patients
from src.agents import build_default_reviewer, run_team
from src.config import get_settings
from src.domain.policies import Route
from src.graph import AgentDependencies
from src.mcp import EHR_TOOLS, McpRoutedTools
from src.prompts import DEFLECTION_SCRIPT
from src.schemas import DEFAULT_CHANNEL
from src.store import FlagCategory, RuntimeStore
from src.tool_contracts import ToolName
from src.tools import CareFlowTools
from src.workflows import TriageDependencies, start_triage


def print_team_run(name: str, result) -> None:
    """Print one team run: which agents ran, the tools they used, whether the reviewer vetoed, the reply."""
    print(f"\n       {name}: agents={result.agents} tools={result.tools_used} vetoed={result.vetoed}\n        -> {result.answer[:240]}")


def main() -> None:
    settings = get_settings()
    checks = VerificationReport()
    record_check = checks.record_check
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        # `local` is this process's view of the store; `tools` sends the EHR tool calls to the MCP server,
        # which opens the same database file, and runs the rest locally.
        local = CareFlowTools(store=RuntimeStore(Path(temp_dir) / "runtime.db"))
        tools = McpRoutedTools(local)
        print(f"MCP server started; tools served: {sorted(tools.client.tool_names)}")
        record_check("mcp: serves the five EHR tools", [] if tools.client.tool_names == set(EHR_TOOLS) else ["tool list differs"])
        dependencies, reviewer = AgentDependencies.build_default(tools, settings), build_default_reviewer(settings)
        quiet = find_quiet_patients(local)  # one per message, so no scenario sees another's flags or bookings

        # 1. Insurance specialist only; its EHR lookup goes over MCP.
        with checks.attempt("insurance: coverage question"):
            before = len(tools.remote_calls)
            result = run_team("Is my insurance active, and what is my deductible?", dependencies, reviewer, patient_ref=quiet[0])
            print_team_run("coverage", result)
            problems = [] if result.agents == ["insurance"] else [f"agents {result.agents}"]
            if ToolName.CHECK_ELIGIBILITY not in tools.remote_calls[before:]:
                problems.append("eligibility was not checked through MCP")
            record_check("insurance: coverage question", problems + (["vetoed"] if result.vetoed else []))

        # 2. Referral Tracker only: slots found and one booked, both over MCP.
        with checks.attempt("referral tracker: slots and booking through MCP"):
            before = len(tools.remote_calls)
            result = run_team("Please book me the earliest in-person dermatology appointment you have.", dependencies, reviewer,
                         patient_ref=quiet[1])
            print_team_run("booking", result)
            problems = [] if result.agents == ["referral_tracker"] else [f"agents {result.agents}"]
            sent = tools.remote_calls[before:]
            if ToolName.FIND_APPOINTMENT_SLOTS not in sent:
                problems.append("slots were not searched through MCP")
            if ToolName.BOOK_APPOINTMENT in sent and quiet[1] not in local.booked.values():
                problems.append("a booking made through MCP is not visible to this process")
            record_check("referral tracker: slots and booking through MCP", problems)

        # 3. Both specialists, in order.
        with checks.attempt("team: insurance then referral tracker"):
            result = run_team("What would I owe for a dermatology consultation, and can you find me an appointment slot?",
                         dependencies, reviewer, patient_ref=quiet[2])
            print_team_run("both", result)
            record_check("team: insurance then referral tracker", [] if result.agents == ["insurance", "referral_tracker"] else [f"agents {result.agents}"])

        # 4. Clinical question: the script, a clinical ticket, nothing clinical in the reply.
        with checks.attempt("clinical: deflected with a ticket"):
            result = run_team("My knee has been swollen for a week. What do you think it is, and should I take ibuprofen?",
                         dependencies, reviewer, patient_ref=quiet[3], thread_id="demo-clinical")
            print_team_run("clinical", result)
            flags = [flag for flag in local.flags if flag["patient_ref"] == quiet[3]]
            problems = [] if DEFLECTION_SCRIPT in result.answer else ["reply is not the deflection script"]
            if not any(flag["category"] == FlagCategory.CLINICAL for flag in flags):
                problems.append("no clinical ticket")
            record_check("clinical: deflected with a ticket", problems)

        # 5. The triage workflow with the team inside it.
        record = {"channel": DEFAULT_CHANNEL, "received_at": DEMO_RECEIVED_AT, "patient_ref": quiet[4],
                  "raw_text": "Hello, I'd like to book a dermatology appointment for a routine skin check."}
        triage = start_triage(record, TriageDependencies.build_default(tools, settings))
        print(f"\n       workflow: route={triage.route} status={triage.status} steps={triage.steps}\n        agents={triage.agent.get('agents')} "
              f"verdict={triage.verdict}\n        -> {triage.response[:240]}")
        problems = [] if "team" in triage.steps and "safety_review" in triage.steps else [f"steps {triage.steps}"]
        if triage.route != Route.AUTO:
            problems.append(f"route {triage.route}: {triage.reasons}")
        record_check("workflow: team and safety review ran", problems)

        print(f"\nCalls sent to the MCP server: {len(tools.remote_calls)} {sorted(set(tools.remote_calls))}")
        tools.close()
        local.store.close()

    print("\n" + checks.build_summary())


if __name__ == "__main__":
    main()
