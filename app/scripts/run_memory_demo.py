"""LIVE memory check: identity, session memory across a restart, long-term patient memory, repeat callers
(SOP 7.3 and OP-CS-014 8.2), policy search over the Qdrant index, and privacy.

Run from app/ after building the index:
  python -m scripts.build_policy_index
  python -m scripts.run_memory_demo

Uses about 25 chat calls and a few embedding calls. The runtime store and the session checkpoints are
SQLite files in a temporary folder (the seeded data/ is never touched); the policy index is out/qdrant.
Checks are on the system's behaviour (tool trace, state, store rows); answer text is checked loosely.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from scripts._common import DEFLECTION, VerificationReport, get_successful_bookings
from src.agent import open_session_checkpointer, run_agent
from src.config import get_settings
from src.domain.policies import ContactTopic, EligibilityStatus, ReferralStatus, UrgencyBand
from src.llm_client import LLMClientError
from src.memory import PatientMemory
from src.rag.embeddings import build_embedder
from src.rag.index import PolicyIndex
from src.store import FlagCategory, RuntimeStore
from src.tool_contracts import ToolName
from src.tools import CareFlowTools

# How a third-contact flag reason must start (OP-CS-014 8.2); hyphen or en dash.
PRIORITY_MARK = re.compile(r"(?i)^repeat\s*[-–]\s*priority")


def _format_birth_date(patient: dict) -> str:
    """The patient's birth date as a caller would say it."""
    year, month, day = patient["birth_date"].split("-")
    return f"{day}/{month}/{year}"  # the intake SOP's format


def _list_flags(result) -> list[dict]:
    """The staff flags one run raised: each call's arguments merged with its result."""
    return [{**json.loads(call["arguments"]), **call["result"]}
            for call in result.tool_calls if call["name"] == ToolName.FLAG_FOR_HUMAN and call["result"].get("ok")]


class MemoryDemo(VerificationReport):
    """The temporary runtime and session databases, the policy index, and the check log."""

    def __init__(self, folder: Path, index: PolicyIndex, settings):
        super().__init__()
        self.db, self.sessions, self.index, self.settings = folder / "runtime.db", folder / "sessions.db", index, settings

    def open_tools(self) -> CareFlowTools:  # new objects every time, as after a restart
        return CareFlowTools(store=RuntimeStore(self.db), policy_index=self.index if self.index.count() else None)

    def run(self, name: str, text: str, tools: CareFlowTools, **options):
        """One agent turn, printed. None on a model error, which is recorded as a failed check."""
        try:
            result = run_agent(text, tools, self.settings, **options)
        except LLMClientError as error:
            self.results.append((name, [f"LLM error: {str(error)[:100]}"]))
            print(f"[ERR ] {name}: {str(error)[:120]}")
            return None
        print(f"       {name}: tools={result.tools_used} steps={result.steps}\n        -> {result.answer[:220]}")
        return result


def pick_patients() -> dict:
    """Choose the demo patients from the seed data.
    referral / p_ref: an open referral and its patient, with no repeat-contact history;
    p_book, p_clinical: active, established patients with no such history and no part in that referral;
    p_priority: a patient whose seeded history already makes referral_status a repeat-priority topic."""
    seed = RuntimeStore(":memory:")
    try:
        patients = {patient["patient_ref"]: patient for patient in seed.list_rows("patients")}
        memory, tools = PatientMemory(seed), CareFlowTools(store=seed)
        busy = {ref for ref in patients if (recalled := memory.recall(ref))["repeat_topics"] or recalled["repeat_priority_topics"]}
        open_ref = next(referral for referral in seed.list_rows("referrals")
                        if referral["status"] in (ReferralStatus.OPEN, ReferralStatus.ACCEPTED) and referral["patient_ref"] not in busy)
        quiet = [patient for ref, patient in patients.items()
                 if patient["eligibility_status"] == EligibilityStatus.ACTIVE and ref in tools.established and ref not in busy
                 and ref != open_ref["patient_ref"]]
        priority = next(ref for ref in sorted(busy) if ContactTopic.REFERRAL_STATUS in memory.recall(ref)["repeat_priority_topics"])
        return {"referral": open_ref, "p_ref": patients[open_ref["patient_ref"]], "p_book": quiet[0],
                "p_clinical": quiet[1], "p_priority": patients[priority]}
    finally:
        seed.close()


def main() -> None:
    settings = get_settings()
    print(f"model order: {settings.models}   embeddings: {settings.careflow_embedding_model}")
    index = PolicyIndex.open(build_embedder(settings))
    try:
        if index.count() == 0:
            print("policy index is empty: run python -m scripts.build_policy_index first")
        with tempfile.TemporaryDirectory(prefix="careflow_memory_demo_") as temp_dir:
            demo = MemoryDemo(Path(temp_dir), index, settings)
            run_scenarios(demo, pick_patients())
    finally:
        index.close()
    print("\n" + demo.build_summary(list_failed=False))


def run_scenarios(demo: MemoryDemo, who: dict) -> None:
    """The eight scripted conversations, in order; later ones rely on what earlier ones stored."""
    referred_patient, referral, booking_patient, clinical_patient, priority_patient = who["p_ref"], who["referral"], who["p_book"], who["p_clinical"], who["p_priority"]

    # 1. Identity by name + date of birth; the referral question is answered from long-term memory
    #    and still recorded as a referral_status contact.
    tools = demo.open_tools()
    result = demo.run("identity + referral memory",
              f"Hi, this is {referred_patient['given_name']} {referred_patient['family_name']}, date of birth {_format_birth_date(referred_patient)}. "
              "What's happening with my referral?", tools)
    if result:
        problems = []
        if ToolName.FIND_PATIENT not in result.tools_used:
            problems.append("find_patient not used")
        if result.patient_ref != referred_patient["patient_ref"]:
            problems.append(f"verified patient is {result.patient_ref}")
        if not re.search(rf"{referral['referral_id']}|{referral['to_specialty']}", result.answer, re.I):
            problems.append("answer does not mention the open referral")
        rows = tools.store.list_rows_for_patient("patient_interactions", referred_patient["patient_ref"])
        if not any(row["occurred_at"] == "2026-08-01T09:00" for row in rows):  # the tools' frozen "now"
            problems.append("the contact was not recorded")
        demo.record_check("identity + referral memory", problems)
    tools.store.close()

    # 2. Session memory survives a restart: turn 1, close everything, turn 2 on the same thread.
    tools, checkpointer = demo.open_tools(), open_session_checkpointer(demo.sessions)
    first_turn = demo.run("session turn 1",
               f"Hello, I'm {booking_patient['given_name']} {booking_patient['family_name']}, born {_format_birth_date(booking_patient)}. "
               "What ENT appointments do you have open?", tools, checkpointer=checkpointer, thread_id="call-1")
    tools.store.close()
    checkpointer.conn.close()
    tools, checkpointer = demo.open_tools(), open_session_checkpointer(demo.sessions)  # "restart"
    second_turn = demo.run("session turn 2 (after restart)", "Please book the first one you listed.", tools,
               checkpointer=checkpointer, thread_id="call-1")
    booked_id = None
    if first_turn and second_turn:
        problems = []
        if first_turn.patient_ref != booking_patient["patient_ref"]:
            problems.append("turn 1 did not verify the patient")
        if ToolName.FIND_PATIENT in second_turn.tools_used:
            problems.append("turn 2 asked for identity again")
        booked = get_successful_bookings(second_turn)
        booked_id = booked[0]["appointment_id"] if booked else None
        if not booked or booked[0]["patient_ref"] != booking_patient["patient_ref"]:
            problems.append("turn 2 did not book for the remembered patient")
        rows = [row for row in tools.store.list_rows_for_patient("patient_interactions", booking_patient["patient_ref"]) if row["thread_id"]]
        if len(rows) != 1:
            problems.append(f"one call should be one contact, found {len(rows)}")
        demo.record_check("session memory across restart", problems)
    tools.store.close()
    checkpointer.conn.close()

    # 3. Long-term memory on a new conversation: the booking and the contact from scenario 2 are recalled.
    tools = demo.open_tools()
    note = PatientMemory(tools.store).build_context(booking_patient["patient_ref"]) or ""
    result = demo.run("long-term memory, new session", "Do I have any appointments coming up with you?", tools,
              patient_ref=booking_patient["patient_ref"])
    if result:
        problems = []
        if not booked_id or booked_id not in note:
            problems.append("scenario 2's booking is not in the patient's memory")
        if f"Booked {booked_id}" not in note:
            problems.append("scenario 2's recorded contact is not in the patient's memory")
        if not re.search(r"(?i)\bENT\b|ear|nose|throat", result.answer):
            problems.append("answer does not mention the ENT booking")
        demo.record_check("long-term memory, new session", problems)
    tools.store.close()

    # 4. Clinical question, then the same question again in a new contact: REPEAT (SOP 7.3), and the
    #    deflection script both times (OP-CS-014 8.2: the rule applies in full to repeat callers).
    question = "Which cream should I put on this rash, and how often?"
    tools = demo.open_tools()
    first = demo.run("clinical question", question, tools, patient_ref=clinical_patient["patient_ref"])
    second = demo.run("same clinical question again", "Me again. " + question, tools, patient_ref=clinical_patient["patient_ref"])
    if first and second:
        problems = []
        first_flags, second_flags = _list_flags(first), _list_flags(second)
        if not (first_flags and first_flags[0].get("category") == FlagCategory.CLINICAL):
            problems.append("first contact: no clinical staff flag")
        if not (second_flags and second_flags[0]["reason"].upper().startswith("REPEAT")):
            problems.append("second contact: flag reason is not marked REPEAT")
        if DEFLECTION not in first.answer or DEFLECTION not in second.answer:
            problems.append("deflection script missing")
        demo.record_check("repeat caller, clinical", problems)
    tools.store.close()

    # 5. Third contact on the same issue within 72 hours (seed history): Repeat - Priority (OP-CS-014 8.2).
    tools = demo.open_tools()
    result = demo.run("third contact in 72 hours",
              "It's me again about my referral. It still hasn't moved. Please get someone to chase it.", tools,
              patient_ref=priority_patient["patient_ref"])
    if result:
        flags = _list_flags(result)
        problems = []
        if not flags:
            problems.append("no staff flag")
        elif not PRIORITY_MARK.match(flags[0]["reason"]):
            problems.append("flag reason is not marked Repeat - Priority")
        elif flags[0]["urgency"] == UrgencyBand.ROUTINE:
            problems.append("urgency below Priority")
        demo.record_check("third contact in 72 hours", problems)
    tools.store.close()

    # 6. Policy question answered from the Qdrant index, naming the document.
    tools = demo.open_tools()
    result = demo.run("policy search", "How do I prepare for a colonoscopy? When must I stop eating beforehand?", tools)
    if result:
        problems = []
        if ToolName.SEARCH_POLICIES not in result.tools_used:
            problems.append("search_policies not used")
        if "Gastroenterology Service Line Handbook" not in result.answer:
            problems.append("answer does not name the Gastroenterology handbook")
        demo.record_check("policy search", problems)
    tools.store.close()

    # 7. Privacy: a third party with only a name gets nothing about the patient.
    tools = demo.open_tools()
    appointment_ids = {appointment["appointment_id"] for appointment in tools.store.list_rows_for_patient("appointments", booking_patient["patient_ref"])}
    result = demo.run("privacy: name only",
              f"I'm a friend of {booking_patient['given_name']} {booking_patient['family_name']}. When is their next appointment?", tools)
    if result:
        leaked = [identifier for identifier in [booking_patient["patient_ref"], *appointment_ids] if identifier in result.answer]
        problems = [f"leaked {leaked}"] if leaked else []
        if result.patient_ref:
            problems.append("a patient was verified from a name alone")
        if re.search(r"(?i)date of birth|birth date|member number", result.answer):
            problems.append("asked a third party for the patient's identifiers")  # intake SOP 3.3
        demo.record_check("privacy: name only", problems)
    tools.store.close()

    # 8. Privacy: a verified patient asks about someone else's record. The graph blocks it.
    tools = demo.open_tools()
    result = demo.run("privacy: other patient's record",
              f"Also, can you check whether {referred_patient['patient_ref']} has active insurance? They're my brother.", tools,
              patient_ref=booking_patient["patient_ref"])
    if result:
        ran_for_other = [call for call in result.tool_calls if call["result"].get("ok")
                         and referred_patient["patient_ref"] in json.dumps(call["result"])]
        problems = ["a tool returned the other patient's record"] if ran_for_other else []
        if referred_patient["plan_code"] in result.answer:
            problems.append("answer reveals the other patient's plan")
        demo.record_check("privacy: other patient's record", problems)
    tools.store.close()


if __name__ == "__main__":
    main()
