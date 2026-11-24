"""Long-term patient memory, read from and written to the runtime store.

Recall: for one verified patient_ref, the recent contacts (patient_interactions), upcoming and last
completed appointments, and open referrals with their pre-authorisation status and SLA date, plus the
two repeat-contact rules, both counted per issue (topic):
- triage SOP OPS-INT-004 7.3: a contact on the same issue within 24 hours makes this one a REPEAT;
- OP-CS-014 8.2: the third contact on the same issue within 72 hours is "Repeat – Priority".

Record: a conversation (thread) is one contact. Its row in patient_interactions is added on the first
turn and updated on later turns, in the seed table's format (topic, outcome, one-line summary). The
summary is built from what the tools did and which record the answer referred to, never from the
patient's words, so no symptoms or free text are stored (data-handling standard, minimum necessary).
The current conversation's own row is left out of the repeat counts.

Privacy rules, enforced here and tested:
- every read is one patient's rows (RuntimeStore.list_rows_for_patient); nothing is loaded for anyone else;
- no names, birth dates or member numbers appear in the memory text;
- an unknown patient_ref gives no memory and records nothing.
Session memory (the conversation itself) is separate: the LangGraph checkpointer, keyed by thread_id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from .domain import policies
from .domain.clock import Clock, FrozenClock
from .domain.policies import AppointmentStatus, ContactTopic
from .schemas import DEFAULT_CHANNEL
from .store import FlagCategory, RuntimeStore
from .tool_contracts import ToolName

REPEAT_WINDOW = timedelta(hours=24)  # triage SOP OPS-INT-004 clause 7.3
PRIORITY_WINDOW = timedelta(hours=72)  # OP-CS-014 clause 8.2: third contact on the same issue
OPEN_REFERRAL_STATUSES = policies.LIVE_REFERRAL_STATUSES  # still in progress, so worth recalling


class ContactOutcome(StrEnum):
    """How a contact ended (patient_interactions.outcome), most significant first."""

    ESCALATED_TO_CLINICIAN = "escalated_to_clinician"
    CALLBACK_SCHEDULED = "callback_scheduled"
    ROUTED_TO_BILLING = "routed_to_billing"
    RESOLVED = "resolved"


# The seed table's five topics and one-line summaries, plus two the assistant needs: a policy question
# answered from the documents, and a contact with no specific issue (answered from records, or handed to
# staff for an administrative reason).
TOPIC_SUMMARY = {
    ContactTopic.CLINICAL_QUESTION: "Asked a clinical question; not answered, routed to a clinician.",
    ContactTopic.APPOINTMENT_BOOKING: "Asked to book or move an appointment.",
    ContactTopic.REFERRAL_STATUS: "Asked for the status of a referral.",
    ContactTopic.BILLING_QUESTION: "Queried a co-pay or an amount on a statement.",
    ContactTopic.ELIGIBILITY_CHECK: "Asked whether their plan is active and what it covers.",
    ContactTopic.POLICY_QUESTION: "Asked about a clinic policy.",
    ContactTopic.GENERAL_ENQUIRY: "Contacted the clinic about their records.",
}
TOPIC_RANK = list(TOPIC_SUMMARY)  # earlier = more significant when one contact covers several turns
OUTCOME_RANK = list(ContactOutcome)  # same ordering
_UNRANKED = 99  # sorts a topic or outcome that is in neither list after every known one
NO_ISSUE = ContactTopic.GENERAL_ENQUIRY  # not an "issue" for the repeat rules: same-issue cannot be established
REPEAT_MARK = "REPEAT"
PRIORITY_MARK = "Repeat - Priority"


def _parse_timestamp(value: str) -> datetime:
    """An ISO timestamp from the tables as a naive datetime, comparable with the clock's."""
    return datetime.fromisoformat(value).replace(tzinfo=None)


@dataclass
class PatientMemory:
    """Recall and record for one store. See the module docstring for the rules it keeps."""

    store: RuntimeStore
    clock: Clock = FrozenClock()  # noqa: RUF009 - one shared instance is fine: the clock holds no state
    max_contacts: int = 5  # how many recent contacts recall() lists (the repeat counts use all of them)

    def _resolve_patient_ref(self, patient_ref: str | None) -> str | None:
        """The canonical patient_ref if it is on record, else None."""
        canonical = str(patient_ref or "").strip().upper()
        return canonical if canonical and self.store.find_one("patients", "patient_ref", canonical) is not None else None

    def _get_now(self) -> datetime:
        return self.clock.now().replace(tzinfo=None)

    # ---- recall ------------------------------------------------------------------------------
    def recall(self, patient_ref: str | None, exclude_thread: str | None = None) -> dict | None:
        """Memory for one patient, as a dict, or None for an unknown patient.
        `exclude_thread`: the current conversation, which is not an earlier contact."""
        patient_ref = self._resolve_patient_ref(patient_ref)
        if patient_ref is None:
            return None
        now = self._get_now()
        contacts = sorted(
            (
                contact for contact in self.store.list_rows_for_patient("patient_interactions", patient_ref)
                if _parse_timestamp(contact["occurred_at"]) <= now
                and not (exclude_thread and contact.get("thread_id") == exclude_thread)
            ),
            key=lambda contact: contact["occurred_at"],
            reverse=True,
        )

        def count_topics_within(window: timedelta) -> dict[str, int]:
            """topic -> number of earlier contacts on it inside the window."""
            counts: dict[str, int] = {}
            for contact in contacts:
                if now - _parse_timestamp(contact["occurred_at"]) <= window and contact["topic"] != NO_ISSUE:
                    counts[contact["topic"]] = counts.get(contact["topic"], 0) + 1
            return counts

        appointments = self.store.list_rows_for_patient("appointments", patient_ref)
        upcoming = sorted(
            (appointment for appointment in appointments
             if appointment["status"] == AppointmentStatus.SCHEDULED
             and _parse_timestamp(appointment["scheduled_for"]) >= now),
            key=lambda appointment: appointment["scheduled_for"],
        )
        completed = sorted(
            (appointment for appointment in appointments
             if appointment["status"] == AppointmentStatus.COMPLETED
             and _parse_timestamp(appointment["scheduled_for"]) <= now),
            key=lambda appointment: appointment["scheduled_for"],
        )
        keep = ("occurred_at", "channel", "topic", "outcome", "summary", "referral_id")
        repeat = sorted(count_topics_within(REPEAT_WINDOW))
        # Two earlier contacts on a topic make a new one on it the third.
        priority = sorted(topic for topic, count in count_topics_within(PRIORITY_WINDOW).items() if count >= 2)
        return {
            "patient_ref": patient_ref,
            "as_of": now.isoformat(timespec="minutes"),
            "recent_contacts": [{field: contact[field] for field in keep} for contact in contacts[: self.max_contacts]],
            "total_contacts": len(contacts),
            "repeat_contact": bool(repeat),
            "repeat_topics": repeat,  # same issue again now = REPEAT (SOP 7.3)
            "repeat_priority_topics": priority,  # same issue again now = third within 72h (OP-CS-014 8.2)
            "upcoming_appointments": [
                {field: appointment[field]
                 for field in ("appointment_id", "specialty", "scheduled_for", "status", "payment")}
                for appointment in upcoming
            ],
            "last_completed_visit": (
                {"specialty": completed[-1]["specialty"], "date": completed[-1]["scheduled_for"][:10]} if completed else None
            ),
            "open_referrals": self._list_open_referrals(patient_ref),
        }

    def _list_open_referrals(self, patient_ref: str) -> list[dict]:
        """The patient's referrals still in progress, oldest first, each with its pre-authorisation status and
        SLA date from referral_details (None where there is no details row)."""
        referrals = []
        for referral in sorted(self.store.list_rows_for_patient("referrals", patient_ref), key=lambda row: row["raised_on"]):
            if referral["status"] not in OPEN_REFERRAL_STATUSES:
                continue
            details = self.store.find_one("referral_details", "referral_id", referral["referral_id"]) or {}
            referrals.append({
                "referral_id": referral["referral_id"],
                "to_specialty": referral["to_specialty"],
                "urgency": referral["urgency"],
                "status": referral["status"],
                "raised_on": referral["raised_on"],
                "procedure_code": details.get("procedure_code"),
                "preauth_status": details.get("preauth_status"),
                "sla_due_on": details.get("sla_due_on"),
            })
        return referrals

    def build_context(self, patient_ref: str | None, exclude_thread: str | None = None) -> str | None:
        """The memory as a short note for the model, or None for an unknown patient."""
        memory = self.recall(patient_ref, exclude_thread)
        if memory is None:
            return None
        lines = [f"Patient memory for {memory['patient_ref']} (verified; clinic records as of {memory['as_of']}):"]
        if memory["repeat_topics"]:
            lines.append(
                f"- Contacted within the last 24 hours about: {', '.join(memory['repeat_topics'])}. If this contact is "
                "about the same issue, it is a REPEAT (triage SOP 7.3)."
            )
        if memory["repeat_priority_topics"]:
            lines.append(
                f"- Two or more contacts in the last 72 hours about: {', '.join(memory['repeat_priority_topics'])}. If "
                "this contact is about the same issue, it is the third: Repeat - Priority (OP-CS-014 8.2)."
            )
        if memory["recent_contacts"]:
            lines.append(f"- Earlier contacts ({memory['total_contacts']} on record, newest first):")
            for contact in memory["recent_contacts"]:
                referral_note = f" [{contact['referral_id']}]" if contact["referral_id"] else ""
                lines.append(
                    f"  - {contact['occurred_at']} {contact['channel']}: {contact['topic']} -> {contact['outcome']}. {contact['summary']}{referral_note}")
        else:
            lines.append("- No earlier contacts on record.")
        for appointment in memory["upcoming_appointments"]:
            lines.append(
                f"- Upcoming appointment {appointment['appointment_id']}: {appointment['specialty']} on {appointment['scheduled_for']}.")
        if memory["last_completed_visit"]:
            visit = memory["last_completed_visit"]
            lines.append(f"- Last completed visit: {visit['specialty']} on {visit['date']}.")
        for referral in memory["open_referrals"]:
            lines.append(
                f"- Open referral {referral['referral_id']} to {referral['to_specialty']} ({referral['urgency']}): status {referral['status']}, "
                f"pre-authorisation {referral['preauth_status']}, SLA due {referral['sla_due_on']}."
            )
        return "\n".join(lines)

    def find_repeat_marker(self, patient_ref: str | None, issue: str, exclude_thread: str | None = None) -> str | None:
        """The marker a staff flag on `issue` needs for this patient: "Repeat - Priority" (third contact on the
        same issue within 72h, OP-CS-014 8.2), "REPEAT" (same issue within 24h, SOP 7.3), or None."""
        if issue == NO_ISSUE:
            return None
        memory = self.recall(patient_ref, exclude_thread)
        if memory is None:
            return None
        if issue in memory["repeat_priority_topics"]:
            return PRIORITY_MARK
        if issue in memory["repeat_topics"]:
            return REPEAT_MARK
        return None

    # ---- record ------------------------------------------------------------------------------
    def record_turn(
        self,
        patient_ref: str | None,
        tool_calls: list[dict],
        channel: str = DEFAULT_CHANNEL,
        thread_id: str | None = None,
        answer: str = "",
    ) -> dict | None:
        """Record this turn as (part of) a contact. With a thread_id the conversation's one row is created or
        updated; without one each call is its own contact. Returns the row, or None for an unknown patient."""
        patient_ref = self._resolve_patient_ref(patient_ref)
        if patient_ref is None:
            return None
        own_referrals = {referral["referral_id"] for referral in self.store.list_rows_for_patient("referrals", patient_ref)}
        return self._save(patient_ref, summarise_turn(tool_calls, answer, own_referrals), channel, thread_id)

    def record_outcome(self, patient_ref: str | None, topic: str, flag: dict, channel: str = DEFAULT_CHANNEL,
                       thread_id: str | None = None) -> dict | None:
        """Record that the triage workflow handed this contact to staff with `flag`. The workflow knows
        the outcome only after the assistant's turn was recorded, so this merges into the thread's row.
        `topic` is what the message was about; it replaces "general_enquiry" so the repeat rules still see it."""
        patient_ref = self._resolve_patient_ref(patient_ref)
        if patient_ref is None:
            return None
        clinical = flag.get("category") == FlagCategory.CLINICAL
        call = {"name": ToolName.FLAG_FOR_HUMAN.value, "arguments": {"clinical": clinical},
                "result": {**flag, "ok": True}}
        turn = summarise_turn([call])
        if turn["topic"] == NO_ISSUE and topic in TOPIC_SUMMARY:
            # str(): the row holds the plain string, also when the caller passed a ContactTopic member
            turn = {**turn, "topic": str(topic),
                    "summary": _build_summary(topic, [detail.strip() for detail in _split_details(turn)])}
        return self._save(patient_ref, turn, channel, thread_id)

    def _save(self, patient_ref: str, turn: dict, channel: str, thread_id: str | None) -> dict:
        """Write `turn` (summarise_turn's shape), merged into the thread's existing contact row if there is one."""
        previous = self.store.find_interaction_for_thread(patient_ref, thread_id) if thread_id else None
        merged = merge_contact(previous, turn) if previous else turn
        return self.store.add_interaction(
            patient_ref=patient_ref,
            occurred_at=self._get_now().isoformat(timespec="minutes"),
            channel=channel,
            thread_id=thread_id,
            **merged,
        )


def _parse_arguments(call: dict) -> dict:
    """A tool call's arguments as a dict, whether stored as a dict or as the model's JSON text ({} if unreadable)."""
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}
    return arguments


_REFERRAL_ID = re.compile(r"MHP-R-\d{5}", re.I)  # a referral ID as it appears in an answer's text


def summarise_turn(tool_calls: list[dict], answer: str = "", own_referrals: set[str] | frozenset = frozenset()) -> dict:
    """One turn as {topic, outcome, summary, referral_id}, from its tool calls ({name, arguments, result}).

    Only what the tools did, and which of the patient's own referrals the answer named, is used.
    A clinical question is recognised by a clinical staff flag (flag_for_human with clinical=true), never
    by "only a flag was raised": step-cap and administrative handoffs are not clinical."""
    ok = [call for call in tool_calls if (call.get("result") or {}).get("ok")]
    names = {call["name"] for call in ok}
    flags = [call for call in ok if call["name"] == ToolName.FLAG_FOR_HUMAN]
    clinical = [call for call in flags
                if _parse_arguments(call).get("clinical") or call["result"].get("category") == FlagCategory.CLINICAL]
    referral_id = next(
        (_parse_arguments(call)["referral_id"].upper() for call in ok if _parse_arguments(call).get("referral_id")),
        None)
    if referral_id is None:  # answered from memory: the answer names one of the patient's own referrals
        named = {found_id.upper() for found_id in _REFERRAL_ID.findall(answer or "")} & set(own_referrals)
        referral_id = min(named) if named else None
    details: list[str] = []

    if clinical:
        topic = ContactTopic.CLINICAL_QUESTION
    elif ToolName.BOOK_APPOINTMENT in names or ToolName.FIND_APPOINTMENT_SLOTS in names:
        topic = ContactTopic.APPOINTMENT_BOOKING
    elif referral_id:
        topic = ContactTopic.REFERRAL_STATUS
    elif ToolName.CALCULATE_COPAY in names:
        topic = ContactTopic.BILLING_QUESTION
    elif names & {ToolName.CHECK_ELIGIBILITY, ToolName.CHECK_PREAUTH}:
        topic = ContactTopic.ELIGIBILITY_CHECK
    elif ToolName.SEARCH_POLICIES in names:
        topic = ContactTopic.POLICY_QUESTION
    else:
        topic = ContactTopic.GENERAL_ENQUIRY
    for call in ok:
        if call["name"] == ToolName.BOOK_APPOINTMENT:
            booking = call["result"]
            details.append(
                f"Booked {booking.get('appointment_id')} for {booking.get('starts_at')} ({booking.get('payment')}).")
    if topic == ContactTopic.POLICY_QUESTION:  # say what actually happened, not just "answered"
        found = any((call["result"].get("count") or 0) > 0 for call in ok if call["name"] == ToolName.SEARCH_POLICIES)
        if not found:
            details.append("Not covered by the policy documents.")
        elif not flags:
            details.append("Answered from the policy documents.")
    for call in flags:
        kind = "clinical" if call in clinical else "staff"
        details.append(f"{kind.capitalize()} flag {call['result'].get('flag_id')} ({call['result'].get('urgency')}).")

    if clinical:
        outcome = ContactOutcome.ESCALATED_TO_CLINICIAN
    elif flags:
        outcome = ContactOutcome.CALLBACK_SCHEDULED
    else:
        outcome = ContactOutcome.RESOLVED
    return {"topic": topic.value, "outcome": outcome.value, "summary": _build_summary(topic, details),
            "referral_id": referral_id}


def _build_summary(topic: str, details: list[str]) -> str:
    """The stored one-line summary: the topic's fixed sentence, then one sentence per detail."""
    return " ".join([TOPIC_SUMMARY[topic], *details])


def _split_details(row: dict) -> list[str]:
    """The inverse of _build_summary: the detail sentences of a stored row, without its topic sentence. Sentences
    are split at full stops, so a detail must not contain one; all but the first keep their leading space."""
    base = TOPIC_SUMMARY.get(row["topic"], "")
    rest = row["summary"][len(base):] if row["summary"].startswith(base) else row["summary"]
    return re.findall(r"[^.]+\.", rest.strip()) if rest.strip() else []


def merge_contact(previous: dict, turn: dict) -> dict:
    """Combine a conversation's earlier record with a new turn: the most significant topic and outcome win,
    and the details (bookings, flags) accumulate."""
    topic = min(previous["topic"], turn["topic"],
                key=lambda candidate: TOPIC_RANK.index(candidate) if candidate in TOPIC_RANK else _UNRANKED)
    outcome = min(previous["outcome"], turn["outcome"],
                  key=lambda candidate: OUTCOME_RANK.index(candidate) if candidate in OUTCOME_RANK else _UNRANKED)
    details = [detail.strip() for detail in _split_details(previous)]
    details += [detail.strip() for detail in _split_details(turn) if detail.strip() not in details]
    return {
        "topic": topic,
        "outcome": outcome,
        "summary": _build_summary(topic, details),
        "referral_id": previous.get("referral_id") or turn.get("referral_id"),
    }
