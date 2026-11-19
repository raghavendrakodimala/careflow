"""M3 long-term patient memory, read from and written to the runtime store (architecture §5.5, tier 2).

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
- every read is one patient's rows (RuntimeStore.rows_for_patient); nothing is loaded for anyone else;
- no names, birth dates or member numbers appear in the memory text;
- an unknown patient_ref gives no memory and records nothing.
Session memory (the conversation itself) is separate: the LangGraph checkpointer, keyed by thread_id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .domain.clock import Clock, FrozenClock
from .store import RuntimeStore

REPEAT_WINDOW = timedelta(hours=24)  # triage SOP OPS-INT-004 clause 7.3
PRIORITY_WINDOW = timedelta(hours=72)  # OP-CS-014 clause 8.2: third contact on the same issue
OPEN_REFERRAL_STATUSES = ("open", "accepted", "scheduled")

# The seed table's five topics and one-line summaries, plus two the assistant needs: a policy question
# answered from the documents, and a contact with no specific issue (answered from records, or handed to
# staff for an administrative reason).
TOPIC_SUMMARY = {
    "clinical_question": "Asked a clinical question; not answered, routed to a clinician.",
    "appointment_booking": "Asked to book or move an appointment.",
    "referral_status": "Asked for the status of a referral.",
    "billing_question": "Queried a co-pay or an amount on a statement.",
    "eligibility_check": "Asked whether their plan is active and what it covers.",
    "policy_question": "Asked about a clinic policy; answered from the policy documents.",
    "general_enquiry": "Contacted the clinic about their records.",
}
TOPIC_RANK = list(TOPIC_SUMMARY)  # earlier = more significant when one contact covers several turns
OUTCOME_RANK = ["escalated_to_clinician", "callback_scheduled", "routed_to_billing", "resolved"]
NO_ISSUE = "general_enquiry"  # not an "issue" for the repeat rules: same-issue cannot be established
REPEAT_MARK = "REPEAT"
PRIORITY_MARK = "Repeat - Priority"


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=None)


@dataclass
class PatientMemory:
    store: RuntimeStore
    clock: Clock = FrozenClock()
    max_contacts: int = 5

    def _known(self, patient_ref: str | None) -> str | None:
        """The canonical patient_ref if it is on record, else None."""
        ref = str(patient_ref or "").strip().upper()
        return ref if ref and self.store.one("patients", "patient_ref", ref) is not None else None

    def _now(self) -> datetime:
        return self.clock.now().replace(tzinfo=None)

    # ---- recall ------------------------------------------------------------------------------
    def recall(self, patient_ref: str | None, exclude_thread: str | None = None) -> dict | None:
        """Memory for one patient. `exclude_thread`: the current conversation, which is not an earlier contact."""
        patient_ref = self._known(patient_ref)
        if patient_ref is None:
            return None
        now = self._now()
        contacts = sorted(
            (
                r for r in self.store.rows_for_patient("patient_interactions", patient_ref)
                if _ts(r["occurred_at"]) <= now and not (exclude_thread and r.get("thread_id") == exclude_thread)
            ),
            key=lambda r: r["occurred_at"],
            reverse=True,
        )

        def topics_within(window: timedelta) -> dict[str, int]:
            counts: dict[str, int] = {}
            for r in contacts:
                if now - _ts(r["occurred_at"]) <= window and r["topic"] != NO_ISSUE:
                    counts[r["topic"]] = counts.get(r["topic"], 0) + 1
            return counts

        appts = self.store.rows_for_patient("appointments", patient_ref)
        upcoming = sorted(
            (a for a in appts if a["status"] == "scheduled" and _ts(a["scheduled_for"]) >= now),
            key=lambda a: a["scheduled_for"],
        )
        completed = sorted(
            (a for a in appts if a["status"] == "completed" and _ts(a["scheduled_for"]) <= now),
            key=lambda a: a["scheduled_for"],
        )
        referrals = []
        for r in sorted(self.store.rows_for_patient("referrals", patient_ref), key=lambda r: r["raised_on"]):
            if r["status"] not in OPEN_REFERRAL_STATUSES:
                continue
            d = self.store.one("referral_details", "referral_id", r["referral_id"]) or {}
            referrals.append({
                "referral_id": r["referral_id"],
                "to_specialty": r["to_specialty"],
                "urgency": r["urgency"],
                "status": r["status"],
                "raised_on": r["raised_on"],
                "procedure_code": d.get("procedure_code"),
                "preauth_status": d.get("preauth_status"),
                "sla_due_on": d.get("sla_due_on"),
            })
        keep = ("occurred_at", "channel", "topic", "outcome", "summary", "referral_id")
        repeat = sorted(topics_within(REPEAT_WINDOW))
        priority = sorted(t for t, n in topics_within(PRIORITY_WINDOW).items() if n >= 2)
        return {
            "patient_ref": patient_ref,
            "as_of": now.isoformat(timespec="minutes"),
            "recent_contacts": [{k: c[k] for k in keep} for c in contacts[: self.max_contacts]],
            "total_contacts": len(contacts),
            "repeat_contact": bool(repeat),
            "repeat_topics": repeat,  # same issue again now = REPEAT (SOP 7.3)
            "repeat_priority_topics": priority,  # same issue again now = third within 72h (OP-CS-014 8.2)
            "upcoming_appointments": [
                {k: a[k] for k in ("appointment_id", "specialty", "scheduled_for", "status", "payment")}
                for a in upcoming
            ],
            "last_completed_visit": (
                {"specialty": completed[-1]["specialty"], "date": completed[-1]["scheduled_for"][:10]} if completed else None
            ),
            "open_referrals": referrals,
        }

    def context(self, patient_ref: str | None, exclude_thread: str | None = None) -> str | None:
        """The memory as a short note for the model, or None for an unknown patient."""
        m = self.recall(patient_ref, exclude_thread)
        if m is None:
            return None
        lines = [f"Patient memory for {m['patient_ref']} (verified; clinic records as of {m['as_of']}):"]
        if m["repeat_topics"]:
            lines.append(
                f"- Contacted within the last 24 hours about: {', '.join(m['repeat_topics'])}. If this contact is "
                "about the same issue, it is a REPEAT (triage SOP 7.3)."
            )
        if m["repeat_priority_topics"]:
            lines.append(
                f"- Two or more contacts in the last 72 hours about: {', '.join(m['repeat_priority_topics'])}. If "
                "this contact is about the same issue, it is the third: Repeat - Priority (OP-CS-014 8.2)."
            )
        if m["recent_contacts"]:
            lines.append(f"- Earlier contacts ({m['total_contacts']} on record, newest first):")
            for c in m["recent_contacts"]:
                ref = f" [{c['referral_id']}]" if c["referral_id"] else ""
                lines.append(f"  - {c['occurred_at']} {c['channel']}: {c['topic']} -> {c['outcome']}. {c['summary']}{ref}")
        else:
            lines.append("- No earlier contacts on record.")
        for a in m["upcoming_appointments"]:
            lines.append(f"- Upcoming appointment {a['appointment_id']}: {a['specialty']} on {a['scheduled_for']}.")
        if m["last_completed_visit"]:
            v = m["last_completed_visit"]
            lines.append(f"- Last completed visit: {v['specialty']} on {v['date']}.")
        for r in m["open_referrals"]:
            lines.append(
                f"- Open referral {r['referral_id']} to {r['to_specialty']} ({r['urgency']}): status {r['status']}, "
                f"pre-authorisation {r['preauth_status']}, SLA due {r['sla_due_on']}."
            )
        return "\n".join(lines)

    def repeat_marker(self, patient_ref: str | None, issue: str, exclude_thread: str | None = None) -> str | None:
        """The marker a staff flag on `issue` needs for this patient: "Repeat - Priority" (third contact on the
        same issue within 72h, OP-CS-014 8.2), "REPEAT" (same issue within 24h, SOP 7.3), or None."""
        if issue == NO_ISSUE:
            return None
        m = self.recall(patient_ref, exclude_thread)
        if m is None:
            return None
        if issue in m["repeat_priority_topics"]:
            return PRIORITY_MARK
        if issue in m["repeat_topics"]:
            return REPEAT_MARK
        return None

    # ---- record ------------------------------------------------------------------------------
    def record_turn(
        self,
        patient_ref: str | None,
        tool_calls: list[dict],
        channel: str = "web_form",
        thread_id: str | None = None,
        answer: str = "",
    ) -> dict | None:
        """Record this turn as (part of) a contact. With a thread_id the conversation's one row is created or
        updated; without one each call is its own contact. Returns the row, or None for an unknown patient."""
        patient_ref = self._known(patient_ref)
        if patient_ref is None:
            return None
        own_referrals = {r["referral_id"] for r in self.store.rows_for_patient("referrals", patient_ref)}
        turn = summarise_turn(tool_calls, answer, own_referrals)
        previous = self.store.interaction_for_thread(patient_ref, thread_id) if thread_id else None
        merged = merge_contact(previous, turn) if previous else turn
        return self.store.add_interaction(
            patient_ref=patient_ref,
            occurred_at=self._now().isoformat(timespec="minutes"),
            channel=channel,
            thread_id=thread_id,
            **merged,
        )


def _args(call: dict) -> dict:
    a = call.get("arguments") or {}
    if isinstance(a, str):
        try:
            return json.loads(a or "{}")
        except json.JSONDecodeError:
            return {}
    return a


_REFERRAL_ID = re.compile(r"MHP-R-\d{5}", re.I)


def summarise_turn(tool_calls: list[dict], answer: str = "", own_referrals: set[str] | frozenset = frozenset()) -> dict:
    """One turn as {topic, outcome, summary, referral_id}, from its tool calls ({name, arguments, result}).

    Only what the tools did, and which of the patient's own referrals the answer named, is used.
    A clinical question is recognised by a clinical staff flag (flag_for_human with clinical=true), never
    by "only a flag was raised": step-cap and administrative handoffs are not clinical."""
    ok = [c for c in tool_calls if (c.get("result") or {}).get("ok")]
    names = {c["name"] for c in ok}
    flags = [c for c in ok if c["name"] == "flag_for_human"]
    clinical = [c for c in flags if _args(c).get("clinical") or c["result"].get("category") == "clinical"]
    referral_id = next((_args(c)["referral_id"].upper() for c in ok if _args(c).get("referral_id")), None)
    if referral_id is None:  # answered from memory: the answer names one of the patient's own referrals
        named = {m.upper() for m in _REFERRAL_ID.findall(answer or "")} & set(own_referrals)
        referral_id = min(named) if named else None
    details: list[str] = []

    if clinical:
        topic = "clinical_question"
    elif "book_appointment" in names or "find_appointment_slots" in names:
        topic = "appointment_booking"
    elif referral_id:
        topic = "referral_status"
    elif "calculate_copay" in names:
        topic = "billing_question"
    elif names & {"check_eligibility", "check_preauth"}:
        topic = "eligibility_check"
    elif "search_policies" in names:
        topic = "policy_question"
    else:
        topic = NO_ISSUE
    for c in ok:
        if c["name"] == "book_appointment":
            b = c["result"]
            details.append(f"Booked {b.get('appointment_id')} for {b.get('starts_at')} ({b.get('payment')}).")
    for c in flags:
        kind = "clinical" if c in clinical else "staff"
        details.append(f"{kind.capitalize()} flag {c['result'].get('flag_id')} ({c['result'].get('urgency')}).")

    if clinical:
        outcome = "escalated_to_clinician"
    elif flags:
        outcome = "callback_scheduled"
    else:
        outcome = "resolved"
    return {"topic": topic, "outcome": outcome, "summary": _summary(topic, details), "referral_id": referral_id}


def _summary(topic: str, details: list[str]) -> str:
    return " ".join([TOPIC_SUMMARY[topic], *details])


def _details(row: dict) -> list[str]:
    base = TOPIC_SUMMARY.get(row["topic"], "")
    rest = row["summary"][len(base):] if row["summary"].startswith(base) else row["summary"]
    return re.findall(r"[^.]+\.", rest.strip()) if rest.strip() else []


def merge_contact(previous: dict, turn: dict) -> dict:
    """Combine a conversation's earlier record with a new turn: the most significant topic and outcome win,
    and the details (bookings, flags) accumulate."""
    topic = min(previous["topic"], turn["topic"], key=lambda t: TOPIC_RANK.index(t) if t in TOPIC_RANK else 99)
    outcome = min(previous["outcome"], turn["outcome"],
                  key=lambda o: OUTCOME_RANK.index(o) if o in OUTCOME_RANK else 99)
    details = _details(previous)
    details += [d.strip() for d in _details(turn) if d.strip() not in details]
    return {
        "topic": topic,
        "outcome": outcome,
        "summary": _summary(topic, [d.strip() for d in details]),
        "referral_id": previous.get("referral_id") or turn.get("referral_id"),
    }
