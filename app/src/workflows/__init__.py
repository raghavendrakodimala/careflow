"""The triage workflow's stable interface for callers (demo scripts, tests, the API):

    checkpointer = open_session_checkpointer()                             # out/sessions.db, survives a restart
    result = start_triage(record, dependencies, checkpointer, thread_id)   # runs to the end, or to the pause
    payload = get_pending_review(result.thread_id, dependencies, checkpointer)  # what staff are asked to decide
    result = resume_triage(result.thread_id, {"reviewer_id": ..., "action": "approve"}, dependencies, checkpointer)

A thread_id is one case: starting a second run on a thread that is paused or finished raises
AlreadyTriagedError. Without a checkpointer start_triage keeps the case only for that call (enough for an
automatic route or a routing check); pausing and resuming need a checkpointer, and the same store.
With a checkpointer and no dependencies, the durable runtime store (out/runtime.db) is used, so the checkpoints
and the flags and reviews they refer to live side by side.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ..domain import policies
from ..state import build_thread_config
from ..store import RuntimeStore
from ..tools import CareFlowTools
from .state import CaseStatus, ReviewDecision, TriageState
from .triage import HANDOFF_ANSWER, NODES, TriageDependencies, TriageNode, build_triage_graph

# The fields of an incoming record that enter the state. Anything else (an evaluation record's labels) is dropped.
RECORD_KEYS = ("record_id", "channel", "received_at", "raw_text", "patient_ref")


class NotAwaitingReviewError(RuntimeError):
    """resume_triage was called for a case that is not paused for staff."""


class AlreadyTriagedError(RuntimeError):
    """start_triage was called on a thread_id that already holds a paused or finished case."""


@dataclass
class TriageResult:
    """A case as it stands after a run: paused for staff or finished. The fields mirror TriageState."""

    thread_id: str
    status: str  # awaiting_human_review | completed
    route: str
    reasons: list[str] = field(default_factory=list)
    response: str = ""
    band: str = policies.DEFAULT_BAND
    patient_ref: str | None = None
    eligibility: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    verdict: dict | None = None
    guard: dict = field(default_factory=dict)
    flag_id: str | None = None
    review_id: str | None = None
    decision: dict | None = None
    audit: list[dict] = field(default_factory=list)

    @property
    def steps(self) -> list[str]:
        """The nodes that left an audit entry, in the order they ran."""
        return [entry["step"] for entry in self.audit]


def _build_result(thread_id: str, values: dict) -> TriageResult:
    """The TriageResult for a thread's saved state values."""
    return TriageResult(
        thread_id=thread_id, status=values.get("status") or CaseStatus.COMPLETED.value, route=values["route"],
        reasons=values.get("reasons") or [], response=values.get("response") or "",
        band=values.get("band") or policies.DEFAULT_BAND, patient_ref=values.get("patient_ref"),
        eligibility=values.get("eligibility") or {}, plan=values.get("plan") or {}, agent=values.get("agent") or {},
        verdict=values.get("verdict"), guard=values.get("guard") or {},
        flag_id=values.get("flag_id"), review_id=values.get("review_id"), decision=values.get("decision"),
        audit=values.get("audit") or [],
    )


def _resolve_dependencies(dependencies: TriageDependencies | None, durable: bool) -> TriageDependencies:
    """The caller's dependencies, or the default ones. `durable`: on the file-backed runtime store, so the flags and
    reviews a saved checkpoint refers to outlive the process too."""
    if dependencies is not None:
        return dependencies
    return TriageDependencies.build_default(CareFlowTools(store=RuntimeStore())) if durable else TriageDependencies.build_default()


def _get_thread_config(thread_id: str) -> dict:
    return build_thread_config(thread_id)


def start_triage(record: dict, dependencies: TriageDependencies | None = None, checkpointer: Any = None,
                 thread_id: str | None = None, answer: bool = True) -> TriageResult:
    """Run one incoming message through the workflow. `answer=False` routes it without running the assistant."""
    thread_id = thread_id or uuid.uuid4().hex
    dependencies = _resolve_dependencies(dependencies, durable=checkpointer is not None)
    graph = build_triage_graph(dependencies, checkpointer or MemorySaver())
    if graph.get_state(_get_thread_config(thread_id)).values.get("status") in (
            CaseStatus.AWAITING_HUMAN_REVIEW, CaseStatus.COMPLETED):
        raise AlreadyTriagedError(thread_id)
    # Every field is set, so a retry after a crashed run never reads the earlier attempt's route or flag.
    fresh: TriageState = {
        "record": {key: record[key] for key in RECORD_KEYS if key in record}, "answer": answer, "guard": {},
        "intake": {},
        "patient_ref": None, "band": policies.DEFAULT_BAND, "repeat": None, "topic": "", "eligibility": {}, "plan": {},
        "route": "", "reasons": [], "response": "", "agent": {}, "verdict": None, "flag_id": None, "review_id": None,
        "decision": None, "status": CaseStatus.RUNNING.value,
    }
    with dependencies.agent.tracer.open_trace(thread_id, "triage", channel=record.get("channel")) as root:
        graph.invoke(fresh, _get_thread_config(thread_id))
        result = _build_result(thread_id, graph.get_state(_get_thread_config(thread_id)).values)
        root.set(route=result.route, case_status=result.status, band=result.band, patient_ref=result.patient_ref,
                 reasons_count=len(result.reasons), flag_id=result.flag_id, review_id=result.review_id)
    return result


def _load_paused_graph(dependencies: TriageDependencies | None, checkpointer: Any):
    """(graph, dependencies) for reading or resuming a paused case."""
    if checkpointer is None:
        raise ValueError("a paused case lives in a checkpointer: pass the one given to start_triage")
    dependencies = _resolve_dependencies(dependencies, durable=True)
    return build_triage_graph(dependencies, checkpointer), dependencies


def get_pending_review(thread_id: str, dependencies: TriageDependencies | None = None, checkpointer: Any = None) -> dict | None:
    """What staff are asked to decide for a paused case, or None if it is not paused."""
    snapshot = _load_paused_graph(dependencies, checkpointer)[0].get_state(_get_thread_config(thread_id))
    waiting = [interrupt.value for task in snapshot.tasks for interrupt in task.interrupts]
    return waiting[0] if waiting else None


def resume_triage(thread_id: str, decision: dict | ReviewDecision, dependencies: TriageDependencies | None = None,
                  checkpointer: Any = None) -> TriageResult:
    """Record the staff decision and finish the case. An invalid decision raises and the case stays paused."""
    graph, dependencies = _load_paused_graph(dependencies, checkpointer)
    snapshot = graph.get_state(_get_thread_config(thread_id))
    if TriageNode.REVIEW not in snapshot.next:
        raise NotAwaitingReviewError(thread_id)
    decision = decision if isinstance(decision, ReviewDecision) else ReviewDecision.model_validate(decision)
    problem = policies.validate_reband(snapshot.values["band"], decision.urgency, decision.note)
    if problem:
        raise ValueError(problem)
    with dependencies.agent.tracer.open_trace(thread_id, "triage.resume", action=decision.action,
                                 reviewer_id=decision.reviewer_id) as root:
        graph.invoke(Command(resume=decision.model_dump()), _get_thread_config(thread_id))
        result = _build_result(thread_id, graph.get_state(_get_thread_config(thread_id)).values)
        root.set(route=result.route, case_status=result.status, band=result.band, review_id=result.review_id)
    return result


__all__ = [
    "HANDOFF_ANSWER",
    "NODES",
    "AlreadyTriagedError",
    "NotAwaitingReviewError",
    "ReviewDecision",
    "TriageDependencies",
    "TriageResult",
    "TriageState",
    "build_triage_graph",
    "get_pending_review",
    "resume_triage",
    "start_triage",
]
