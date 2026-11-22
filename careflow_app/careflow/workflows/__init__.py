"""M5 workflows. The stable interface for callers (demo scripts, tests, M8's API):

    cp = session_checkpointer()                                         # out/sessions.db, survives a restart
    result = start_triage(record, deps, cp, thread_id)                  # runs to the end, or to the pause
    payload = pending_review(result.thread_id, deps, cp)                # what staff are asked to decide
    result = resume_triage(result.thread_id, {"reviewer_id": ..., "action": "approve"}, deps, cp)

A thread_id is one case: starting a second run on a thread that is paused or finished raises
AlreadyTriaged. Without a checkpointer start_triage keeps the case only for that call (enough for an
automatic route or a routing check); pausing and resuming need a checkpointer, and the same store.
With a checkpointer and no deps, the durable runtime store (out/runtime.db) is used, so the checkpoints
and the flags and reviews they refer to live side by side.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ..domain import policies
from ..store import RuntimeStore
from ..tools import CareFlowTools
from .state import ReviewDecision, TriageState
from .triage import HANDOFF_ANSWER, NODES, TriageDeps, build_triage_graph

RECORD_KEYS = ("record_id", "channel", "received_at", "raw_text", "patient_ref")  # labels never enter the state
AWAITING, COMPLETED = "awaiting_human_review", "completed"


class NotAwaitingReview(RuntimeError):
    """resume_triage was called for a case that is not paused for staff."""


class AlreadyTriaged(RuntimeError):
    """start_triage was called on a thread_id that already holds a paused or finished case."""


@dataclass
class TriageResult:
    thread_id: str
    status: str  # awaiting_human_review | completed
    route: str
    reasons: list[str] = field(default_factory=list)
    response: str = ""
    band: str = "Routine"
    patient_ref: str | None = None
    eligibility: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    verdict: dict | None = None
    flag_id: str | None = None
    review_id: str | None = None
    decision: dict | None = None
    audit: list[dict] = field(default_factory=list)

    @property
    def steps(self) -> list[str]:
        return [a["step"] for a in self.audit]


def _result(thread_id: str, values: dict) -> TriageResult:
    return TriageResult(
        thread_id=thread_id, status=values.get("status") or COMPLETED, route=values["route"],
        reasons=values.get("reasons") or [], response=values.get("response") or "",
        band=values.get("band") or "Routine", patient_ref=values.get("patient_ref"),
        eligibility=values.get("eligibility") or {}, plan=values.get("plan") or {}, agent=values.get("agent") or {},
        verdict=values.get("verdict"),
        flag_id=values.get("flag_id"), review_id=values.get("review_id"), decision=values.get("decision"),
        audit=values.get("audit") or [],
    )


def _deps(deps: TriageDeps | None, durable: bool) -> TriageDeps:
    if deps is not None:
        return deps
    return TriageDeps.default(CareFlowTools(store=RuntimeStore())) if durable else TriageDeps.default()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_triage(record: dict, deps: TriageDeps | None = None, checkpointer: Any = None,
                 thread_id: str | None = None, answer: bool = True) -> TriageResult:
    """Run one incoming message through the workflow. `answer=False` routes it without running the assistant."""
    thread_id = thread_id or uuid.uuid4().hex
    deps = _deps(deps, durable=checkpointer is not None)
    graph = build_triage_graph(deps, checkpointer or MemorySaver())
    if graph.get_state(_config(thread_id)).values.get("status") in (AWAITING, COMPLETED):
        raise AlreadyTriaged(thread_id)
    # Every field is set, so a retry after a crashed run never reads the earlier attempt's route or flag.
    fresh: TriageState = {
        "record": {k: record[k] for k in RECORD_KEYS if k in record}, "answer": answer, "intake": {},
        "patient_ref": None, "band": "Routine", "repeat": None, "topic": "", "eligibility": {}, "plan": {},
        "route": "", "reasons": [], "response": "", "agent": {}, "verdict": None, "flag_id": None, "review_id": None,
        "decision": None, "status": "running",
    }
    with deps.agent.tracer.trace(thread_id, "triage", channel=record.get("channel")) as root:
        graph.invoke(fresh, _config(thread_id))
        result = _result(thread_id, graph.get_state(_config(thread_id)).values)
        root.set(route=result.route, case_status=result.status, band=result.band, patient_ref=result.patient_ref,
                 reasons_count=len(result.reasons), flag_id=result.flag_id, review_id=result.review_id)
    return result


def _paused_graph(deps: TriageDeps | None, checkpointer: Any):
    if checkpointer is None:
        raise ValueError("a paused case lives in a checkpointer: pass the one given to start_triage")
    deps = _deps(deps, durable=True)
    return build_triage_graph(deps, checkpointer), deps


def pending_review(thread_id: str, deps: TriageDeps | None = None, checkpointer: Any = None) -> dict | None:
    """What staff are asked to decide for a paused case, or None if it is not paused."""
    snapshot = _paused_graph(deps, checkpointer)[0].get_state(_config(thread_id))
    waiting = [i.value for t in snapshot.tasks for i in t.interrupts]
    return waiting[0] if waiting else None


def resume_triage(thread_id: str, decision: dict | ReviewDecision, deps: TriageDeps | None = None,
                  checkpointer: Any = None) -> TriageResult:
    """Record the staff decision and finish the case. An invalid decision raises and the case stays paused."""
    graph, deps = _paused_graph(deps, checkpointer)
    snapshot = graph.get_state(_config(thread_id))
    if "review" not in snapshot.next:
        raise NotAwaitingReview(thread_id)
    decision = decision if isinstance(decision, ReviewDecision) else ReviewDecision.model_validate(decision)
    problem = policies.validate_reband(snapshot.values["band"], decision.urgency, decision.note)
    if problem:
        raise ValueError(problem)
    with deps.agent.tracer.trace(thread_id, "triage.resume", action=decision.action,
                                 reviewer_id=decision.reviewer_id) as root:
        graph.invoke(Command(resume=decision.model_dump()), _config(thread_id))
        result = _result(thread_id, graph.get_state(_config(thread_id)).values)
        root.set(route=result.route, case_status=result.status, band=result.band, review_id=result.review_id)
    return result


__all__ = ["AlreadyTriaged", "HANDOFF_ANSWER", "NODES", "NotAwaitingReview", "ReviewDecision", "TriageDeps",
           "TriageResult", "TriageState", "build_triage_graph", "pending_review", "resume_triage", "start_triage"]
