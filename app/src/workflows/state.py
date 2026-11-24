"""State and the staff decision for the triage workflow. Plain dicts only, so checkpoints serialise simply."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import Urgency


class CaseStatus(StrEnum):
    """TriageState["status"]. Each member equals its string value; the saved state holds the plain string."""

    RUNNING = "running"                              # until a run has stopped
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"  # paused for a staff decision
    COMPLETED = "completed"


class ReviewAction(StrEnum):
    """What staff decide on a paused case (the values of ReviewDecision.action, which stays a Literal so the
    API's published schema does not change)."""

    APPROVE = "approve"
    REJECT = "reject"


AWAITING, COMPLETED = CaseStatus.AWAITING_HUMAN_REVIEW, CaseStatus.COMPLETED  # short aliases


class TriageState(TypedDict, total=False):
    record: dict            # the incoming message: channel, received_at, raw_text, patient_ref (if verified)
    answer: bool            # False = route only, do not run the assistant (routing evals)
    guard: dict             # the input gate's result: blocked, category, layer, reason
    intake: dict            # PatientRequest as a dict; empty when the message was blocked or could not be parsed
    patient_ref: str | None  # verified by the channel, or matched on full name plus member number
    band: str               # urgency band after "highest band wins"
    repeat: str | None      # REPEAT / Repeat - Priority marker for this contact, if any
    topic: str              # what the message is about (memory topic), for the repeat rules
    eligibility: dict       # coverage status, plan, payment path; `unavailable` when the EHR did not answer
    plan: dict              # specialty, sign-off need, referral, SLA date, pre-authorisation, payment path
    route: str              # auto | human_review | refuse
    reasons: list[str]      # why the route is not auto, as staff read them
    response: str           # what the patient is told
    agent: dict             # summary of the team step (specialists, tools used, flags, grounding)
    verdict: dict | None    # the Clinical Safety Reviewer's verdict on the reply
    flag_id: str | None     # the staff flag for this case
    review_id: str | None   # the review request staff decide on
    decision: dict | None   # the staff decision (ReviewDecision) once given
    status: str             # running | awaiting_human_review | completed
    audit: Annotated[list[dict], operator.add]  # the nodes' audit entries; appended, never replaced


class ReviewDecision(BaseModel):
    """What a staff member records to release a paused case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reviewer_id: str = Field(min_length=1, max_length=128)
    action: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=2000)
    urgency: Urgency | None = None  # a re-band; lowering needs a note (triage SOP 4.5)
