"""State and the staff decision for the M5 triage workflow. Plain dicts only, so checkpoints serialise simply."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import Urgency


class TriageState(TypedDict, total=False):
    record: dict            # the incoming message: channel, received_at, raw_text, patient_ref (if verified)
    answer: bool            # False = route only, do not run the assistant (routing evals)
    intake: dict            # PatientRequest as a dict (M1)
    patient_ref: str | None
    band: str               # urgency band after "highest band wins"
    repeat: str | None      # REPEAT / Repeat - Priority marker for this contact, if any
    topic: str              # what the message is about (memory topic), for the repeat rules
    eligibility: dict
    plan: dict              # specialty, sign-off need, referral, SLA date, pre-authorisation, payment path
    route: str              # auto | human_review | refuse
    reasons: list[str]
    response: str           # what the patient is told
    agent: dict             # summary of the assistant sub-step (tools used, flags, grounding)
    flag_id: str | None
    review_id: str | None
    decision: dict | None   # the staff decision (ReviewDecision) once given
    status: str             # awaiting_human_review | completed
    audit: Annotated[list[dict], operator.add]


class ReviewDecision(BaseModel):
    """What a staff member records to release a paused case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reviewer_id: str = Field(min_length=1, max_length=128)
    action: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=2000)
    urgency: Urgency | None = None  # a re-band; lowering needs a note (triage SOP 4.5)
