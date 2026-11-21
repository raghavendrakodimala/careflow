"""Graph state. total=False so later milestones add keys (retrieved docs, human decision)
without touching existing nodes."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class CareFlowState(TypedDict, total=False):
    # Plain OpenAI-style dicts (what LiteLLM takes), appended across turns on the same thread.
    # The system prompt is NOT stored here: the model node prepends it on every call.
    messages: Annotated[list[dict], operator.add]
    # Verified patient for this thread: given by the caller (authenticated channel) or set by a
    # successful find_patient. Kept across turns by the checkpointer.
    patient_ref: str | None
    steps: int  # model calls this turn; reset to 0 by the caller on every new turn
    escalated: bool  # True when the step cap sent the case to staff
    # M3: long-term memory note for patient_ref, rebuilt from the runtime store before every model call
    # (so a booking made a minute ago is already in it). Never holds another patient's data.
    patient_memory: str | None
    channel: str  # contact channel recorded with the interaction (web_form, sms, email, phone)
    # M4: the model's final answer waits here until the `ground` node checks it; only then is it (or a staff
    # handoff in its place) added to messages. M8's guard node checks the same draft.
    draft: dict | None
    # M4: groundedness check of this turn's answer (grounding.GroundingResult.as_dict()); reset every turn.
    grounding: dict
