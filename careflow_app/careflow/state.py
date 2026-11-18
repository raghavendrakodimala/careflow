"""Graph state. total=False so later milestones add keys (memory, retrieved docs, human decision)
without touching existing nodes."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class CareFlowState(TypedDict, total=False):
    # Plain OpenAI-style dicts (what LiteLLM takes), appended across turns on the same thread.
    # The system prompt is NOT stored here: the model node prepends it on every call.
    messages: Annotated[list[dict], operator.add]
    patient_ref: str | None
    steps: int  # model calls this turn; reset to 0 by the caller on every new turn
    escalated: bool  # True when the step cap sent the case to staff
