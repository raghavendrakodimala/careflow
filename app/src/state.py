"""Graph state. total=False so a node returns only the keys it changes and a new key never touches
existing nodes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, TypedDict


class Role(StrEnum):
    """The `role` of a message dict. A stored or sent message holds the plain string (`.value`)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    REDACT = "redact"  # not a chat role: an instruction to the reducer below


REDACT = Role.REDACT  # module-level alias of Role.REDACT

# The keys of a LangGraph run config that name the conversation: {"configurable": {"thread_id": ...}}.
CONFIGURABLE = "configurable"
THREAD_ID = "thread_id"


def build_thread_config(thread_id: str) -> dict:
    """The run config that selects one stored conversation (a checkpoint thread)."""
    return {CONFIGURABLE: {THREAD_ID: thread_id}}


def add_messages(old: list[dict], new: list[dict]) -> list[dict]:
    """Reducer for `messages`: appends `new` to `old`. One exception:
    {"role": "redact", "content": text, "count": n} replaces the
    text of the last n assistant answers instead of being appended. The Clinical Safety Reviewer reads a
    reply after the agent has stored it; a vetoed reply must not stay in the session, where the next turn
    would send it back to the model and treat it as a trusted source. Text the model wrote beside a tool
    call earlier in the same turn ("Those are beta blockers; let me check your co-pay") is removed too: the
    patient never saw it, but the next turn would."""
    out = list(old or [])
    for message in new or []:
        if message.get("role") != Role.REDACT:
            out.append(message)
            continue
        # Walk back from the newest message. `left` = answers still to redact; `in_turn` = we are between a
        # redacted answer and the user message that opened its turn.
        left, in_turn = int(message.get("count") or 1), False
        for index in range(len(out) - 1, -1, -1):
            role, calls = out[index].get("role"), out[index].get("tool_calls")
            if role == Role.USER:
                if in_turn and not left:
                    break  # the last redacted answer's turn is fully cleaned
                in_turn = False
            elif left and role == Role.ASSISTANT and out[index].get("content") and not calls:
                out[index] = {**out[index], "content": message.get("content") or ""}  # a final answer: replace its text
                left, in_turn = left - 1, True
            elif in_turn and role == Role.ASSISTANT and calls and out[index].get("content"):
                out[index] = {**out[index], "content": None}  # text beside a tool call: drop it, keep the call
    return out


class CareFlowState(TypedDict, total=False):
    # Plain OpenAI-style dicts (what LiteLLM takes), appended across turns on the same thread.
    # The system prompt is NOT stored here: the model node prepends it on every call.
    messages: Annotated[list[dict], add_messages]
    # Verified patient for this thread: given by the caller (authenticated channel) or set by a
    # successful find_patient. Kept across turns by the checkpointer.
    patient_ref: str | None
    steps: int  # model calls this turn; reset to 0 by the caller on every new turn
    escalated: bool  # True when the step cap sent the case to staff
    # Long-term memory note for patient_ref, rebuilt from the runtime store before every model call
    # (so a booking made a minute ago is already in it). Never holds another patient's data.
    patient_memory: str | None
    channel: str  # contact channel recorded with the interaction (web_form, sms, email, phone)
    # The model's final answer waits here until the `ground` node checks it; only then is it (or a staff
    # handoff in its place) added to messages. The guard node checks the same draft.
    draft: dict | None
    # Groundedness check of this turn's answer (grounding.GroundingResult.to_dict()); reset every turn.
    grounding: dict
