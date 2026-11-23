"""Graph state. total=False so later milestones add keys (retrieved docs, human decision)
without touching existing nodes."""

from __future__ import annotations

from typing import Annotated, TypedDict

REDACT = "redact"  # not a chat role: an instruction to the reducer below


def add_messages(old: list[dict], new: list[dict]) -> list[dict]:
    """Append, as before. One exception: {"role": "redact", "content": text, "count": n} replaces the
    text of the last n assistant answers instead of being appended. The Clinical Safety Reviewer reads a
    reply after the agent has stored it; a vetoed reply must not stay in the session, where the next turn
    would send it back to the model and treat it as a trusted source. Text the model wrote beside a tool
    call earlier in the same turn ("Those are beta blockers; let me check your co-pay") is removed too: the
    patient never saw it, but the next turn would."""
    out = list(old or [])
    for m in new or []:
        if m.get("role") != REDACT:
            out.append(m)
            continue
        left, in_turn = int(m.get("count") or 1), False  # in_turn: between a redacted answer and its user message
        for i in range(len(out) - 1, -1, -1):
            role, calls = out[i].get("role"), out[i].get("tool_calls")
            if role == "user":
                if in_turn and not left:
                    break
                in_turn = False
            elif left and role == "assistant" and out[i].get("content") and not calls:
                out[i] = {**out[i], "content": m.get("content") or ""}
                left, in_turn = left - 1, True
            elif in_turn and role == "assistant" and calls and out[i].get("content"):
                out[i] = {**out[i], "content": None}
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
    # M3: long-term memory note for patient_ref, rebuilt from the runtime store before every model call
    # (so a booking made a minute ago is already in it). Never holds another patient's data.
    patient_memory: str | None
    channel: str  # contact channel recorded with the interaction (web_form, sms, email, phone)
    # M4: the model's final answer waits here until the `ground` node checks it; only then is it (or a staff
    # handoff in its place) added to messages. M8's guard node checks the same draft.
    draft: dict | None
    # M4: groundedness check of this turn's answer (grounding.GroundingResult.as_dict()); reset every turn.
    grounding: dict
