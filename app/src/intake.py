"""Intake: turn a raw patient message into a validated PatientRequest."""

from __future__ import annotations

from .config import Settings
from .llm_client import complete_structured
from .prompts import INTAKE_PROMPT
from .schemas import PatientRequest
from .state import Role

SYSTEM_PROMPT = INTAKE_PROMPT  # the text is in prompt_templates/intake_parser.md


def build_messages(record: dict) -> list[dict]:
    """Only the message itself reaches the model, never the labels or IDs."""
    user = f"Channel: {record['channel']}\nReceived: {record['received_at']}\n\nMessage:\n{record['raw_text']}"
    return [{"role": Role.SYSTEM.value, "content": SYSTEM_PROMPT}, {"role": Role.USER.value, "content": user}]


def parse_intake(record: dict, settings: Settings | None = None) -> PatientRequest:
    """Parse one intake record (keys: channel, received_at, raw_text). Makes a model call; raises
    LLMClientError if no model returns valid output. The channel is copied from the record, not parsed."""
    parsed = complete_structured(build_messages(record), PatientRequest, settings)
    return parsed.model_copy(update={"channel": record["channel"]})
