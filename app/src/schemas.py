"""Validated structures for intake: what a parsed patient message looks like.

PatientRequest's JSON schema (field names, descriptions and the class docstring) is sent to the model as the
response format, so its wording is part of the prompt."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .domain.policies import SAME_DAY_BANDS

# The Literal types below stay Literals: they are part of the JSON schemas sent to the model and published by
# the API, and an enum would change those schemas. tests/test_states_and_constants.py checks each against
# its enum (policies.UrgencyBand, ContactChannel, MissingFieldName).
Urgency = Literal["Routine", "Priority", "Urgent", "Emergent"]
Specialty = Literal[
    "Cardiology",
    "Orthopaedics",
    "Dermatology",
    "Endocrinology",
    "Gastroenterology",
    "Pulmonology",
    "Neurology",
    "ENT",
]
PlanCode = Literal["MERIDIAN-GOLD", "MERIDIAN-SILVER", "MERIDIAN-BRONZE", "CIVIC-BASE"]
Channel = Literal["phone_transcript", "web_form", "email", "sms"]
MissingField = Literal["insurance_id", "specialty", "preferred_date", "callback_number"]


class ContactChannel(StrEnum):
    """The channel a message arrives on (the values of Channel)."""

    PHONE_TRANSCRIPT = "phone_transcript"
    WEB_FORM = "web_form"
    EMAIL = "email"
    SMS = "sms"


class MissingFieldName(StrEnum):
    """What an intake message can lack (the values of MissingField)."""

    INSURANCE_ID = "insurance_id"
    SPECIALTY = "specialty"
    PREFERRED_DATE = "preferred_date"
    CALLBACK_NUMBER = "callback_number"


DEFAULT_CHANNEL = ContactChannel.WEB_FORM.value  # recorded when a caller names no channel

# An insurance member number: plan prefix then digits, e.g. MER64867187 (a regex fragment, upper-case form).
MEMBER_NUMBER = r"[A-Z]{3}\d{6,10}"
# A member number after normalisation (upper case, no spaces or hyphens).
_MEMBER_NUMBER_PATTERN = re.compile(rf"^{MEMBER_NUMBER}$")


class PatientRequest(BaseModel):
    """One patient intake message, parsed. No clinical interpretation is made."""

    patient_name: str | None = Field(None, description="Name as the patient gave it, else null")
    symptoms: list[str] = Field(
        default_factory=list,
        description="Short phrases quoting what the patient reports. Never a diagnosis.",
    )
    urgency: Urgency
    specialty: Specialty | None = Field(None, description="Specialty requested or clearly implied, else null")
    insurance_plan: PlanCode | None = None
    insurance_member_number: str | None = Field(None, description="Member number such as MER12345678, else null")
    seeks_clinical_advice: bool = Field(
        description="True if the patient asks the clinic to interpret symptoms, "
        "diagnose, or advise on medication or dosing"
    )
    missing_fields: list[MissingField] = Field(default_factory=list)
    channel: Channel | None = Field(None, description="Filled in by the caller, not by the model")

    @property
    def urgent_flag(self) -> bool:
        """The urgency flag: Urgent and Emergent need same-day attention."""
        return self.urgency in SAME_DAY_BANDS

    @property
    def insurance_id(self) -> str | None:
        """Plan and member number, or the member number alone when no plan was stated.

        The member number is what identifies the patient (check_eligibility accepts either form), so a
        missing plan name must not hide it; that also keeps this in line with missing_fields.
        """
        if self.insurance_plan and self.insurance_member_number:
            return f"{self.insurance_plan} {self.insurance_member_number}"
        return self.insurance_member_number

    @field_validator("insurance_plan", mode="before")
    @classmethod
    def _normalise_plan(cls, value):
        """Plan names as people write them ("meridian silver", "Meridian_Silver") -> "MERIDIAN-SILVER"."""
        if isinstance(value, str):
            value = re.sub(r"[\s_]+", "-", value.strip().upper())
        return value or None

    @field_validator("insurance_member_number", mode="before")
    @classmethod
    def _normalise_member(cls, value):
        """Strip spaces and hyphens and upper-case ("mer 6486-7187" -> "MER64867187"). A value that is still
        not a member number is rejected, which sends the validation error back to the model for another try."""
        if value is None:
            return None
        value = re.sub(r"[\s\-]+", "", str(value)).upper()
        if not value:
            return None
        if not _MEMBER_NUMBER_PATTERN.match(value):
            raise ValueError(f"member number {value!r} must be 3 letters then 6-10 digits")
        return value

    @model_validator(mode="after")
    def _reconcile_missing_fields(self):
        """Keep missing_fields consistent with what was actually extracted."""
        found = set(self.missing_fields)
        specialty, insurance_id = MissingFieldName.SPECIALTY.value, MissingFieldName.INSURANCE_ID.value
        if self.specialty is None:
            found.add(specialty)
        else:
            found.discard(specialty)
        if self.insurance_member_number is None:
            found.add(insurance_id)
        else:
            found.discard(insurance_id)
        self.missing_fields = sorted(found)
        return self
