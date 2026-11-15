"""Validated structures for M1: what a parsed patient message looks like."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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

_MEMBER_RE = re.compile(r"^[A-Z]{3}\d{6,10}$")


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
        """The brief's 'urgency flag': Urgent and Emergent need same-day attention."""
        return self.urgency in ("Urgent", "Emergent")

    @property
    def insurance_id(self) -> str | None:
        """Plan and member number in the form the patients table stores them."""
        if self.insurance_plan and self.insurance_member_number:
            return f"{self.insurance_plan} {self.insurance_member_number}"
        return None

    @field_validator("insurance_plan", mode="before")
    @classmethod
    def _normalise_plan(cls, v):
        if isinstance(v, str):
            v = re.sub(r"[\s_]+", "-", v.strip().upper())
        return v or None

    @field_validator("insurance_member_number", mode="before")
    @classmethod
    def _normalise_member(cls, v):
        if v is None:
            return None
        v = re.sub(r"[\s\-]+", "", str(v)).upper()
        if not v:
            return None
        if not _MEMBER_RE.match(v):
            raise ValueError(f"member number {v!r} must be 3 letters then 6-10 digits")
        return v

    @model_validator(mode="after")
    def _missing_fields_agree(self):
        """Keep missing_fields consistent with what was actually extracted."""
        found = set(self.missing_fields)
        if self.specialty is None:
            found.add("specialty")
        else:
            found.discard("specialty")
        if self.insurance_member_number is None:
            found.add("insurance_id")
        else:
            found.discard("insurance_id")
        self.missing_fields = sorted(found)
        return self
