"""Tool contracts, defined once (architecture ADR-03).

Each tool's input is a Pydantic model. From it come:
- the OpenAI/LiteLLM function spec the model sees (TOOL_SPECS),
- validation of the model's arguments before any business logic runs (CareFlowTools.call),
- and in M6 the MCP server's tool schema (the same input model), so the three can never drift apart.
`idempotent` marks the read-only tools that M7 may retry safely; booking is not idempotent.
Outputs stay plain dicts: they go straight back to the model as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import PlanCode, Specialty, Urgency


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt argument name is an error, not silently ignored

    @model_validator(mode="before")
    @classmethod
    def _null_means_default(cls, data):
        """Models often send an explicit null for an optional argument they mean to omit
        ({"limit": null}). Drop it so the declared default (5, False, "Routine") applies. A null for a
        required field is still dropped, so it is reported as missing rather than accepted."""
        if isinstance(data, dict):
            return {
                k: v for k, v in data.items()
                if not (v is None and k in cls.model_fields and cls.model_fields[k].default is not None)
            }
        return data


class CheckEligibilityInput(_Input):
    patient_ref: str | None = Field(None, description="e.g. MHP-P-00001")
    member_number: str | None = Field(None, description="e.g. MER90272292")


class FindAppointmentSlotsInput(_Input):
    specialty: str = Field(
        description="Cardiology, Orthopaedics, Dermatology, Endocrinology, Gastroenterology, Pulmonology, "
        "Neurology or ENT"
    )
    mode: Literal["in_person", "telehealth"] | None = None
    earliest_date: str | None = Field(None, description="ISO date, e.g. 2026-08-05")
    limit: int = Field(5, description="max slots, default 5")
    patient_ref: str | None = None
    new_patient: bool | None = Field(None, description="override: true for a patient new to the clinic")


class BookAppointmentInput(_Input):
    patient_ref: str
    slot_id: str
    self_pay: bool = Field(False, description="true only after the patient agreed to self-pay")


class CheckPreauthInput(_Input):
    procedure_code: str | None = Field(None, description="e.g. ORTH-MRI, GAST-COLO")
    plan_code: PlanCode | None = None
    patient_ref: str | None = None
    referral_id: str | None = Field(None, description="e.g. MHP-R-00000")


class CalculateCopayInput(_Input):
    patient_ref: str | None = None
    procedure_code: str = Field(description="e.g. GAST-COLO, CARD-CONS")
    telehealth: bool = Field(False, description="true for a telehealth consultation")


class FlagForHumanInput(_Input):
    reason: str = Field(description="why staff are needed, in one sentence")
    urgency: Urgency = "Routine"
    patient_ref: str | None = None
    clinical: bool = Field(
        False, description="true for a clinical question or emergency (symptoms, diagnosis, medication, dosing, "
        "results); false for an administrative handoff"
    )


class FindPatientInput(_Input):
    given_name: str
    family_name: str
    birth_date: str | None = Field(None, description="date of birth, e.g. 1981-12-26 or 26/12/1981")
    member_number: str | None = Field(None, description="insurance member number, e.g. MER90272292")


class SearchPoliciesInput(_Input):
    query: str = Field(description="the policy or coverage question, in plain words")
    specialty: Specialty | None = Field(
        None, description="set only for a question about one specialty's service line handbook"
    )
    limit: int = Field(4, description="max passages, default 4")


@dataclass(frozen=True)
class ToolDefinition:
    name: str  # also the CareFlowTools method name
    description: str
    input_model: type[_Input]
    idempotent: bool

    def openai_spec(self) -> dict:
        schema = self.input_model.model_json_schema()
        props = {k: _plain(v) for k, v in schema.get("properties", {}).items()}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": schema.get("required", [])},
            },
        }


def _plain(prop: dict) -> dict:
    """Pydantic's JSON Schema -> the plain subset every provider accepts (no titles, defaults or null unions)."""
    prop = {k: v for k, v in prop.items() if k not in ("title", "default")}
    if "anyOf" in prop:
        options = [o for o in prop.pop("anyOf") if o.get("type") != "null"]
        if len(options) == 1:
            prop = {**options[0], **prop}
        else:
            prop["anyOf"] = options
    return prop


TOOLS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        "check_eligibility",
        "Look up a patient's insurance plan and eligibility status. Use before booking or quoting costs.",
        CheckEligibilityInput,
        idempotent=True,
    ),
    ToolDefinition(
        "find_appointment_slots",
        "Find open appointment slots for a specialty, soonest first. Pass patient_ref when known so only "
        "providers able to see this patient are shown.",
        FindAppointmentSlotsInput,
        idempotent=True,
    ),
    ToolDefinition(
        "book_appointment",
        "Book an open slot for a patient. Active coverage books on insurance; coverage awaiting verification "
        "books and staff verify it. Otherwise (e.g. lapsed) it books only as self-pay, and only after the "
        "patient has agreed to pay themselves. New patients only with providers accepting new patients.",
        BookAppointmentInput,
        idempotent=False,
    ),
    ToolDefinition(
        "check_preauth",
        "Check whether a procedure is covered under a plan, whether it needs pre-authorisation, and the "
        "approval window in business days. Give procedure_code with plan_code or patient_ref, or a referral_id.",
        CheckPreauthInput,
        idempotent=True,
    ),
    ToolDefinition(
        "calculate_copay",
        "Work out what the patient owes for a procedure under their plan. If insurance cannot be used "
        "(no insurance, coverage not active, procedure not covered) it gives the self-pay price instead. "
        "Leave out patient_ref for a patient with no insurance.",
        CalculateCopayInput,
        idempotent=True,
    ),
    ToolDefinition(
        "flag_for_human",
        "Hand the case to clinic staff. MUST be used for any clinical question (symptoms, diagnosis, "
        "medication or dosing), anything urgent, and anything the other tools cannot settle.",
        FlagForHumanInput,
        idempotent=True,  # same patient and reason is recorded once
    ),
    ToolDefinition(
        "find_patient",
        "Identify the patient you are talking to when you have no patient_ref: their full name plus their "
        "date of birth or member number. Returns only the patient_ref.",
        FindPatientInput,
        idempotent=True,
    ),
    ToolDefinition(
        "search_policies",
        "Search the clinic's policy and insurance documents (referral policy, pre-authorisation matrix, "
        "cost-share schedule, intake SOP, data handling, specialty handbooks) for rules, preparation "
        "instructions, timings and coverage terms. Returns passages with their document title.",
        SearchPoliciesInput,
        idempotent=True,
    ),
)
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
TOOL_SPECS = [t.openai_spec() for t in TOOLS]
