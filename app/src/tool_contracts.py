"""Tool contracts, defined once.

Each tool's input is a Pydantic model. From it come:
- the OpenAI/LiteLLM function spec the model sees (TOOL_SPECS),
- validation of the model's arguments before any business logic runs (CareFlowTools.call),
- and the MCP server's tool schema (the same input model), so the three can never drift apart.
`idempotent` marks the tools the reliability layer may retry safely; booking is not idempotent.
Outputs stay plain dicts: they go straight back to the model as JSON.

The tool descriptions and field descriptions are prompt text, tuned against the live model. The input
models deliberately have no class docstrings: Pydantic would add one to the schema as a description.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import PlanCode, Specialty, Urgency

# Hard caps on result sizes, whatever limit a caller asks for. The tools apply the same caps.
MAX_SLOTS = 20
MAX_PASSAGES = 8
# How many are returned when the caller gives no limit (the tool methods use the same defaults).
DEFAULT_SLOTS = 5
DEFAULT_PASSAGES = 4


class ToolName(StrEnum):
    """The tools, by the name the model calls them (also the CareFlowTools method names). Each member equals
    its name; a spec, a message, a span or a result holds the plain string (`.value`)."""

    CHECK_ELIGIBILITY = "check_eligibility"
    FIND_APPOINTMENT_SLOTS = "find_appointment_slots"
    BOOK_APPOINTMENT = "book_appointment"
    CHECK_PREAUTH = "check_preauth"
    CALCULATE_COPAY = "calculate_copay"
    FLAG_FOR_HUMAN = "flag_for_human"
    FIND_PATIENT = "find_patient"
    SEARCH_POLICIES = "search_policies"


class Backend(StrEnum):
    """The system a tool depends on, for retries and circuit breakers."""

    EHR = "ehr"                    # served over MCP
    POLICY_INDEX = "policy_index"
    LOCAL = "local"                # only this process's own store: no retries, no breaker


# The trusted value the server adds to a tool call (CareFlowTools.call `context`), never set by the model:
# the contact (conversation) the call belongs to. Also flag_for_human's parameter and the run config's key.
CONTACT_ID = "contact_id"


# Base of every tool's input model.
class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt argument name is an error, not silently ignored

    @model_validator(mode="before")
    @classmethod
    def _drop_null_for_default(cls, data):
        """Models often send an explicit null for an optional argument they mean to omit
        ({"limit": null}). Drop it so the declared default (5, False, "Routine") applies. A null for a
        required field is still dropped, so it is reported as missing rather than accepted."""
        if isinstance(data, dict):
            return {
                key: value for key, value in data.items()
                if not (value is None and key in cls.model_fields and cls.model_fields[key].default is not None)
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
    limit: int = Field(DEFAULT_SLOTS, ge=1, le=MAX_SLOTS, description="max slots, default 5")
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
    # A what-if estimate when there is no patient record to use ("a patient on MERIDIAN-SILVER who has
    # met $500 of the deductible"). Ignored when patient_ref is given: the patient's own record is used.
    plan_code: str | None = Field(None, description="plan for an estimate without a patient_ref, e.g. MERIDIAN-SILVER")
    deductible_met_usd: float | None = Field(
        None, ge=0, description="deductible already met this year, for an estimate without a patient_ref"
    )


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
    # Bounded, and generic: the query goes to the embedding API (data-handling standard).
    query: str = Field(
        min_length=1, max_length=500,
        description="the policy or coverage question in plain, generic words; never a patient's name, "
        "date of birth, patient_ref or member number",
    )
    specialty: Specialty | None = Field(
        None, description="set only for a question about one specialty's service line handbook"
    )
    limit: int = Field(DEFAULT_PASSAGES, ge=1, le=MAX_PASSAGES, description="max passages, default 4")


@dataclass(frozen=True)
class ToolDefinition:
    name: str  # also the CareFlowTools method name
    description: str  # what the model reads when choosing a tool
    input_model: type[_Input]
    idempotent: bool  # True = a repeated call changes nothing more, so it may be retried
    backend: str = Backend.LOCAL.value  # a Backend value

    def build_openai_spec(self) -> dict:
        """The tool as an OpenAI-style function spec (the `tools` argument of a chat completion)."""
        schema = self.input_model.model_json_schema()
        properties = {key: _simplify_property(value) for key, value in schema.get("properties", {}).items()}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": schema.get("required", [])},
            },
        }


def _simplify_property(property_schema: dict) -> dict:
    """Pydantic's JSON Schema -> the plain subset every provider accepts (no titles, defaults or null unions)."""
    property_schema = {key: value for key, value in property_schema.items() if key not in ("title", "default")}
    if "anyOf" in property_schema:
        options = [option for option in property_schema.pop("anyOf") if option.get("type") != "null"]
        if len(options) == 1:
            property_schema = {**options[0], **property_schema}
        else:
            property_schema["anyOf"] = options
    return property_schema


TOOLS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        ToolName.CHECK_ELIGIBILITY.value,
        "Look up a patient's insurance plan and eligibility status. Use before booking or quoting costs.",
        CheckEligibilityInput,
        idempotent=True,
        backend=Backend.EHR.value,
    ),
    ToolDefinition(
        ToolName.FIND_APPOINTMENT_SLOTS.value,
        "Find open appointment slots for a specialty, soonest first. Pass patient_ref when known so only "
        "providers able to see this patient are shown.",
        FindAppointmentSlotsInput,
        idempotent=True,
        backend=Backend.EHR.value,
    ),
    ToolDefinition(
        ToolName.BOOK_APPOINTMENT.value,
        "Book an open slot for a patient. Active coverage books on insurance; coverage awaiting verification "
        "books and staff verify it. Otherwise (e.g. lapsed) it books only as self-pay, and only after the "
        "patient has agreed to pay themselves. New patients only with providers accepting new patients.",
        BookAppointmentInput,
        idempotent=False,
        backend=Backend.EHR.value,
    ),
    ToolDefinition(
        ToolName.CHECK_PREAUTH.value,
        "Check whether a procedure is covered under a plan, whether it needs pre-authorisation, and the "
        "approval window in business days. For a covered service it also says what the patient still pays: the "
        "flat co-pay for consultations, imaging and lab work, or that the deductible and coinsurance apply. "
        "Give procedure_code with plan_code or patient_ref, or a referral_id.",
        CheckPreauthInput,
        idempotent=True,
        backend=Backend.EHR.value,
    ),
    ToolDefinition(
        ToolName.CALCULATE_COPAY.value,
        "Work out what the patient owes for a procedure under their plan. If insurance cannot be used "
        "(no insurance, coverage not active, procedure not covered) it gives the self-pay price instead. "
        "Leave out patient_ref for a patient with no insurance. For someone with no patient_ref (a what-if "
        "question), pass plan_code and deductible_met_usd for an estimate.",
        CalculateCopayInput,
        idempotent=True,
    ),
    ToolDefinition(
        ToolName.FLAG_FOR_HUMAN.value,
        "Hand the case to clinic staff. MUST be used for any clinical question (symptoms, diagnosis, "
        "medication or dosing), anything urgent, and anything the other tools cannot settle.",
        FlagForHumanInput,
        idempotent=True,  # same patient and reason is recorded once
    ),
    ToolDefinition(
        ToolName.FIND_PATIENT.value,
        "Identify the patient you are talking to when you have no patient_ref: their full name plus their "
        "date of birth or member number. Returns only the patient_ref.",
        FindPatientInput,
        idempotent=True,
        backend=Backend.EHR.value,
    ),
    ToolDefinition(
        ToolName.SEARCH_POLICIES.value,
        "Search the clinic's policy and insurance documents (referral policy, pre-authorisation matrix, "
        "cost-share schedule, intake SOP, data handling, specialty handbooks) for rules, preparation "
        "instructions, timings and coverage terms. Returns passages with their document title, or none when "
        "the documents do not cover the question.",
        SearchPoliciesInput,
        idempotent=True,
        backend=Backend.POLICY_INDEX.value,
    ),
)
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
TOOL_SPECS = [tool.build_openai_spec() for tool in TOOLS]
