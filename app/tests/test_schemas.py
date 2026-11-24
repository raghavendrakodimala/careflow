"""Offline tests for PatientRequest. Run from app/: python -m tests.test_schemas"""

import pytest
from pydantic import ValidationError

from src.schemas import PatientRequest


def test_valid_and_normalised():
    request = PatientRequest.model_validate(
        {
            "urgency": "Urgent",
            "specialty": "Cardiology",
            "insurance_plan": "Meridian Silver",
            "insurance_member_number": "mer 6486-7187",
            "seeks_clinical_advice": False,
        }
    )
    assert request.insurance_plan == "MERIDIAN-SILVER"
    assert request.insurance_member_number == "MER64867187"
    assert request.insurance_id == "MERIDIAN-SILVER MER64867187"
    assert request.urgent_flag is True
    assert request.missing_fields == []


def test_missing_fields_follow_extraction():
    request = PatientRequest(urgency="Routine", seeks_clinical_advice=False, missing_fields=["preferred_date"])
    assert request.missing_fields == ["insurance_id", "preferred_date", "specialty"]
    assert request.urgent_flag is False and request.insurance_id is None


def test_rejects_bad_enum_and_member_number():
    with pytest.raises(ValidationError):
        PatientRequest(urgency="Critical", seeks_clinical_advice=False)
    with pytest.raises(ValidationError):
        PatientRequest(urgency="Routine", seeks_clinical_advice=False, specialty="Podiatry")
    with pytest.raises(ValidationError):
        PatientRequest(urgency="Routine", seeks_clinical_advice=False, insurance_member_number="12345")


def test_member_number_without_plan_still_gives_an_insurance_id():
    # insurance_id must not be None here while missing_fields does not list it.
    request = PatientRequest(urgency="Routine", seeks_clinical_advice=False, specialty="ENT", insurance_member_number="MER91530308")
    assert request.insurance_id == "MER91530308" and "insurance_id" not in request.missing_fields


def test_both_insurance_id_forms_resolve_in_the_eligibility_tool():
    # The combined "PLAN MEMBER" form must not break the handoff from the intake parse to the eligibility tool.
    from src.tools import CareFlowTools

    tools = CareFlowTools()
    patient = next(iter(tools.patients.values()))
    for with_plan in (True, False):
        request = PatientRequest(
            urgency="Routine",
            seeks_clinical_advice=False,
            insurance_plan=patient["plan_code"] if with_plan else None,
            insurance_member_number=patient["member_number"],
        )
        assert tools.check_eligibility(member_number=request.insurance_id)["patient_ref"] == patient["patient_ref"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
