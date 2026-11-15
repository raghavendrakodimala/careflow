"""Offline tests for PatientRequest. Run from careflow_app/: python -m tests.test_schemas"""

import pytest
from pydantic import ValidationError

from careflow.schemas import PatientRequest


def test_valid_and_normalised():
    r = PatientRequest.model_validate(
        {
            "urgency": "Urgent",
            "specialty": "Cardiology",
            "insurance_plan": "Meridian Silver",
            "insurance_member_number": "mer 6486-7187",
            "seeks_clinical_advice": False,
        }
    )
    assert r.insurance_plan == "MERIDIAN-SILVER"
    assert r.insurance_member_number == "MER64867187"
    assert r.insurance_id == "MERIDIAN-SILVER MER64867187"
    assert r.urgent_flag is True
    assert r.missing_fields == []


def test_missing_fields_follow_extraction():
    r = PatientRequest(urgency="Routine", seeks_clinical_advice=False, missing_fields=["preferred_date"])
    assert r.missing_fields == ["insurance_id", "preferred_date", "specialty"]
    assert r.urgent_flag is False and r.insurance_id is None


def test_rejects_bad_enum_and_member_number():
    with pytest.raises(ValidationError):
        PatientRequest(urgency="Critical", seeks_clinical_advice=False)
    with pytest.raises(ValidationError):
        PatientRequest(urgency="Routine", seeks_clinical_advice=False, specialty="Podiatry")
    with pytest.raises(ValidationError):
        PatientRequest(urgency="Routine", seeks_clinical_advice=False, insurance_member_number="12345")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
