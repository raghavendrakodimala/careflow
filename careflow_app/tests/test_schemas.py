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


def test_member_number_without_plan_still_gives_an_insurance_id():
    # Audit B2: insurance_id used to be None here while missing_fields did not list it.
    r = PatientRequest(urgency="Routine", seeks_clinical_advice=False, specialty="ENT", insurance_member_number="MER91530308")
    assert r.insurance_id == "MER91530308" and "insurance_id" not in r.missing_fields


def test_both_insurance_id_forms_resolve_in_the_m2_eligibility_tool():
    # Audit 5.2.4: the combined "PLAN MEMBER" form must not break the M1 -> M2 handoff.
    from careflow.tools import CareFlowTools

    t = CareFlowTools()
    p = next(iter(t.patients.values()))
    for with_plan in (True, False):
        r = PatientRequest(
            urgency="Routine",
            seeks_clinical_advice=False,
            insurance_plan=p["plan_code"] if with_plan else None,
            insurance_member_number=p["member_number"],
        )
        assert t.check_eligibility(member_number=r.insurance_id)["patient_ref"] == p["patient_ref"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
