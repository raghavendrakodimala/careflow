"""Offline tests for check_preauth, against preauth_rules, referrals and referral_details."""

import pytest

from src.tools import TOOL_SPECS, CareFlowTools


@pytest.fixture()
def tools():
    return CareFlowTools()


def test_preauth_required_with_approval_window(tools):
    result = tools.check_preauth("ORTH-MRI", plan_code="MERIDIAN-BRONZE")
    assert result["ok"] and result["covered"] and result["preauth_required"]
    assert result["approval_window_business_days"] == 5 and "5 business days" in result["note"]


def test_preauth_not_needed_has_no_window(tools):
    result = tools.check_preauth("CARD-CONS", plan_code="MERIDIAN-GOLD")
    assert result["ok"] and result["preauth_required"] is False and result["approval_window_business_days"] is None


def test_not_covered_procedure_says_so_with_list_price(tools):
    result = tools.check_preauth("CARD-CATH", plan_code="CIVIC-BASE")
    assert result["covered"] is False and "$7,800" in result["note"]  # cardiology handbook clause 6.4


def test_plan_comes_from_the_patient_when_not_given(tools):
    patient = next(candidate for candidate in tools.patients.values() if candidate["plan_code"] == "MERIDIAN-SILVER")
    result = tools.check_preauth("gast-colo", patient_ref=patient["patient_ref"].lower())
    assert result["plan_code"] == "MERIDIAN-SILVER" and result["approval_window_business_days"] == 3


def test_referral_lookup_joins_referrals_to_find_the_plan(tools):
    # referral_details has no patient or plan column; the plan must come via referrals -> patients.
    referral_id = "MHP-R-00000"
    details, ref_row = tools.referral_details[referral_id], tools.referrals[referral_id]
    result = tools.check_preauth(referral_id=referral_id.lower())
    assert result["ok"] and result["procedure_code"] == details["procedure_code"]
    assert result["plan_code"] == tools.patients[ref_row["patient_ref"]]["plan_code"]
    assert result["referral"]["preauth_status"] == details["preauth_status"]
    assert result["referral"]["sla_due_on"] == details["sla_due_on"]
    assert "patient_ref" not in result["referral"]


def test_referral_reads_are_live_not_cached_at_startup(tools):
    # Referrals can be updated from another node or process, so each read must go to the store.
    referral_id = "MHP-R-00000"
    with tools.store._begin_transaction() as connection:
        connection.execute("UPDATE referral_details SET preauth_status='denied' WHERE referral_id=?", (referral_id,))
    assert tools.check_preauth(referral_id=referral_id)["referral"]["preauth_status"] == "denied"
    assert tools.referral_details[referral_id]["preauth_status"] == "denied"


def test_bad_inputs_are_errors_not_exceptions(tools):
    assert tools.check_preauth()["ok"] is False
    assert tools.check_preauth("ORTH-MRI")["ok"] is False  # no plan source
    assert "valid_codes" in tools.check_preauth("XXXX", plan_code="MERIDIAN-GOLD")
    assert "valid_plans" in tools.check_preauth("ORTH-MRI", plan_code="PLATINUM")
    assert tools.check_preauth(referral_id="MHP-R-99999")["ok"] is False
    assert tools.check_preauth("ORTH-MRI", patient_ref="NOPE")["ok"] is False


def test_tool_is_registered_and_described_to_the_model(tools):
    assert tools.call("check_preauth", {"procedure_code": "ORTH-MRI", "plan_code": "MERIDIAN-BRONZE"})["ok"]
    assert "check_preauth" in {spec["function"]["name"] for spec in TOOL_SPECS}


# ---- what a covered service still costs -----------------------------------------------------
def test_covered_imaging_reports_the_flat_copay_and_cites_the_schedule():
    tools = CareFlowTools()
    for code in ("ORTH-MRI", "NEUR-MRI"):
        result = tools.call("check_preauth", {"procedure_code": code, "plan_code": "MERIDIAN-BRONZE"})
        assert (result["covered"], result["preauth_required"], result["approval_window_business_days"]) == (True, True, 5)
        assert (result["service_category"], result["flat_copay_usd"]) == ("imaging", 160)
        assert "$160" in result["cost_share"] and "not covered at 100 percent" in result["cost_share"]
        assert result["source"] == ["Co-payment and Cost-Share Schedule", "Pre-Authorisation Requirement Matrix"]


def test_covered_procedure_points_to_the_calculator_and_gives_no_amount():
    result = CareFlowTools().call("check_preauth", {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER"})
    assert result["service_category"] == "procedure" and "flat_copay_usd" not in result
    assert "calculate_copay" in result["cost_share"] and result["source"] == ["Pre-Authorisation Requirement Matrix"]


def test_no_cost_share_for_an_uncovered_service_or_a_patient_without_active_cover():
    tools = CareFlowTools()
    result = tools.call("check_preauth", {"procedure_code": "ORTH-ARTH", "plan_code": "CIVIC-BASE"})
    assert result["covered"] is False and "cost_share" not in result and "flat_copay_usd" not in result
    for patient in tools.patients.values():
        if patient["eligibility_status"] != "active":
            result = tools.call("check_preauth", {"procedure_code": "ORTH-MRI", "patient_ref": patient["patient_ref"]})
            if result.get("ok") and result["covered"]:
                assert "flat_copay_usd" not in result and "cost_share" not in result
                break
    else:
        raise AssertionError("no covered check for a patient without active cover in this dataset")
