"""Offline tests for check_preauth (audit A1), against preauth_rules, referrals and referral_details."""

import pytest

from careflow.tools import TOOL_SPECS, CareFlowTools


@pytest.fixture()
def t():
    return CareFlowTools()


def test_preauth_required_with_approval_window(t):
    r = t.check_preauth("ORTH-MRI", plan_code="MERIDIAN-BRONZE")
    assert r["ok"] and r["covered"] and r["preauth_required"]
    assert r["approval_window_business_days"] == 5 and "5 business days" in r["note"]


def test_preauth_not_needed_has_no_window(t):
    r = t.check_preauth("CARD-CONS", plan_code="MERIDIAN-GOLD")
    assert r["ok"] and r["preauth_required"] is False and r["approval_window_business_days"] is None


def test_not_covered_procedure_says_so_with_list_price(t):
    r = t.check_preauth("CARD-CATH", plan_code="CIVIC-BASE")
    assert r["covered"] is False and "$7,800" in r["note"]  # cardiology handbook clause 6.4


def test_plan_comes_from_the_patient_when_not_given(t):
    p = next(p for p in t.patients.values() if p["plan_code"] == "MERIDIAN-SILVER")
    r = t.check_preauth("gast-colo", patient_ref=p["patient_ref"].lower())
    assert r["plan_code"] == "MERIDIAN-SILVER" and r["approval_window_business_days"] == 3


def test_referral_lookup_joins_referrals_to_find_the_plan(t):
    # referral_details has no patient or plan column; the plan must come via referrals -> patients.
    rid = "MHP-R-00000"
    details, ref_row = t.referral_details[rid], t.referrals[rid]
    r = t.check_preauth(referral_id=rid.lower())
    assert r["ok"] and r["procedure_code"] == details["procedure_code"]
    assert r["plan_code"] == t.patients[ref_row["patient_ref"]]["plan_code"]
    assert r["referral"]["preauth_status"] == details["preauth_status"]
    assert r["referral"]["sla_due_on"] == details["sla_due_on"]
    assert "patient_ref" not in r["referral"]


def test_referral_reads_are_live_not_cached_at_startup(t):
    # Audit §8.3 #2: M5/M6 will update referrals from another node or process.
    rid = "MHP-R-00000"
    with t.store._tx() as c:
        c.execute("UPDATE referral_details SET preauth_status='denied' WHERE referral_id=?", (rid,))
    assert t.check_preauth(referral_id=rid)["referral"]["preauth_status"] == "denied"
    assert t.referral_details[rid]["preauth_status"] == "denied"


def test_bad_inputs_are_errors_not_exceptions(t):
    assert t.check_preauth()["ok"] is False
    assert t.check_preauth("ORTH-MRI")["ok"] is False  # no plan source
    assert "valid_codes" in t.check_preauth("XXXX", plan_code="MERIDIAN-GOLD")
    assert "valid_plans" in t.check_preauth("ORTH-MRI", plan_code="PLATINUM")
    assert t.check_preauth(referral_id="MHP-R-99999")["ok"] is False
    assert t.check_preauth("ORTH-MRI", patient_ref="NOPE")["ok"] is False


def test_tool_is_registered_and_described_to_the_model(t):
    assert t.call("check_preauth", {"procedure_code": "ORTH-MRI", "plan_code": "MERIDIAN-BRONZE"})["ok"]
    assert "check_preauth" in {s["function"]["name"] for s in TOOL_SPECS}


# ---- what a covered service still costs (golden case EV-904) --------------------------------
def test_covered_imaging_reports_the_flat_copay_and_cites_the_schedule():
    t = CareFlowTools()
    for code in ("ORTH-MRI", "NEUR-MRI"):
        r = t.call("check_preauth", {"procedure_code": code, "plan_code": "MERIDIAN-BRONZE"})
        assert (r["covered"], r["preauth_required"], r["approval_window_business_days"]) == (True, True, 5)
        assert (r["service_category"], r["flat_copay_usd"]) == ("imaging", 160)
        assert "$160" in r["cost_share"] and "not covered at 100 percent" in r["cost_share"]
        assert r["source"] == ["Co-payment and Cost-Share Schedule", "Pre-Authorisation Requirement Matrix"]


def test_covered_procedure_points_to_the_calculator_and_gives_no_amount():
    r = CareFlowTools().call("check_preauth", {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER"})
    assert r["service_category"] == "procedure" and "flat_copay_usd" not in r
    assert "calculate_copay" in r["cost_share"] and r["source"] == ["Pre-Authorisation Requirement Matrix"]


def test_no_cost_share_for_an_uncovered_service_or_a_patient_without_active_cover():
    t = CareFlowTools()
    r = t.call("check_preauth", {"procedure_code": "ORTH-ARTH", "plan_code": "CIVIC-BASE"})
    assert r["covered"] is False and "cost_share" not in r and "flat_copay_usd" not in r
    for p in t.patients.values():
        if p["eligibility_status"] != "active":
            r = t.call("check_preauth", {"procedure_code": "ORTH-MRI", "patient_ref": p["patient_ref"]})
            if r.get("ok") and r["covered"]:
                assert "flat_copay_usd" not in r and "cost_share" not in r
                break
    else:
        raise AssertionError("no covered check for a patient without active cover in this dataset")
