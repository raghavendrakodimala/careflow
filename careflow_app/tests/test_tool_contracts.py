"""Tool contracts: one Pydantic input model per tool generates the spec and validates the model's arguments."""

import inspect

from careflow.tool_contracts import TOOL_SPECS, TOOLS
from careflow.tools import CareFlowTools


def test_every_contract_matches_a_tool_method_signature():
    # The drift guard: every argument the model is offered must exist on the method, and every required
    # method parameter must be required in the contract.
    for tool in TOOLS:
        params = inspect.signature(getattr(CareFlowTools, tool.name)).parameters
        offered = set(tool.input_model.model_fields)
        assert offered <= set(params) - {"self"}, tool.name
        required_by_method = {n for n, p in params.items() if n != "self" and p.default is inspect.Parameter.empty}
        required_by_contract = {n for n, f in tool.input_model.model_fields.items() if f.is_required()}
        assert required_by_method <= required_by_contract, tool.name


def test_generated_specs_use_only_the_plain_schema_subset():
    for spec in TOOL_SPECS:
        params = spec["function"]["parameters"]
        assert params["type"] == "object" and set(params["required"]) <= set(params["properties"])
        for prop in params["properties"].values():
            assert not {"title", "default", "anyOf", "$ref"} & set(prop), (spec["function"]["name"], prop)
            assert "type" in prop


def test_enums_come_from_the_shared_types():
    specs = {s["function"]["name"]: s["function"]["parameters"]["properties"] for s in TOOL_SPECS}
    assert specs["flag_for_human"]["urgency"]["enum"] == ["Routine", "Priority", "Urgent", "Emergent"]
    assert specs["check_preauth"]["plan_code"]["enum"] == ["MERIDIAN-GOLD", "MERIDIAN-SILVER", "MERIDIAN-BRONZE", "CIVIC-BASE"]
    assert specs["find_appointment_slots"]["mode"]["enum"] == ["in_person", "telehealth"]


def test_only_booking_is_not_idempotent():
    assert {t.name for t in TOOLS if not t.idempotent} == {"book_appointment"}


def test_explicit_null_for_an_optional_argument_means_use_the_default():
    # Audit §8.3 #1: models often send {"limit": null} for an argument they mean to omit.
    t = CareFlowTools()
    r = t.call("find_appointment_slots", {"specialty": "ENT", "limit": None, "mode": None, "new_patient": None})
    assert r["ok"] and r["count"] == 5
    flag = t.call("flag_for_human", {"reason": "x", "urgency": None, "patient_ref": None})
    assert flag["ok"] and flag["urgency"] == "Routine"
    ref = next(iter(t.patients))
    assert t.call("calculate_copay", {"patient_ref": ref, "procedure_code": "CARD-CONS", "telehealth": None})["ok"]
    missing = t.call("book_appointment", {"patient_ref": None, "slot_id": "MHP-SL-00001"})
    assert missing["ok"] is False and "patient_ref" in missing["error"]  # required stays required


def test_call_validates_against_the_contract_before_running_the_tool():
    t = CareFlowTools()
    r = t.call("flag_for_human", {"reason": "x", "urgency": "Critical"})
    assert r["ok"] is False and "urgency" in r["error"] and t.flags == []
    assert "limit" in t.call("find_appointment_slots", {"specialty": "ENT", "limit": "lots"})["error"]
    assert t.call("find_appointment_slots", {"specialty": "ENT", "limit": "3"})["count"] == 3  # lax coercion
    assert "extra" in t.call("check_eligibility", {"patient": "x"})["error"].lower()
