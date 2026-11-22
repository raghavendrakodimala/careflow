"""M6: the EHR MCP server. The server runs as a real child process over stdio, on a database in tmp_path;
each test compares what it returns with the direct function, so the two can never drift apart."""

import pytest

from careflow.mcp import EHR_TOOLS, McpClient, McpRoutedTools, run_tool, tool_list
from careflow.store import RuntimeStore
from careflow.tool_contracts import TOOLS_BY_NAME
from careflow.tools import CareFlowTools


@pytest.fixture(scope="module")
def ehr(tmp_path_factory):
    """(routed tools, local tools) sharing one runtime database, with the server started once."""
    db = tmp_path_factory.mktemp("mcp") / "runtime.db"
    local = CareFlowTools(store=RuntimeStore(db))
    routed = McpRoutedTools(local)
    yield routed, local
    routed.close()
    local.store.close()


def _active(tools):
    est = tools.established
    return next(ref for ref, p in tools.patients.items() if p["eligibility_status"] == "active" and ref in est)


def test_server_lists_exactly_the_ehr_tools_with_the_shared_schemas(ehr):
    routed, _ = ehr
    listed = {t["name"]: t for t in routed.client.tools}
    assert set(listed) == set(EHR_TOOLS)
    for name, tool in listed.items():
        definition = TOOLS_BY_NAME[name]
        assert tool["description"] == definition.description
        assert tool["input_schema"] == definition.openai_spec()["function"]["parameters"]
    # Staff flags, the co-pay calculator and policy search are not EHR tools and stay in-process.
    assert not {"flag_for_human", "calculate_copay", "search_policies"} & set(listed)


def test_tool_list_is_built_from_the_tool_definitions():
    assert [t.name for t in tool_list()] == list(EHR_TOOLS)


@pytest.mark.parametrize("name, arguments", [
    ("check_eligibility", lambda ref: {"patient_ref": ref}),
    ("find_appointment_slots", lambda ref: {"specialty": "Dermatology", "patient_ref": ref, "limit": 3}),
    ("check_preauth", lambda ref: {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER"}),
    ("check_preauth", lambda ref: {"procedure_code": "GAST-COLO", "patient_ref": ref}),
    ("find_patient", lambda ref: {"given_name": "Nobody", "family_name": "Here", "member_number": "MER00000000"}),
    ("check_eligibility", lambda ref: {"patient_ref": "MHP-P-99999"}),
])
def test_results_match_the_direct_function(ehr, name, arguments):
    routed, local = ehr
    args = arguments(_active(local))
    assert routed.call(name, args) == local.call(name, args)
    assert routed.remote_calls[-1] == name


def test_bad_arguments_give_a_structured_error_not_a_crash(ehr):
    routed, local = ehr
    for args in ({"nope": 1}, {"patient_ref": 12345}, {}):
        got = routed.call("check_eligibility", args)
        assert got["ok"] is False and got == local.call("check_eligibility", args)
    assert routed.call("check_eligibility", "{not json")["ok"] is False  # malformed JSON never reaches the server
    assert routed.call("check_eligibility", {"patient_ref": _active(local)})["ok"] is True  # still serving


def test_unknown_tool_is_refused_by_the_server():
    tools = CareFlowTools()
    got = run_tool(tools, "flag_for_human", {"reason": "x"})
    assert got["ok"] is False and "unknown tool" in got["error"] and tools.flags == []


def test_booking_through_mcp_is_seen_by_every_process(ehr):
    routed, local = ehr
    ref = _active(local)
    slot = routed.call("find_appointment_slots", {"specialty": "ENT", "patient_ref": ref})["slots"][0]["slot_id"]
    booked = routed.call("book_appointment", {"patient_ref": ref, "slot_id": slot})
    assert booked["ok"] and booked["slot_id"] == slot
    # The main process's own connection, and a brand new one, both see it.
    assert local.booked[slot] == ref
    other = RuntimeStore(local.store.path)
    assert other.one("appointment_slots", "slot_id", slot)["status"] == "booked"
    other.close()
    # The slot cannot be booked twice, whichever side asks.
    assert routed.call("book_appointment", {"patient_ref": ref, "slot_id": slot})["ok"] is False
    assert local.call("book_appointment", {"patient_ref": ref, "slot_id": slot})["ok"] is False


def test_other_tools_stay_local(ehr):
    routed, local = ehr
    before = len(routed.remote_calls)
    flag = routed.call("flag_for_human", {"reason": "Needs a callback"}, context={"contact_id": "t1"})
    assert flag["ok"] and flag["contact_id"] == "t1" and len(routed.remote_calls) == before
    assert routed.available_tools() == local.available_tools() and routed.store is local.store


def test_mcp_needs_a_file_backed_store():
    with pytest.raises(ValueError, match="file-backed"):
        McpRoutedTools(CareFlowTools())


def test_a_stopped_server_raises_a_clear_error(tmp_path):
    from careflow.mcp import McpUnavailable

    client = McpClient(tmp_path / "runtime.db")
    assert client.call("check_eligibility", {"patient_ref": "MHP-P-00001"})["ok"] is True
    client.close()
    with pytest.raises(McpUnavailable):
        client.call("check_eligibility", {"patient_ref": "MHP-P-00001"})
