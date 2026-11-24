"""Offline tests for the LangGraph foundation: graph shape, tool subsets, and checkpointed sessions."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import run_agent
from src.graph import AgentDependencies, build_agent_graph
from src.tools import CareFlowTools


def _make_call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _make_message(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls or [])


def _script_model(steps):
    seen = []

    def respond(messages, specs):
        seen.append({"messages": [dict(message) for message in messages], "specs": specs})
        return steps[len(seen) - 1]

    return respond, seen


def test_graph_shape_is_memory_model_tools_ground_escalate_record():
    graph = build_agent_graph(AgentDependencies(tools=CareFlowTools(), model=lambda messages, specs: None))
    drawn = graph.get_graph()
    assert {"memory", "model", "tools", "ground", "escalate", "record"} <= set(drawn.nodes)
    edges = {(edge.source, edge.target) for edge in drawn.edges}
    assert {
        ("__start__", "memory"), ("memory", "model"), ("model", "tools"), ("tools", "memory"),
        ("tools", "escalate"), ("model", "ground"), ("ground", "record"), ("escalate", "record"),
        ("record", "__end__"),
    } <= edges
    assert ("model", "record") not in edges  # no final answer skips the groundedness check


def test_only_the_allowed_tool_subset_is_offered_and_enforced():
    tools = CareFlowTools()
    ref = next(patient["patient_ref"] for patient in tools.patients.values() if patient["eligibility_status"] == "active")
    model, seen = _script_model(
        [_make_message(calls=[_make_call("1", "check_eligibility", patient_ref=ref), _make_call("2", "flag_for_human", reason="x")]), _make_message("done")]
    )
    graph = build_agent_graph(AgentDependencies(tools=tools, model=model), tool_names={"check_eligibility"})
    out = graph.invoke({"messages": [{"role": "user", "content": "hi"}], "steps": 0})
    assert [spec["function"]["name"] for spec in seen[0]["specs"]] == ["check_eligibility"]
    replies = {message["tool_call_id"]: json.loads(message["content"]) for message in out["messages"] if message["role"] == "tool"}
    assert replies["1"]["ok"] is True
    assert replies["2"]["ok"] is False and "not available" in replies["2"]["error"]
    assert tools.flags == []  # the forbidden tool never ran


def test_system_prompt_is_prepended_each_call_but_not_stored_in_state():
    model, seen = _script_model([_make_message("ok")])
    graph = build_agent_graph(AgentDependencies(tools=CareFlowTools(), model=model), system_prompt="SYS")
    out = graph.invoke({"messages": [{"role": "user", "content": "hi"}], "steps": 0})
    assert seen[0]["messages"][0] == {"role": "system", "content": "SYS"}
    assert all(message["role"] != "system" for message in out["messages"])


def test_escalation_records_a_clean_tool_trace_and_one_flag():
    tools = CareFlowTools()
    loop = _make_message(calls=[_make_call("l", "find_appointment_slots", specialty="ENT")])
    model, _ = _script_model([loop] * 2)
    result = run_agent("x", tools, max_steps=2, model=model, patient_ref="NOPE-IGNORED")
    assert result.stopped_early and result.tools_used == ["find_appointment_slots", "find_appointment_slots", "flag_for_human"]
    assert len(tools.flags) == 1


def test_escalation_goes_through_the_shared_tools_call_seam():
    # Tracing wraps tools.call; the step-cap handoff must pass through it too.
    tools = CareFlowTools()
    seen = []
    real_call = tools.call
    tools.call = lambda name, args, **keywords: seen.append(name) or real_call(name, args, **keywords)
    loop = _make_message(calls=[_make_call("l", "find_appointment_slots", specialty="ENT")])
    model, _ = _script_model([loop] * 2)
    result = run_agent("x", tools, max_steps=2, model=model)
    assert seen == ["find_appointment_slots", "find_appointment_slots", "flag_for_human"]
    trace = result.tool_calls[-1]
    assert trace["name"] == "flag_for_human" and json.loads(trace["arguments"])["urgency"] == "Routine"


def test_large_step_cap_does_not_hit_the_framework_recursion_limit():
    loop = _make_message(calls=[_make_call("l", "find_appointment_slots", specialty="ENT")])
    model, seen = _script_model([loop] * 30)
    result = run_agent("x", CareFlowTools(), max_steps=30, model=model)
    assert result.stopped_early and result.steps == 30 and len(seen) == 30


# ---- session memory foundation and workflow checkpoints --------------------------
def _run_two_turns(checkpointer, thread_a="a", thread_b="b"):
    tools = CareFlowTools()
    first_model, _ = _script_model([_make_message("first answer")])
    second_model, seen2 = _script_model([_make_message("second answer")])
    run_agent("first question", tools, model=first_model, checkpointer=checkpointer, thread_id=thread_a)
    result = run_agent("second question", tools, model=second_model, checkpointer=checkpointer, thread_id=thread_a)
    other_model, seen3 = _script_model([_make_message("other thread answer")])
    run_agent("unrelated", tools, model=other_model, checkpointer=checkpointer, thread_id=thread_b)
    return result, seen2[0]["messages"], seen3[0]["messages"]


def test_same_thread_remembers_earlier_turns_and_other_threads_do_not():
    result, second_turn_input, other_thread_input = _run_two_turns(MemorySaver())
    contents = [message["content"] for message in second_turn_input]
    assert "first question" in contents and "first answer" in contents and "second question" in contents
    assert all("first question" not in message["content"] for message in other_thread_input)  # privacy: no bleed across threads
    assert result.answer == "second answer"


def test_each_turn_reports_only_its_own_tool_calls_and_resets_the_step_count():
    tools = CareFlowTools()
    checkpointer = MemorySaver()
    lookup = _make_message(calls=[_make_call("t1", "find_appointment_slots", specialty="ENT")])
    first_model, _ = _script_model([lookup, _make_message("here are slots")])
    first_result = run_agent("slots?", tools, model=first_model, checkpointer=checkpointer, thread_id="t")
    second_model, _ = _script_model([_make_message("anything else?")])
    second_result = run_agent("thanks", tools, model=second_model, checkpointer=checkpointer, thread_id="t")
    assert first_result.tools_used == ["find_appointment_slots"] and first_result.steps == 2
    assert second_result.tools_used == [] and second_result.steps == 1  # not 3: steps restart each turn


def test_sqlite_checkpoint_survives_a_new_connection_and_new_graph(tmp_path):
    db_path = str(tmp_path / "cp.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    first_model, _ = _script_model([_make_message("remember me")])
    run_agent("my first message", CareFlowTools(), model=first_model, checkpointer=SqliteSaver(conn), thread_id="p1")
    conn.close()

    conn2 = sqlite3.connect(db_path, check_same_thread=False)
    second_model, seen = _script_model([_make_message("welcome back")])
    result = run_agent("am I still here?", CareFlowTools(), model=second_model, checkpointer=SqliteSaver(conn2), thread_id="p1")
    conn2.close()
    assert result.answer == "welcome back"
    assert next(message["content"] for message in seen[0]["messages"] if message["role"] == "user") == "my first message"
    assert "remember me" in [message["content"] for message in seen[0]["messages"]]


def test_a_failing_model_raises_and_does_not_poison_the_thread():
    from src.llm_client import LLMClientError

    checkpointer = MemorySaver()

    def raise_client_error(messages, specs):
        raise LLMClientError("all models failed")

    with pytest.raises(LLMClientError):
        run_agent("x", CareFlowTools(), model=raise_client_error, checkpointer=checkpointer, thread_id="t")
    model, _ = _script_model([_make_message("recovered")])
    assert run_agent("again", CareFlowTools(), model=model, checkpointer=checkpointer, thread_id="t").answer == "recovered"


# ---- long-term patient memory, identity, history window ----------------------------------
def _read_memory_notes(seen_call):
    return [message["content"] for message in seen_call["messages"] if message["role"] == "system" and message["content"].startswith("Patient memory")]


def _find_known_patient(tools):
    return tools.store.list_rows("patient_interactions")[0]["patient_ref"]


def test_memory_note_is_given_for_a_verified_patient_only():
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    first_model, seen1 = _script_model([_make_message("hello")])
    run_agent("hi", tools, model=first_model, patient_ref=ref)
    assert len(_read_memory_notes(seen1[0])) == 1 and ref in _read_memory_notes(seen1[0])[0]
    second_model, seen2 = _script_model([_make_message("hello")])
    run_agent("hi", tools, model=second_model)  # no patient: no memory
    assert _read_memory_notes(seen2[0]) == []


def test_find_patient_verifies_the_caller_and_loads_memory_for_the_thread():
    tools = CareFlowTools()
    patient = tools.patients[_find_known_patient(tools)]
    checkpointer = MemorySaver()
    ask = _make_message(calls=[_make_call("f", "find_patient", given_name=patient["given_name"], family_name=patient["family_name"],
                            birth_date=patient["birth_date"])])
    first_model, seen1 = _script_model([ask, _make_message("thanks, found you")])
    first_result = run_agent("I'm calling about my referral", tools, model=first_model, checkpointer=checkpointer, thread_id="t")
    assert first_result.patient_ref == patient["patient_ref"]
    assert _read_memory_notes(seen1[0]) == [] and len(_read_memory_notes(seen1[1])) == 1  # loaded right after the check
    second_model, seen2 = _script_model([_make_message("still you")])
    run_agent("and another thing", tools, model=second_model, checkpointer=checkpointer, thread_id="t")  # no patient_ref passed
    assert len(_read_memory_notes(seen2[0])) == 1


def test_failed_identity_check_loads_no_memory():
    tools = CareFlowTools()
    ask = _make_message(calls=[_make_call("f", "find_patient", given_name="No", family_name="Body", birth_date="2000-01-01")])
    model, seen = _script_model([ask, _make_message("sorry")])
    result = run_agent("it's me", tools, model=model)
    assert result.patient_ref is None and all(_read_memory_notes(seen_call) == [] for seen_call in seen)


def test_memory_is_rebuilt_before_each_model_call_so_a_new_booking_shows():
    tools = CareFlowTools()
    ref = next(patient_ref for patient_ref, patient in tools.patients.items() if patient["eligibility_status"] == "active" and patient_ref in tools.established)
    slot = tools.find_appointment_slots("ENT", patient_ref=ref)["slots"][0]["slot_id"]
    book = _make_message(calls=[_make_call("b", "book_appointment", patient_ref=ref, slot_id=slot)])
    model, seen = _script_model([book, _make_message("booked")])
    run_agent("book it", tools, model=model, patient_ref=ref)
    appt = tools.store.list_rows_for_patient("appointments", ref)[-1]["appointment_id"]
    assert appt not in _read_memory_notes(seen[0])[0] and appt in _read_memory_notes(seen[1])[0]


def test_a_conversation_is_recorded_as_one_contact_for_the_patient():
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    before = len(tools.store.list_rows_for_patient("patient_interactions", ref))
    checkpointer = MemorySaver()
    model, _ = _script_model([_make_message(calls=[_make_call("e", "check_eligibility", patient_ref=ref)]), _make_message("you're covered")])
    run_agent("am I covered?", tools, model=model, patient_ref=ref, channel="sms", checkpointer=checkpointer, thread_id="c1")
    rows = tools.store.list_rows_for_patient("patient_interactions", ref)
    assert len(rows) == before + 1 and rows[-1]["topic"] == "eligibility_check" and rows[-1]["channel"] == "sms"
    second_model, seen2 = _script_model([_make_message("you're welcome")])
    run_agent("thanks", tools, model=second_model, checkpointer=checkpointer, thread_id="c1")  # same call: same contact
    assert len(tools.store.list_rows_for_patient("patient_interactions", ref)) == before + 1
    assert "eligibility_check" not in _find_repeat_line(seen2[0])  # the call does not count as its own REPEAT
    third_model, seen3 = _script_model([_make_message("hello again")])
    run_agent("hi", tools, model=third_model, patient_ref=ref)  # a new contact, answered without tools: still recorded
    assert len(tools.store.list_rows_for_patient("patient_interactions", ref)) == before + 2
    assert "eligibility_check" in _find_repeat_line(seen3[0])  # ...and the earlier call now counts


def _find_repeat_line(seen_call):
    lines = _read_memory_notes(seen_call)[0].splitlines()
    return next((line for line in lines if line.startswith("- Contacted within the last 24 hours")), "")


def test_step_cap_escalation_is_recorded_as_administrative_not_clinical():
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    loop = _make_message(calls=[_make_call("l", "find_appointment_slots", specialty="Podiatry")])  # fails every time
    model, _ = _script_model([loop] * 2)
    run_agent("x", tools, max_steps=2, model=model, patient_ref=ref)
    row = tools.store.list_rows_for_patient("patient_interactions", ref)[-1]
    assert row["topic"] == "general_enquiry" and row["outcome"] == "callback_scheduled"
    assert tools.flags[-1]["category"] == "administrative"


# ---- a verified conversation only touches its own patient's records -------------
def _find_other_patient(tools, ref):
    return next(patient_ref for patient_ref in tools.patients if patient_ref != ref)


@pytest.mark.parametrize("arg", ["patient_ref", "member_number", "referral_id"])
def test_tools_for_another_patient_are_blocked_on_a_verified_thread(arg):
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    other = next(referral["patient_ref"] for referral in tools.store.list_rows("referrals") if referral["patient_ref"] != ref)
    value = {
        "patient_ref": other.lower(),
        "member_number": tools.patients[other]["member_number"],
        "referral_id": next(referral["referral_id"] for referral in tools.store.list_rows("referrals") if referral["patient_ref"] == other),
    }[arg]
    name = "check_preauth" if arg == "referral_id" else "check_eligibility"
    model, _ = _script_model([_make_message(calls=[_make_call("x", name, **{arg: value})]), _make_message("sorry")])
    result = run_agent("check my brother", tools, model=model, patient_ref=ref)
    assert result.tool_calls[0]["result"]["ok"] is False and "only that patient's records" in result.tool_calls[0]["result"]["error"]
    assert other not in json.dumps(result.tool_calls[0]["result"])


def test_own_records_and_unbound_threads_are_not_blocked():
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    model, _ = _script_model([_make_message(calls=[_make_call("x", "check_eligibility", patient_ref=ref.lower())]), _make_message("ok")])
    assert run_agent("me", tools, model=model, patient_ref=f"  {ref.lower()} ").tool_calls[0]["result"]["ok"]
    model, _ = _script_model([_make_message(calls=[_make_call("x", "check_eligibility", patient_ref=_find_other_patient(tools, ref))]), _make_message("ok")])
    assert run_agent("staff lookup", tools, model=model).tool_calls[0]["result"]["ok"]  # no verified patient


def test_find_patient_cannot_switch_a_verified_thread_to_someone_else():
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    other_patient = tools.patients[_find_other_patient(tools, ref)]
    ask = _make_message(calls=[_make_call("f", "find_patient", given_name=other_patient["given_name"], family_name=other_patient["family_name"],
                            birth_date=other_patient["birth_date"])])
    model, _ = _script_model([ask, _make_message("sorry")])
    result = run_agent("now me as my brother", tools, model=model, patient_ref=ref)
    assert result.patient_ref == ref and result.tool_calls[0]["result"]["ok"] is False


def test_caller_patient_ref_is_canonicalised():
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    model, seen = _script_model([_make_message("hi")])
    result = run_agent("hi", tools, model=model, patient_ref=f" {ref.lower()} ")
    assert result.patient_ref == ref and len(_read_memory_notes(seen[0])) == 1


def test_open_session_checkpointer_uses_wal(tmp_path):
    from src.agent import open_session_checkpointer

    checkpointer = open_session_checkpointer(tmp_path / "s.db")
    assert checkpointer.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert checkpointer.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    checkpointer.conn.close()


def test_memory_can_be_switched_off():
    tools = CareFlowTools()
    ref = _find_known_patient(tools)
    before = len(tools.store.list_rows("patient_interactions"))
    model, seen = _script_model([_make_message(calls=[_make_call("e", "check_eligibility", patient_ref=ref)]), _make_message("ok")])
    run_agent("covered?", dependencies=AgentDependencies(tools=tools, model=model, memory=None), patient_ref=ref)
    assert _read_memory_notes(seen[0]) == [] and len(tools.store.list_rows("patient_interactions")) == before


def test_model_sees_a_rolling_window_but_the_checkpoint_keeps_everything():
    tools = CareFlowTools()
    checkpointer = MemorySaver()
    seen = []

    def record_messages(messages, specs):
        seen.append(messages)
        return _make_message(f"a{len(seen)}")

    graph = build_agent_graph(AgentDependencies(tools=tools, model=record_messages, memory=None), checkpointer=checkpointer, history_turns=2)
    config = {"configurable": {"thread_id": "w"}}
    for turn in range(4):
        graph.invoke({"messages": [{"role": "user", "content": f"q{turn}"}], "steps": 0}, config)
    users = [message["content"] for message in seen[-1] if message["role"] == "user"]
    assert users == ["q2", "q3"]
    assert [message["content"] for message in graph.get_state(config).values["messages"] if message["role"] == "user"] == ["q0", "q1", "q2", "q3"]


def test_policy_search_is_not_offered_without_an_index():
    model, seen = _script_model([_make_message("ok")])
    run_agent("hi", CareFlowTools(), model=model)
    names = {spec["function"]["name"] for spec in seen[0]["specs"]}
    assert "find_patient" in names and "search_policies" not in names


# ---- repeat markers on staff flags are applied from the records -------------
def _find_fresh_patient(tools):
    seen = {row["patient_ref"] for row in tools.store.list_rows("patient_interactions")}
    return next(patient_ref for patient_ref in tools.patients if patient_ref not in seen)


def _run_flagging_turn(tools, ref, **flag_args):
    model, _ = _script_model([_make_message(calls=[_make_call("f", "flag_for_human", **flag_args)]), _make_message("done")])
    result = run_agent("question", tools, model=model, patient_ref=ref)
    return tools.flags[-1], result


def test_clinical_flag_after_a_clinical_contact_today_is_marked_repeat():
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    tools.store.add_interaction(ref, "2026-08-01T08:00", "sms", "clinical_question", "escalated_to_clinician", "x")
    flag, _ = _run_flagging_turn(tools, ref, reason="Asked which cream to use", urgency="Routine", clinical=True)
    assert flag["reason"] == "REPEAT: Asked which cream to use" and flag["urgency"] == "Routine"


def test_third_clinical_contact_in_72_hours_is_repeat_priority_with_raised_urgency():
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    for when in ("2026-07-29T15:00", "2026-08-01T08:00"):
        tools.store.add_interaction(ref, when, "sms", "clinical_question", "escalated_to_clinician", "x")
    flag, _ = _run_flagging_turn(tools, ref, reason="REPEAT - asked again", urgency="Routine", clinical=True)
    assert flag["reason"] == "Repeat - Priority: asked again" and flag["urgency"] == "Priority"  # not doubled


def test_no_marker_without_same_issue_history_or_a_known_issue():
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    tools.store.add_interaction(ref, "2026-08-01T08:00", "sms", "billing_question", "resolved", "x")
    flag, _ = _run_flagging_turn(tools, ref, reason="Clinical question", clinical=True)  # different issue
    assert flag["reason"] == "Clinical question"
    flag, _ = _run_flagging_turn(tools, ref, reason="Please call me back")  # administrative, no issue known
    assert flag["reason"] == "Please call me back"


def test_administrative_flag_takes_the_issue_from_this_turns_tools():
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    tools.store.add_interaction(ref, "2026-08-01T08:00", "sms", "eligibility_check", "resolved", "x")
    model, _ = _script_model([
        _make_message(calls=[_make_call("e", "check_eligibility", patient_ref=ref)]),
        _make_message(calls=[_make_call("f", "flag_for_human", reason="Help with coverage")]),
        _make_message("done"),
    ])
    run_agent("coverage again", tools, model=model, patient_ref=ref)
    assert tools.flags[-1]["reason"] == "REPEAT: Help with coverage"


# ---- flags are not de-duplicated across contacts -------
def test_second_clinical_contact_gets_its_own_marked_ticket():
    # The model writes the same reason on both contacts and does not add REPEAT. If the store
    # returned the first contact's flag (duplicate=True), the second contact would have no ticket at all.
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    reason = "Patient asked which cream to use for a rash and how often."
    for _ in range(2):
        _run_flagging_turn(tools, ref, reason=reason, urgency="Routine", clinical=True)
    flags = [flag for flag in tools.flags if flag["patient_ref"] == ref]
    assert [flag["reason"] for flag in flags] == [reason, f"REPEAT: {reason}"]
    assert all(flag["category"] == "clinical" for flag in flags) and flags[0]["contact_id"] != flags[1]["contact_id"]


def test_same_flag_within_one_contact_is_still_recorded_once():
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    twice = _make_message(calls=[_make_call("a", "flag_for_human", reason="Call me back"), _make_call("b", "flag_for_human", reason="Call me back")])
    model, _ = _script_model([twice, _make_message("ok")])
    result = run_agent("x", tools, model=model, patient_ref=ref)
    assert [call["result"]["duplicate"] for call in result.tool_calls] == [False, True]
    assert len([flag for flag in tools.flags if flag["patient_ref"] == ref]) == 1


def test_a_conversation_keeps_one_contact_id_across_turns():
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    checkpointer = MemorySaver()
    for _ in range(2):
        model, _ = _script_model([_make_message(calls=[_make_call("a", "flag_for_human", reason="Call me back")]), _make_message("ok")])
        run_agent("x", tools, model=model, patient_ref=ref, checkpointer=checkpointer, thread_id="call-7")
    flags = [flag for flag in tools.flags if flag["patient_ref"] == ref]
    assert len(flags) == 1 and flags[0]["contact_id"] == "call-7"


def test_the_model_cannot_set_the_contact_id():
    tools = CareFlowTools()
    result = tools.call("flag_for_human", {"reason": "x", "contact_id": "forged"})
    assert result["ok"] is False and "contact_id" in result["error"]
    assert tools.call("flag_for_human", {"reason": "x"}, context={"contact_id": "c1", "evil": 1})["contact_id"] == "c1"


def test_flag_on_a_verified_conversation_is_attributed_to_the_patient():
    tools = CareFlowTools()
    ref = _find_fresh_patient(tools)
    _run_flagging_turn(tools, ref, reason="Please call me back")  # the model left out patient_ref
    assert tools.flags[-1]["patient_ref"] == ref
    model, _ = _script_model([_make_message(calls=[_make_call("f", "flag_for_human", reason="Anonymous question")]), _make_message("ok")])
    run_agent("x", tools, model=model)  # no verified patient: nothing to attribute
    assert tools.flags[-1]["patient_ref"] is None
