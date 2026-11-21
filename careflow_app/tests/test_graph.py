"""Offline tests for the LangGraph foundation: graph shape, tool subsets, and checkpointed sessions."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from careflow.agent import run_agent
from careflow.graph import Deps, build_agent_graph
from careflow.tools import CareFlowTools


def _call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _msg(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls or [])


def _model(steps):
    seen = []

    def fake(messages, specs):
        seen.append({"messages": [dict(m) for m in messages], "specs": specs})
        return steps[len(seen) - 1]

    return fake, seen


def test_graph_shape_is_memory_model_tools_ground_escalate_record():
    g = build_agent_graph(Deps(tools=CareFlowTools(), model=lambda m, s: None))
    drawn = g.get_graph()
    assert {"memory", "model", "tools", "ground", "escalate", "record"} <= set(drawn.nodes)
    edges = {(e.source, e.target) for e in drawn.edges}
    assert {
        ("__start__", "memory"), ("memory", "model"), ("model", "tools"), ("tools", "memory"),
        ("tools", "escalate"), ("model", "ground"), ("ground", "record"), ("escalate", "record"),
        ("record", "__end__"),
    } <= edges
    assert ("model", "record") not in edges  # M4: no final answer skips the groundedness check


def test_only_the_allowed_tool_subset_is_offered_and_enforced():
    tools = CareFlowTools()
    ref = next(p["patient_ref"] for p in tools.patients.values() if p["eligibility_status"] == "active")
    model, seen = _model(
        [_msg(calls=[_call("1", "check_eligibility", patient_ref=ref), _call("2", "flag_for_human", reason="x")]), _msg("done")]
    )
    g = build_agent_graph(Deps(tools=tools, model=model), tool_names={"check_eligibility"})
    out = g.invoke({"messages": [{"role": "user", "content": "hi"}], "steps": 0})
    assert [s["function"]["name"] for s in seen[0]["specs"]] == ["check_eligibility"]
    replies = {m["tool_call_id"]: json.loads(m["content"]) for m in out["messages"] if m["role"] == "tool"}
    assert replies["1"]["ok"] is True
    assert replies["2"]["ok"] is False and "not available" in replies["2"]["error"]
    assert tools.flags == []  # the forbidden tool never ran


def test_system_prompt_is_prepended_each_call_but_not_stored_in_state():
    model, seen = _model([_msg("ok")])
    g = build_agent_graph(Deps(tools=CareFlowTools(), model=model), system_prompt="SYS")
    out = g.invoke({"messages": [{"role": "user", "content": "hi"}], "steps": 0})
    assert seen[0]["messages"][0] == {"role": "system", "content": "SYS"}
    assert all(m["role"] != "system" for m in out["messages"])


def test_escalation_records_a_clean_tool_trace_and_one_flag():
    tools = CareFlowTools()
    loop = _msg(calls=[_call("l", "find_appointment_slots", specialty="ENT")])
    model, _ = _model([loop] * 2)
    r = run_agent("x", tools, max_steps=2, model=model, patient_ref="NOPE-IGNORED")
    assert r.stopped_early and r.tools_used == ["find_appointment_slots", "find_appointment_slots", "flag_for_human"]
    assert len(tools.flags) == 1


def test_escalation_goes_through_the_shared_tools_call_seam():
    # Audit §8.3 #3: M7 wraps tools.call for tracing; the step-cap handoff must pass through it too.
    tools = CareFlowTools()
    seen = []
    real_call = tools.call
    tools.call = lambda name, args, **kw: seen.append(name) or real_call(name, args, **kw)
    loop = _msg(calls=[_call("l", "find_appointment_slots", specialty="ENT")])
    model, _ = _model([loop] * 2)
    r = run_agent("x", tools, max_steps=2, model=model)
    assert seen == ["find_appointment_slots", "find_appointment_slots", "flag_for_human"]
    trace = r.tool_calls[-1]
    assert trace["name"] == "flag_for_human" and json.loads(trace["arguments"])["urgency"] == "Routine"


def test_large_step_cap_does_not_hit_the_framework_recursion_limit():
    loop = _msg(calls=[_call("l", "find_appointment_slots", specialty="ENT")])
    model, seen = _model([loop] * 30)
    r = run_agent("x", CareFlowTools(), max_steps=30, model=model)
    assert r.stopped_early and r.steps == 30 and len(seen) == 30


# ---- session memory foundation (M3) and workflow checkpoints (M5) --------------------------
def _two_turns(checkpointer, thread_a="a", thread_b="b"):
    tools = CareFlowTools()
    m1, _ = _model([_msg("first answer")])
    m2, seen2 = _model([_msg("second answer")])
    run_agent("first question", tools, model=m1, checkpointer=checkpointer, thread_id=thread_a)
    r = run_agent("second question", tools, model=m2, checkpointer=checkpointer, thread_id=thread_a)
    m3, seen3 = _model([_msg("other thread answer")])
    run_agent("unrelated", tools, model=m3, checkpointer=checkpointer, thread_id=thread_b)
    return r, seen2[0]["messages"], seen3[0]["messages"]


def test_same_thread_remembers_earlier_turns_and_other_threads_do_not():
    r, second_turn_input, other_thread_input = _two_turns(MemorySaver())
    contents = [m["content"] for m in second_turn_input]
    assert "first question" in contents and "first answer" in contents and "second question" in contents
    assert all("first question" not in m["content"] for m in other_thread_input)  # privacy: no bleed across threads
    assert r.answer == "second answer"


def test_each_turn_reports_only_its_own_tool_calls_and_resets_the_step_count():
    tools = CareFlowTools()
    cp = MemorySaver()
    lookup = _msg(calls=[_call("t1", "find_appointment_slots", specialty="ENT")])
    m1, _ = _model([lookup, _msg("here are slots")])
    r1 = run_agent("slots?", tools, model=m1, checkpointer=cp, thread_id="t")
    m2, _ = _model([_msg("anything else?")])
    r2 = run_agent("thanks", tools, model=m2, checkpointer=cp, thread_id="t")
    assert r1.tools_used == ["find_appointment_slots"] and r1.steps == 2
    assert r2.tools_used == [] and r2.steps == 1  # not 3: steps restart each turn


def test_sqlite_checkpoint_survives_a_new_connection_and_new_graph(tmp_path):
    db = str(tmp_path / "cp.db")
    conn = sqlite3.connect(db, check_same_thread=False)
    m1, _ = _model([_msg("remember me")])
    run_agent("my first message", CareFlowTools(), model=m1, checkpointer=SqliteSaver(conn), thread_id="p1")
    conn.close()

    conn2 = sqlite3.connect(db, check_same_thread=False)
    m2, seen = _model([_msg("welcome back")])
    r = run_agent("am I still here?", CareFlowTools(), model=m2, checkpointer=SqliteSaver(conn2), thread_id="p1")
    conn2.close()
    assert r.answer == "welcome back"
    assert [m["content"] for m in seen[0]["messages"] if m["role"] == "user"][0] == "my first message"
    assert "remember me" in [m["content"] for m in seen[0]["messages"]]


def test_a_failing_model_raises_and_does_not_poison_the_thread():
    from careflow.llm_client import LLMClientError

    cp = MemorySaver()

    def boom(messages, specs):
        raise LLMClientError("all models failed")

    with pytest.raises(LLMClientError):
        run_agent("x", CareFlowTools(), model=boom, checkpointer=cp, thread_id="t")
    m, _ = _model([_msg("recovered")])
    assert run_agent("again", CareFlowTools(), model=m, checkpointer=cp, thread_id="t").answer == "recovered"


# ---- M3: long-term patient memory, identity, history window ----------------------------------
def _sys_memory(seen_call):
    return [m["content"] for m in seen_call["messages"] if m["role"] == "system" and m["content"].startswith("Patient memory")]


def _known_patient(tools):
    return tools.store.rows("patient_interactions")[0]["patient_ref"]


def test_memory_note_is_given_for_a_verified_patient_only():
    tools = CareFlowTools()
    ref = _known_patient(tools)
    m1, seen1 = _model([_msg("hello")])
    run_agent("hi", tools, model=m1, patient_ref=ref)
    assert len(_sys_memory(seen1[0])) == 1 and ref in _sys_memory(seen1[0])[0]
    m2, seen2 = _model([_msg("hello")])
    run_agent("hi", tools, model=m2)  # no patient: no memory
    assert _sys_memory(seen2[0]) == []


def test_find_patient_verifies_the_caller_and_loads_memory_for_the_thread():
    tools = CareFlowTools()
    p = tools.patients[_known_patient(tools)]
    cp = MemorySaver()
    ask = _msg(calls=[_call("f", "find_patient", given_name=p["given_name"], family_name=p["family_name"],
                            birth_date=p["birth_date"])])
    m1, seen1 = _model([ask, _msg("thanks, found you")])
    r1 = run_agent("I'm calling about my referral", tools, model=m1, checkpointer=cp, thread_id="t")
    assert r1.patient_ref == p["patient_ref"]
    assert _sys_memory(seen1[0]) == [] and len(_sys_memory(seen1[1])) == 1  # loaded right after the check
    m2, seen2 = _model([_msg("still you")])
    run_agent("and another thing", tools, model=m2, checkpointer=cp, thread_id="t")  # no patient_ref passed
    assert len(_sys_memory(seen2[0])) == 1


def test_failed_identity_check_loads_no_memory():
    tools = CareFlowTools()
    ask = _msg(calls=[_call("f", "find_patient", given_name="No", family_name="Body", birth_date="2000-01-01")])
    m, seen = _model([ask, _msg("sorry")])
    r = run_agent("it's me", tools, model=m)
    assert r.patient_ref is None and all(_sys_memory(s) == [] for s in seen)


def test_memory_is_rebuilt_before_each_model_call_so_a_new_booking_shows():
    tools = CareFlowTools()
    ref = next(r for r, p in tools.patients.items() if p["eligibility_status"] == "active" and r in tools.established)
    slot = tools.find_appointment_slots("ENT", patient_ref=ref)["slots"][0]["slot_id"]
    book = _msg(calls=[_call("b", "book_appointment", patient_ref=ref, slot_id=slot)])
    m, seen = _model([book, _msg("booked")])
    run_agent("book it", tools, model=m, patient_ref=ref)
    appt = tools.store.rows_for_patient("appointments", ref)[-1]["appointment_id"]
    assert appt not in _sys_memory(seen[0])[0] and appt in _sys_memory(seen[1])[0]


def test_a_conversation_is_recorded_as_one_contact_for_the_patient():
    tools = CareFlowTools()
    ref = _known_patient(tools)
    before = len(tools.store.rows_for_patient("patient_interactions", ref))
    cp = MemorySaver()
    m, _ = _model([_msg(calls=[_call("e", "check_eligibility", patient_ref=ref)]), _msg("you're covered")])
    run_agent("am I covered?", tools, model=m, patient_ref=ref, channel="sms", checkpointer=cp, thread_id="c1")
    rows = tools.store.rows_for_patient("patient_interactions", ref)
    assert len(rows) == before + 1 and rows[-1]["topic"] == "eligibility_check" and rows[-1]["channel"] == "sms"
    m2, seen2 = _model([_msg("you're welcome")])
    run_agent("thanks", tools, model=m2, checkpointer=cp, thread_id="c1")  # same call: same contact
    assert len(tools.store.rows_for_patient("patient_interactions", ref)) == before + 1
    assert "eligibility_check" not in _repeat_line(seen2[0])  # the call does not count as its own REPEAT
    m3, seen3 = _model([_msg("hello again")])
    run_agent("hi", tools, model=m3, patient_ref=ref)  # a new contact, answered without tools: still recorded
    assert len(tools.store.rows_for_patient("patient_interactions", ref)) == before + 2
    assert "eligibility_check" in _repeat_line(seen3[0])  # ...and the earlier call now counts


def _repeat_line(seen_call):
    lines = _sys_memory(seen_call)[0].splitlines()
    return next((ln for ln in lines if ln.startswith("- Contacted within the last 24 hours")), "")


def test_step_cap_escalation_is_recorded_as_administrative_not_clinical():
    tools = CareFlowTools()
    ref = _known_patient(tools)
    loop = _msg(calls=[_call("l", "find_appointment_slots", specialty="Podiatry")])  # fails every time
    model, _ = _model([loop] * 2)
    run_agent("x", tools, max_steps=2, model=model, patient_ref=ref)
    row = tools.store.rows_for_patient("patient_interactions", ref)[-1]
    assert row["topic"] == "general_enquiry" and row["outcome"] == "callback_scheduled"
    assert tools.flags[-1]["category"] == "administrative"


# ---- feedback M3-5: a verified conversation only touches its own patient's records -------------
def _other(tools, ref):
    return next(r for r in tools.patients if r != ref)


@pytest.mark.parametrize("arg", ["patient_ref", "member_number", "referral_id"])
def test_tools_for_another_patient_are_blocked_on_a_verified_thread(arg):
    tools = CareFlowTools()
    ref = _known_patient(tools)
    other = next(r["patient_ref"] for r in tools.store.rows("referrals") if r["patient_ref"] != ref)
    value = {
        "patient_ref": other.lower(),
        "member_number": tools.patients[other]["member_number"],
        "referral_id": next(r["referral_id"] for r in tools.store.rows("referrals") if r["patient_ref"] == other),
    }[arg]
    name = "check_preauth" if arg == "referral_id" else "check_eligibility"
    model, _ = _model([_msg(calls=[_call("x", name, **{arg: value})]), _msg("sorry")])
    r = run_agent("check my brother", tools, model=model, patient_ref=ref)
    assert r.tool_calls[0]["result"]["ok"] is False and "only that patient's records" in r.tool_calls[0]["result"]["error"]
    assert other not in json.dumps(r.tool_calls[0]["result"])


def test_own_records_and_unbound_threads_are_not_blocked():
    tools = CareFlowTools()
    ref = _known_patient(tools)
    model, _ = _model([_msg(calls=[_call("x", "check_eligibility", patient_ref=ref.lower())]), _msg("ok")])
    assert run_agent("me", tools, model=model, patient_ref=f"  {ref.lower()} ").tool_calls[0]["result"]["ok"]
    model, _ = _model([_msg(calls=[_call("x", "check_eligibility", patient_ref=_other(tools, ref))]), _msg("ok")])
    assert run_agent("staff lookup", tools, model=model).tool_calls[0]["result"]["ok"]  # no verified patient


def test_find_patient_cannot_switch_a_verified_thread_to_someone_else():
    tools = CareFlowTools()
    ref = _known_patient(tools)
    o = tools.patients[_other(tools, ref)]
    ask = _msg(calls=[_call("f", "find_patient", given_name=o["given_name"], family_name=o["family_name"],
                            birth_date=o["birth_date"])])
    model, _ = _model([ask, _msg("sorry")])
    r = run_agent("now me as my brother", tools, model=model, patient_ref=ref)
    assert r.patient_ref == ref and r.tool_calls[0]["result"]["ok"] is False


def test_caller_patient_ref_is_canonicalised():
    tools = CareFlowTools()
    ref = _known_patient(tools)
    model, seen = _model([_msg("hi")])
    r = run_agent("hi", tools, model=model, patient_ref=f" {ref.lower()} ")
    assert r.patient_ref == ref and len(_sys_memory(seen[0])) == 1


def test_session_checkpointer_uses_wal(tmp_path):
    from careflow.agent import session_checkpointer

    cp = session_checkpointer(tmp_path / "s.db")
    assert cp.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert cp.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    cp.conn.close()


def test_memory_can_be_switched_off():
    tools = CareFlowTools()
    ref = _known_patient(tools)
    before = len(tools.store.rows("patient_interactions"))
    m, seen = _model([_msg(calls=[_call("e", "check_eligibility", patient_ref=ref)]), _msg("ok")])
    run_agent("covered?", deps=Deps(tools=tools, model=m, memory=None), patient_ref=ref)
    assert _sys_memory(seen[0]) == [] and len(tools.store.rows("patient_interactions")) == before


def test_model_sees_a_rolling_window_but_the_checkpoint_keeps_everything():
    tools = CareFlowTools()
    cp = MemorySaver()
    seen = []

    def model(messages, specs):
        seen.append(messages)
        return _msg(f"a{len(seen)}")

    g = build_agent_graph(Deps(tools=tools, model=model, memory=None), checkpointer=cp, history_turns=2)
    cfg = {"configurable": {"thread_id": "w"}}
    for i in range(4):
        g.invoke({"messages": [{"role": "user", "content": f"q{i}"}], "steps": 0}, cfg)
    users = [m["content"] for m in seen[-1] if m["role"] == "user"]
    assert users == ["q2", "q3"]
    assert [m["content"] for m in g.get_state(cfg).values["messages"] if m["role"] == "user"] == ["q0", "q1", "q2", "q3"]


def test_policy_search_is_not_offered_without_an_index():
    model, seen = _model([_msg("ok")])
    run_agent("hi", CareFlowTools(), model=model)
    names = {s["function"]["name"] for s in seen[0]["specs"]}
    assert "find_patient" in names and "search_policies" not in names


# ---- repeat markers on staff flags are applied from the records (live demo finding) -------------
def _fresh_patient(tools):
    seen = {r["patient_ref"] for r in tools.store.rows("patient_interactions")}
    return next(r for r in tools.patients if r not in seen)


def _flag_run(tools, ref, **flag_args):
    model, _ = _model([_msg(calls=[_call("f", "flag_for_human", **flag_args)]), _msg("done")])
    r = run_agent("question", tools, model=model, patient_ref=ref)
    return tools.flags[-1], r


def test_clinical_flag_after_a_clinical_contact_today_is_marked_repeat():
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    tools.store.add_interaction(ref, "2026-08-01T08:00", "sms", "clinical_question", "escalated_to_clinician", "x")
    flag, _ = _flag_run(tools, ref, reason="Asked which cream to use", urgency="Routine", clinical=True)
    assert flag["reason"] == "REPEAT: Asked which cream to use" and flag["urgency"] == "Routine"


def test_third_clinical_contact_in_72_hours_is_repeat_priority_with_raised_urgency():
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    for when in ("2026-07-29T15:00", "2026-08-01T08:00"):
        tools.store.add_interaction(ref, when, "sms", "clinical_question", "escalated_to_clinician", "x")
    flag, _ = _flag_run(tools, ref, reason="REPEAT - asked again", urgency="Routine", clinical=True)
    assert flag["reason"] == "Repeat - Priority: asked again" and flag["urgency"] == "Priority"  # not doubled


def test_no_marker_without_same_issue_history_or_a_known_issue():
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    tools.store.add_interaction(ref, "2026-08-01T08:00", "sms", "billing_question", "resolved", "x")
    flag, _ = _flag_run(tools, ref, reason="Clinical question", clinical=True)  # different issue
    assert flag["reason"] == "Clinical question"
    flag, _ = _flag_run(tools, ref, reason="Please call me back")  # administrative, no issue known
    assert flag["reason"] == "Please call me back"


def test_administrative_flag_takes_the_issue_from_this_turns_tools():
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    tools.store.add_interaction(ref, "2026-08-01T08:00", "sms", "eligibility_check", "resolved", "x")
    model, _ = _model([
        _msg(calls=[_call("e", "check_eligibility", patient_ref=ref)]),
        _msg(calls=[_call("f", "flag_for_human", reason="Help with coverage")]),
        _msg("done"),
    ])
    run_agent("coverage again", tools, model=model, patient_ref=ref)
    assert tools.flags[-1]["reason"] == "REPEAT: Help with coverage"


# ---- root cause of the live scenario-4 failure: flags were de-duplicated across contacts -------
def test_replay_of_live_failure_second_clinical_contact_gets_its_own_marked_ticket():
    # The model wrote the same reason on both contacts and did not add REPEAT. Before the fix the store
    # returned the first contact's flag (duplicate=True), so the second contact had no ticket at all.
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    reason = "Patient asked which cream to use for a rash and how often."
    for _ in range(2):
        _flag_run(tools, ref, reason=reason, urgency="Routine", clinical=True)
    flags = [f for f in tools.flags if f["patient_ref"] == ref]
    assert [f["reason"] for f in flags] == [reason, f"REPEAT: {reason}"]
    assert all(f["category"] == "clinical" for f in flags) and flags[0]["contact_id"] != flags[1]["contact_id"]


def test_same_flag_within_one_contact_is_still_recorded_once():
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    twice = _msg(calls=[_call("a", "flag_for_human", reason="Call me back"), _call("b", "flag_for_human", reason="Call me back")])
    model, _ = _model([twice, _msg("ok")])
    r = run_agent("x", tools, model=model, patient_ref=ref)
    assert [c["result"]["duplicate"] for c in r.tool_calls] == [False, True]
    assert len([f for f in tools.flags if f["patient_ref"] == ref]) == 1


def test_a_conversation_keeps_one_contact_id_across_turns():
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    cp = MemorySaver()
    for _ in range(2):
        model, _ = _model([_msg(calls=[_call("a", "flag_for_human", reason="Call me back")]), _msg("ok")])
        run_agent("x", tools, model=model, patient_ref=ref, checkpointer=cp, thread_id="call-7")
    flags = [f for f in tools.flags if f["patient_ref"] == ref]
    assert len(flags) == 1 and flags[0]["contact_id"] == "call-7"


def test_the_model_cannot_set_the_contact_id():
    t = CareFlowTools()
    r = t.call("flag_for_human", {"reason": "x", "contact_id": "forged"})
    assert r["ok"] is False and "contact_id" in r["error"]
    assert t.call("flag_for_human", {"reason": "x"}, context={"contact_id": "c1", "evil": 1})["contact_id"] == "c1"


def test_flag_on_a_verified_conversation_is_attributed_to_the_patient():
    tools = CareFlowTools()
    ref = _fresh_patient(tools)
    _flag_run(tools, ref, reason="Please call me back")  # the model left out patient_ref
    assert tools.flags[-1]["patient_ref"] == ref
    model, _ = _model([_msg(calls=[_call("f", "flag_for_human", reason="Anonymous question")]), _msg("ok")])
    run_agent("x", tools, model=model)  # no verified patient: nothing to attribute
    assert tools.flags[-1]["patient_ref"] is None
