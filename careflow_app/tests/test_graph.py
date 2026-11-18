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


def test_graph_shape_is_model_tools_escalate():
    g = build_agent_graph(Deps(tools=CareFlowTools(), model=lambda m, s: None))
    drawn = g.get_graph()
    assert {"model", "tools", "escalate"} <= set(drawn.nodes)
    edges = {(e.source, e.target) for e in drawn.edges}
    assert {("__start__", "model"), ("model", "tools"), ("tools", "model"), ("tools", "escalate"), ("escalate", "__end__")} <= edges


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
    tools.call = lambda name, args: seen.append(name) or real_call(name, args)
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
