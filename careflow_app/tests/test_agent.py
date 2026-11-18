"""Offline tests for the M2 agent loop. complete_with_tools is replaced by a scripted fake."""

import json
from types import SimpleNamespace

import pytest

from careflow.agent import run_agent
from careflow.llm_client import LLMClientError
from careflow.tools import CareFlowTools


def _call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _msg(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls or [])


def _script(steps):
    """steps: messages (or Exceptions) returned in order. Returns (model, seen); pass the model to run_agent(model=...)."""
    seen = []

    def fake(messages, tools):
        seen.append([dict(m) for m in messages])
        step = steps[len(seen) - 1]
        if isinstance(step, Exception):
            raise step
        return step

    return fake, seen


@pytest.fixture()
def tools():
    return CareFlowTools()


@pytest.fixture()
def active(tools):
    return next(p for p in tools.patients.values() if p["eligibility_status"] == "active")


def test_plain_answer_needs_no_tools(tools):
    model, _ = _script([_msg("Hello, how can I help?")])
    r = run_agent("hi", tools, model=model)
    assert r.answer == "Hello, how can I help?" and r.tool_calls == [] and r.steps == 1


def test_tool_result_is_fed_back_and_answer_follows(tools, active):
    model, seen = _script([
            _msg(calls=[_call("c1", "check_eligibility", patient_ref=active["patient_ref"])]),
            _msg("You are covered."),
        ])
    r = run_agent("am I covered?", tools, model=model)
    assert r.tools_used == ["check_eligibility"] and r.tool_calls[0]["result"]["eligible"] is True
    assert r.answer == "You are covered." and r.steps == 2
    last = seen[1][-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "c1"
    assert json.loads(last["content"])["plan_code"] == active["plan_code"]


def test_multi_tool_turn_runs_in_order_and_state_changes(tools, active):
    ref = active["patient_ref"]
    slot = tools.find_appointment_slots("Cardiology")["slots"][0]["slot_id"]
    model, _ = _script(
        [
            _msg(calls=[_call("a", "check_eligibility", patient_ref=ref), _call("b", "find_appointment_slots", specialty="Cardiology")]),
            _msg(calls=[_call("c", "book_appointment", patient_ref=ref, slot_id=slot)]),
            _msg("Booked."),
        ]
    )
    r = run_agent("book cardiology", tools, model=model)
    assert r.tools_used == ["check_eligibility", "find_appointment_slots", "book_appointment"]
    assert tools.booked == {slot: ref}


def test_tool_errors_reach_the_model_and_do_not_crash(tools):
    model, seen = _script([_msg(calls=[_call("x", "calculate_copay", patient_ref="NOPE", procedure_code="CARD-CONS")]), _msg("I could not find you.")])
    r = run_agent("what do I owe", tools, model=model)
    assert r.tool_calls[0]["result"]["ok"] is False
    assert json.loads(seen[1][-1]["content"])["ok"] is False and r.answer == "I could not find you."


def test_unknown_tool_and_malformed_arguments_are_handled(tools):
    bad = SimpleNamespace(id="z", function=SimpleNamespace(name="drop_tables", arguments="{oops"))
    model, _ = _script([_msg(calls=[bad]), _msg("Sorry.")])
    r = run_agent("x", tools, model=model)
    assert r.tool_calls[0]["result"]["ok"] is False and r.answer == "Sorry."


def test_step_limit_flags_for_staff_instead_of_looping(tools):
    looping = _msg(calls=[_call("l", "find_appointment_slots", specialty="ENT")])
    model, _ = _script([looping] * 3)
    r = run_agent("x", tools, max_steps=3, model=model)
    assert r.stopped_early and r.steps == 3
    assert r.tools_used[-1] == "flag_for_human" and len(tools.flags) == 1
    assert "staff" in r.answer


def test_patient_ref_context_reaches_the_model(tools, active):
    model, seen = _script([_msg("ok")])
    run_agent("hello", tools, patient_ref=active["patient_ref"], model=model)
    assert active["patient_ref"] in seen[0][1]["content"]


def test_system_prompt_forbids_clinical_advice_and_is_sent_first(tools):
    model, seen = _script([_msg("ok")])
    run_agent("x", tools, model=model)
    sys = seen[0][0]
    assert sys["role"] == "system"
    low = sys["content"].lower()
    assert "never give clinical advice" in low and "flag_for_human" in low and "doctor" in low


def test_provider_failure_propagates_as_client_error(tools):
    model, _ = _script([LLMClientError("all models failed")])
    with pytest.raises(LLMClientError):
        run_agent("x", tools, model=model)
