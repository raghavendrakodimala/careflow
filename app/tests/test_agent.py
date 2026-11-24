"""Offline tests for the agent loop. complete_with_tools is replaced by a scripted fake."""

import json
from types import SimpleNamespace

import pytest

from src.agent import run_agent
from src.llm_client import LLMClientError
from src.tools import CareFlowTools


def _make_call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _make_message(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls or [])


def _script_model(steps):
    """steps: messages (or Exceptions) returned in order. Returns (model, seen); pass the model to run_agent(model=...)."""
    seen = []

    def respond(messages, tools):
        seen.append([dict(message) for message in messages])
        step = steps[len(seen) - 1]
        if isinstance(step, Exception):
            raise step
        return step

    return respond, seen


@pytest.fixture()
def tools():
    return CareFlowTools()


@pytest.fixture()
def active(tools):
    return next(patient for patient in tools.patients.values() if patient["eligibility_status"] == "active")


def test_plain_answer_needs_no_tools(tools):
    model, _ = _script_model([_make_message("Hello, how can I help?")])
    result = run_agent("hi", tools, model=model)
    assert result.answer == "Hello, how can I help?" and result.tool_calls == [] and result.steps == 1


def test_tool_result_is_fed_back_and_answer_follows(tools, active):
    model, seen = _script_model([
            _make_message(calls=[_make_call("c1", "check_eligibility", patient_ref=active["patient_ref"])]),
            _make_message("You are covered."),
        ])
    result = run_agent("am I covered?", tools, model=model)
    assert result.tools_used == ["check_eligibility"] and result.tool_calls[0]["result"]["eligible"] is True
    assert result.answer == "You are covered." and result.steps == 2
    last = seen[1][-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "c1"
    assert json.loads(last["content"])["plan_code"] == active["plan_code"]


def test_multi_tool_turn_runs_in_order_and_state_changes(tools, active):
    ref = active["patient_ref"]
    slot = tools.find_appointment_slots("Cardiology")["slots"][0]["slot_id"]
    model, _ = _script_model(
        [
            _make_message(calls=[_make_call("a", "check_eligibility", patient_ref=ref), _make_call("b", "find_appointment_slots", specialty="Cardiology")]),
            _make_message(calls=[_make_call("c", "book_appointment", patient_ref=ref, slot_id=slot)]),
            _make_message("Booked."),
        ]
    )
    result = run_agent("book cardiology", tools, model=model)
    assert result.tools_used == ["check_eligibility", "find_appointment_slots", "book_appointment"]
    assert tools.booked == {slot: ref}


def test_tool_errors_reach_the_model_and_do_not_crash(tools):
    model, seen = _script_model([_make_message(calls=[_make_call("x", "calculate_copay", patient_ref="NOPE", procedure_code="CARD-CONS")]), _make_message("I could not find you.")])
    result = run_agent("what do I owe", tools, model=model)
    assert result.tool_calls[0]["result"]["ok"] is False
    assert json.loads(seen[1][-1]["content"])["ok"] is False and result.answer == "I could not find you."


def test_unknown_tool_and_malformed_arguments_are_handled(tools):
    bad = SimpleNamespace(id="z", function=SimpleNamespace(name="drop_tables", arguments="{oops"))
    model, _ = _script_model([_make_message(calls=[bad]), _make_message("Sorry.")])
    result = run_agent("x", tools, model=model)
    assert result.tool_calls[0]["result"]["ok"] is False and result.answer == "Sorry."


def test_step_limit_flags_for_staff_instead_of_looping(tools):
    looping = _make_message(calls=[_make_call("l", "find_appointment_slots", specialty="ENT")])
    model, _ = _script_model([looping] * 3)
    result = run_agent("x", tools, max_steps=3, model=model)
    assert result.stopped_early and result.steps == 3
    assert result.tools_used[-1] == "flag_for_human" and len(tools.flags) == 1
    assert "staff" in result.answer


def test_patient_ref_context_reaches_the_model(tools, active):
    model, seen = _script_model([_make_message("ok")])
    run_agent("hello", tools, patient_ref=active["patient_ref"], model=model)
    assert active["patient_ref"] in seen[0][1]["content"]


def test_system_prompt_forbids_clinical_advice_and_is_sent_first(tools):
    model, seen = _script_model([_make_message("ok")])
    run_agent("x", tools, model=model)
    system_message = seen[0][0]
    assert system_message["role"] == "system"
    low = system_message["content"].lower()
    assert "never give clinical advice" in low and "flag_for_human" in low and "doctor" in low


def test_provider_failure_propagates_as_client_error(tools):
    model, _ = _script_model([LLMClientError("all models failed")])
    with pytest.raises(LLMClientError):
        run_agent("x", tools, model=model)
