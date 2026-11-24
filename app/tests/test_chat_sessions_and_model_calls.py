"""Vetoed turns, chat threads, the search limit and model-call spans. Offline: fakes for the model,
the reviewer and the classifier."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from src.agents import SafetyVerdict, run_team
from src.api.app import CHAT_PREFIX, Resources, create_app
from src.config import Settings
from src.graph import MAX_SEARCHES, SEARCH_LIMIT_NOTE, build_agent_graph
from src.guardrails import ClinicalRisk
from src.llm_client import LAST_CALL, _note_call
from src.prompts import DEFLECTION_SCRIPT
from src.reliability import OPEN, CircuitBreaker, ModelUnavailableError, guard_model_call, make_resilient_model
from src.telemetry import MemorySink, Tracer
from src.tools import CareFlowTools

from .test_agents import VETO, Team
from .test_session_safety import _Index
from .test_workflow import Harness, _find_patient, _make_call, _make_message


# ---- a veto on a two-specialist turn leaves the withheld text in no stored message ----------
def test_a_vetoed_two_specialist_turn_leaves_no_trace_in_any_role():
    first, second = "Those tablets are mainly used for blood pressure.", "And the cream is a mild steroid."
    team, saver = Team(steps=[_make_message(first), _make_message(second)]), MemorySaver()
    ref = _find_patient(team.tools)
    result = run_team("What will I owe for a dermatology consultation, and can you book it?", team.dependencies,
                 reviewer=lambda message, draft: VETO, patient_ref=ref, thread_id="v2", checkpointer=saver)
    assert result.agents == ["insurance", "referral_tracker"] and result.vetoed and result.answer == DEFLECTION_SCRIPT
    stored = build_agent_graph(team.dependencies, checkpointer=saver).get_state({"configurable": {"thread_id": "v2"}}).values["messages"]
    text = json.dumps(stored)
    assert first not in text and second not in text and "tablets" not in text and "steroid" not in text
    assert [message["content"] for message in stored if message["role"] == "assistant"] == [DEFLECTION_SCRIPT, DEFLECTION_SCRIPT]
    assert "see the reply above" in stored[2]["content"]  # the second specialist's message carries no answer


def test_without_a_session_the_second_specialist_is_still_told_the_first_answer():
    team = Team(steps=[_make_message("You would owe $45."), _make_message("Booked.")])
    run_team("What will I owe for a dermatology consultation, and can you book it?", team.dependencies)
    assert "You would owe $45." in team.calls[1]["user"]


# ---- the first chat turn is stored; chat and triage threads are separate ----------------
def _open_client(harness: Harness) -> TestClient:
    harness.dependencies.review = lambda message, reply: SafetyVerdict(safe=True)
    return TestClient(create_app(Resources(harness.dependencies, harness.checkpointer, classifier=lambda text: ClinicalRisk(score=0.0),
                                           settings=Settings(careflow_tracing="off"))))


def test_chat_without_a_thread_id_stores_the_turn_under_the_id_it_returns():
    harness = Harness()
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        first = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": ref}).json()
        again = client.post("/v1/chat", json={"message": "And my co-pay?", "thread_id": first["thread_id"]}).json()
    assert not first["thread_id"].startswith(CHAT_PREFIX)  # the prefix is internal
    assert again["patient_ref"] == ref  # the patient verified on turn 1, with no patient_ref sent on turn 2
    stored = harness.checkpointer.get_tuple({"configurable": {"thread_id": CHAT_PREFIX + first["thread_id"]}})
    assert stored is not None and harness.checkpointer.get_tuple({"configurable": {"thread_id": first["thread_id"]}}) is None


def test_triage_and_review_requests_cannot_use_a_chat_thread():
    harness = Harness()
    with _open_client(harness) as client:
        triage = client.post("/v1/triage", json={"raw_text": "I need an appointment.", "thread_id": "chat:abc"})
        review = client.get("/v1/reviews/chat:abc")
        decide = client.post("/v1/reviews/Chat:abc/decision", json={"action": "approve", "reviewer_id": "nurse-1"})
        same_id = client.post("/v1/triage", json={"raw_text": "I need an appointment.", "thread_id": "abc"})
        chat = client.post("/v1/chat", json={"message": "Is my plan active?", "thread_id": "abc"})
    assert (triage.status_code, review.status_code, decide.status_code) == (422, 422, 422)
    assert same_id.status_code == 200 and chat.status_code == 200  # the same client id, two separate threads


# ---- at most three policy searches in a turn ----------------------------------------------------
def test_the_fourth_search_in_a_turn_is_not_run():
    index = _Index()
    steps = [_make_message(calls=[_make_call(f"c{number}", "search_policies", query=f"co-pay wording {number}")]) for number in range(MAX_SEARCHES + 1)]
    team = Team(tools=CareFlowTools(policy_index=index), steps=[*steps, _make_message("I do not have that information.")])
    result = run_team("What is the specialist co-pay on MERIDIAN-GOLD?", team.dependencies)
    assert len(index.queries) == MAX_SEARCHES and not result.stopped_early
    assert result.tool_calls[-1]["result"] == {"ok": False, "error": SEARCH_LIMIT_NOTE, "search_limit": MAX_SEARCHES}
    assert all(call["result"]["ok"] for call in result.tool_calls[:MAX_SEARCHES])


# ---- structured calls behind the breaker; the model that answered and its tokens ------------
def _make_response(prompt=11, completion=7):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def test_structured_calls_share_the_breaker_and_fail_fast_when_it_is_open():
    breaker, calls = CircuitBreaker("model", failure_threshold=2, cooldown_s=30, now=lambda: 0.0), []

    def parse(record):
        calls.append(record)
        raise RuntimeError("provider down")

    call = guard_model_call(parse, breaker)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            call({"raw_text": "x"})
    assert breaker.state == OPEN
    with pytest.raises(ModelUnavailableError):
        call({"raw_text": "x"})
    assert len(calls) == 2  # the third call never reached the provider


def test_spans_record_the_model_that_answered_and_its_tokens():
    sink = MemorySink()
    tracer = Tracer([sink])

    def parse_structured(record):
        _note_call("gemini/gemini-flash-lite-latest", _make_response())  # the fallback answered
        return "parsed"

    def answer_with_tools(messages, specs):
        _note_call("gemini/gemini-flash-latest", _make_response(120, 30))
        return _make_message("ok")

    with tracer.open_trace("t1", "team"):
        assert guard_model_call(parse_structured, CircuitBreaker("model"), tracer, name="gemini/gemini-flash-latest")({}) == "parsed"
        make_resilient_model(answer_with_tools, CircuitBreaker("model"), tracer, name="gemini/gemini-flash-latest")([], [])
    first, second = [recorded_span for recorded_span in sink.spans if recorded_span["name"] == "model.complete"]
    assert (first["model"], first["input_tokens"], first["output_tokens"]) == ("gemini/gemini-flash-lite-latest", 11, 7)
    assert (second["model"], second["input_tokens"], second["output_tokens"]) == ("gemini/gemini-flash-latest", 120, 30)


def test_a_call_that_records_nothing_keeps_the_configured_name():
    sink = MemorySink()
    tracer = Tracer([sink])
    LAST_CALL.set({"model": "stale", "input_tokens": 1, "output_tokens": 1})
    with tracer.open_trace("t2", "team"):
        guard_model_call(lambda record: "parsed", CircuitBreaker("model"), tracer, name="configured")({})
    span = next(recorded_span for recorded_span in sink.spans if recorded_span["name"] == "model.complete")
    assert span["model"] == "configured" and "input_tokens" not in span


# ---- text written beside a tool call in a vetoed turn is removed too ------------
def test_a_veto_also_removes_text_the_model_wrote_beside_a_tool_call():
    from src.state import add_messages

    call = [{"id": "c1", "type": "function", "function": {"name": "calculate_copay", "arguments": "{}"}}]
    old = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Let me check that.", "tool_calls": call},  # an earlier turn: kept
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "What will I owe?"},
        {"role": "assistant", "content": "Those are beta blockers; let me check your co-pay.", "tool_calls": call},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "assistant", "content": "Take one a day. You owe $40."},
    ]
    new = add_messages(old, [{"role": "redact", "content": DEFLECTION_SCRIPT, "count": 1}])
    assert "beta blockers" not in json.dumps(new) and "Take one a day" not in json.dumps(new)
    assert new[5]["content"] is None and new[5]["tool_calls"] == call and new[7]["content"] == DEFLECTION_SCRIPT
    assert new[:5] == old[:5] and new[6] == old[6]  # the earlier turn and the tool result are untouched
