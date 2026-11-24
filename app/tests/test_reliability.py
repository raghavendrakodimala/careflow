"""Reliability: circuit breaker, retries and fallbacks. Time and sleeping are injected, so nothing waits."""

import json

import pytest

from src.reliability import (
    BOOKING_NOTE,
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    ModelUnavailableError,
    ResilientTools,
    RetryPolicy,
    make_resilient_model,
)
from src.tools import CareFlowTools

from .test_workflow import Harness, _find_patient, _make_call, _make_message


class Clock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time


class FlakyTools:
    """Tools whose named calls raise `failures` times (or forever with None) before working."""

    def __init__(self, inner, names, failures=None):
        self.inner, self.names, self.left, self.calls = inner, set(names), failures, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def call(self, name, arguments=None, context=None):
        self.calls.append(name)
        if name in self.names and (self.left is None or self.left > 0):
            if self.left is not None:
                self.left -= 1
            raise ConnectionError("EHR did not answer")
        return self.inner.call(name, arguments, context)


def _make_resilient(inner, clock=None, attempts=3, threshold=3, cooldown=30.0):
    sleeps = []
    breakers = {backend: CircuitBreaker(backend, threshold, cooldown, now=clock or Clock()) for backend in ("ehr", "policy_index")}
    return ResilientTools(inner, retry=RetryPolicy(attempts=attempts, sleep=sleeps.append), breakers=breakers), sleeps


# ---- circuit breaker -----------------------------------------------------------------------------------
def test_breaker_opens_after_the_threshold_and_fails_fast():
    clock = Clock()
    breaker = CircuitBreaker("ehr", failure_threshold=3, cooldown_s=30, now=clock)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == CLOSED and breaker.allow()
    breaker.record_failure()
    assert breaker.state == OPEN and not breaker.allow()
    clock.time = 29.9
    assert breaker.state == OPEN and not breaker.allow()


def test_breaker_goes_half_open_after_the_cool_down_and_lets_one_trial_through():
    clock = Clock()
    breaker = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30, now=clock)
    breaker.record_failure()
    clock.time = 30
    assert breaker.state == HALF_OPEN
    assert breaker.allow() and not breaker.allow()  # one trial only


def test_a_successful_trial_closes_the_breaker_and_a_failed_one_reopens_it():
    clock = Clock()
    breaker = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30, now=clock)
    breaker.record_failure()
    clock.time = 30
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == OPEN and not breaker.allow()  # the cool-down starts again from now
    clock.time = 60
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == CLOSED and breaker.failures == 0 and breaker.allow()


def test_a_success_resets_the_failure_count():
    breaker = CircuitBreaker("ehr", failure_threshold=3)
    breaker.record_failure(), breaker.record_failure(), breaker.record_success(), breaker.record_failure(), breaker.record_failure()
    assert breaker.state == CLOSED


# ---- retries -------------------------------------------------------------------------------------------
def test_a_tool_that_fails_twice_then_works_gives_one_success_after_two_retries():
    inner = FlakyTools(CareFlowTools(), {"check_eligibility"}, failures=2)
    tools, sleeps = _make_resilient(inner)
    result = tools.call("check_eligibility", {"patient_ref": "MHP-P-00001"})
    assert result["ok"] is True and inner.calls == ["check_eligibility"] * 3
    assert sleeps == [0.2, 0.4]  # backs off between attempts
    assert tools.breakers["ehr"].state == CLOSED


def test_a_failed_booking_is_never_sent_twice():
    inner = FlakyTools(CareFlowTools(), {"book_appointment"}, failures=1)
    tools, sleeps = _make_resilient(inner)
    result = tools.call("book_appointment", {"patient_ref": "MHP-P-00001", "slot_id": "SLOT-1"})
    assert inner.calls == ["book_appointment"] and sleeps == []
    assert result == {"ok": False, "unavailable": True, "error": result["error"], "note": BOOKING_NOTE}
    assert inner.inner.booked == {}


def test_a_business_error_is_an_answer_not_a_failure():
    inner = FlakyTools(CareFlowTools(), set())
    tools, sleeps = _make_resilient(inner, threshold=1)
    result = tools.call("check_eligibility", {"patient_ref": "MHP-P-99999"})
    assert result["ok"] is False and "unavailable" not in result
    assert inner.calls == ["check_eligibility"] and sleeps == [] and tools.breakers["ehr"].state == CLOSED


def test_a_dead_backend_opens_the_breaker_and_later_calls_do_not_reach_it():
    clock = Clock()
    inner = FlakyTools(CareFlowTools(), {"check_eligibility", "find_appointment_slots"})
    tools, _ = _make_resilient(inner, clock=clock, attempts=3, threshold=3)
    first = tools.call("check_eligibility", {"patient_ref": "MHP-P-00001"})
    assert first["unavailable"] is True and len(inner.calls) == 3 and tools.breakers["ehr"].state == OPEN
    second = tools.call("find_appointment_slots", {"specialty": "ENT"})  # same backend: fails at once
    assert second["unavailable"] is True and len(inner.calls) == 3
    assert "flag_for_human" in second["note"] and "Do not guess" in second["note"]
    # A tool on another backend still works.
    assert tools.call("flag_for_human", {"reason": "EHR down"})["ok"] is True
    # After the cool-down one trial goes through; the service is back, so the breaker closes.
    clock.time, inner.names = 30, set()
    assert tools.call("check_eligibility", {"patient_ref": "MHP-P-00001"})["ok"] is True
    assert tools.breakers["ehr"].state == CLOSED


def test_resilient_tools_look_like_the_tools_they_wrap():
    inner = CareFlowTools()
    tools, _ = _make_resilient(inner)
    assert tools.store is inner.store and tools.list_available_tools() == inner.list_available_tools()
    assert tools.call("flag_for_human", {"reason": "x"}, context={"contact_id": "t9"})["contact_id"] == "t9"


# ---- model ---------------------------------------------------------------------------------------------
def test_model_breaker_fails_fast_then_recovers():
    clock, calls = Clock(), []

    def call_model(messages, specs):
        calls.append(1)
        if len(calls) <= 2:
            raise TimeoutError("provider timed out")
        return _make_message("ok")

    safe = make_resilient_model(call_model, CircuitBreaker("model", failure_threshold=2, cooldown_s=10, now=clock))
    for _ in range(2):
        with pytest.raises(TimeoutError):
            safe([], [])
    with pytest.raises(ModelUnavailableError):
        safe([], [])
    assert len(calls) == 2  # the third call never reached the provider
    clock.time = 10
    assert safe([], []).content == "ok" and safe.breaker.state == CLOSED


# ---- end to end: no invented answers -------------------------------------------------------------------
def test_ehr_down_sends_the_case_to_staff_and_the_model_is_told_not_to_guess():
    seen = []
    steps = [_make_message(calls=[_make_call("c1", "find_appointment_slots", specialty="Dermatology")]),
             _make_message("I cannot check appointments right now; staff will contact you.")]
    harness = Harness(steps=steps)
    inner = FlakyTools(harness.tools, {"check_eligibility", "check_preauth", "find_appointment_slots", "book_appointment"})
    harness.dependencies.agent.tools, _ = _make_resilient(inner)
    original = harness.dependencies.agent.model
    harness.dependencies.agent.model = lambda messages, specs: seen.append(messages) or original(messages, specs)
    result = harness.start(_find_patient(harness.tools))
    assert (result.route, result.status) == ("human_review", "awaiting_human_review")
    assert any("could not be reached (eligibility)" in reason for reason in result.reasons)
    assert result.eligibility["unavailable"] is True and harness.tools.booked == {}
    assert all("book_appointment" not in names for names in harness.offered)  # nothing is booked on a guess
    tool_results = [json.loads(message["content"]) for message in seen[-1] if message["role"] == "tool"]
    assert tool_results and all(tool_result["unavailable"] and "Do not guess" in tool_result["note"] for tool_result in tool_results)
    assert len(harness.tools.flags) == 1


def test_flaky_ehr_gives_the_same_route_as_a_healthy_one():
    healthy = Harness().start(_find_patient(CareFlowTools()), answer=False)
    harness = Harness()
    harness.dependencies.agent.tools, sleeps = _make_resilient(FlakyTools(harness.tools, {"check_eligibility"}, failures=2))
    flaky = harness.start(_find_patient(harness.tools), answer=False)
    assert (flaky.route, flaky.eligibility) == (healthy.route, healthy.eligibility) and len(sleeps) == 2


def test_model_down_hands_the_case_to_staff_without_an_answer():
    harness = Harness()
    breaker = CircuitBreaker("model", failure_threshold=1)
    breaker.record_failure()
    harness.dependencies.agent.model = make_resilient_model(harness.dependencies.agent.model, breaker)
    result = harness.start(_find_patient(harness.tools))
    assert (result.route, result.status) == ("human_review", "awaiting_human_review")
    assert harness.offered == [] and "unavailable" in result.reasons[0]
    assert "staff" in result.response.lower() and harness.tools.booked == {}


def test_a_result_key_named_like_a_span_field_does_not_break_the_call():
    class OddTools:
        def call(self, name, arguments=None, context=None):
            return {"ok": True, "attempts": 9, "breaker": "mine"}

    tools = ResilientTools(OddTools(), retry=RetryPolicy(attempts=0, sleep=lambda seconds: None))
    assert tools.call("check_eligibility", {})["ok"] is True  # and zero configured attempts still means one try
