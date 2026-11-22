"""M7 reliability: circuit breaker, retries and fallbacks. Time and sleeping are injected, so nothing waits."""

import json

import pytest

from careflow.reliability import (BOOKING_NOTE, CLOSED, HALF_OPEN, OPEN, CircuitBreaker, ModelUnavailable,
                                  ResilientTools, RetryPolicy, resilient_model)
from careflow.tools import CareFlowTools

from .test_workflow import Harness, _call, _msg, _patient


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Flaky:
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


def _resilient(inner, clock=None, attempts=3, threshold=3, cooldown=30.0):
    sleeps = []
    breakers = {b: CircuitBreaker(b, threshold, cooldown, now=clock or Clock()) for b in ("ehr", "policy_index")}
    return ResilientTools(inner, retry=RetryPolicy(attempts=attempts, sleep=sleeps.append), breakers=breakers), sleeps


# ---- circuit breaker -----------------------------------------------------------------------------------
def test_breaker_opens_after_the_threshold_and_fails_fast():
    clock = Clock()
    b = CircuitBreaker("ehr", failure_threshold=3, cooldown_s=30, now=clock)
    for _ in range(2):
        b.record_failure()
    assert b.state == CLOSED and b.allow()
    b.record_failure()
    assert b.state == OPEN and not b.allow()
    clock.t = 29.9
    assert b.state == OPEN and not b.allow()


def test_breaker_goes_half_open_after_the_cool_down_and_lets_one_trial_through():
    clock = Clock()
    b = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30, now=clock)
    b.record_failure()
    clock.t = 30
    assert b.state == HALF_OPEN
    assert b.allow() and not b.allow()  # one trial only


def test_a_successful_trial_closes_the_breaker_and_a_failed_one_reopens_it():
    clock = Clock()
    b = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30, now=clock)
    b.record_failure()
    clock.t = 30
    assert b.allow()
    b.record_failure()
    assert b.state == OPEN and not b.allow()  # the cool-down starts again from now
    clock.t = 60
    assert b.allow()
    b.record_success()
    assert b.state == CLOSED and b.failures == 0 and b.allow()


def test_a_success_resets_the_failure_count():
    b = CircuitBreaker("ehr", failure_threshold=3)
    b.record_failure(), b.record_failure(), b.record_success(), b.record_failure(), b.record_failure()
    assert b.state == CLOSED


# ---- retries -------------------------------------------------------------------------------------------
def test_a_tool_that_fails_twice_then_works_gives_one_success_after_two_retries():
    inner = Flaky(CareFlowTools(), {"check_eligibility"}, failures=2)
    tools, sleeps = _resilient(inner)
    result = tools.call("check_eligibility", {"patient_ref": "MHP-P-00001"})
    assert result["ok"] is True and inner.calls == ["check_eligibility"] * 3
    assert sleeps == [0.2, 0.4]  # backs off between attempts
    assert tools.breakers["ehr"].state == CLOSED


def test_a_failed_booking_is_never_sent_twice():
    inner = Flaky(CareFlowTools(), {"book_appointment"}, failures=1)
    tools, sleeps = _resilient(inner)
    result = tools.call("book_appointment", {"patient_ref": "MHP-P-00001", "slot_id": "SLOT-1"})
    assert inner.calls == ["book_appointment"] and sleeps == []
    assert result == {"ok": False, "unavailable": True, "error": result["error"], "note": BOOKING_NOTE}
    assert inner.inner.booked == {}


def test_a_business_error_is_an_answer_not_a_failure():
    inner = Flaky(CareFlowTools(), set())
    tools, sleeps = _resilient(inner, threshold=1)
    result = tools.call("check_eligibility", {"patient_ref": "MHP-P-99999"})
    assert result["ok"] is False and "unavailable" not in result
    assert inner.calls == ["check_eligibility"] and sleeps == [] and tools.breakers["ehr"].state == CLOSED


def test_a_dead_backend_opens_the_breaker_and_later_calls_do_not_reach_it():
    clock = Clock()
    inner = Flaky(CareFlowTools(), {"check_eligibility", "find_appointment_slots"})
    tools, _ = _resilient(inner, clock=clock, attempts=3, threshold=3)
    first = tools.call("check_eligibility", {"patient_ref": "MHP-P-00001"})
    assert first["unavailable"] is True and len(inner.calls) == 3 and tools.breakers["ehr"].state == OPEN
    second = tools.call("find_appointment_slots", {"specialty": "ENT"})  # same backend: fails at once
    assert second["unavailable"] is True and len(inner.calls) == 3
    assert "flag_for_human" in second["note"] and "Do not guess" in second["note"]
    # A tool on another backend still works.
    assert tools.call("flag_for_human", {"reason": "EHR down"})["ok"] is True
    # After the cool-down one trial goes through; the service is back, so the breaker closes.
    clock.t, inner.names = 30, set()
    assert tools.call("check_eligibility", {"patient_ref": "MHP-P-00001"})["ok"] is True
    assert tools.breakers["ehr"].state == CLOSED


def test_resilient_tools_look_like_the_tools_they_wrap():
    inner = CareFlowTools()
    tools, _ = _resilient(inner)
    assert tools.store is inner.store and tools.available_tools() == inner.available_tools()
    assert tools.call("flag_for_human", {"reason": "x"}, context={"contact_id": "t9"})["contact_id"] == "t9"


# ---- model ---------------------------------------------------------------------------------------------
def test_model_breaker_fails_fast_then_recovers():
    clock, calls = Clock(), []

    def model(messages, specs):
        calls.append(1)
        if len(calls) <= 2:
            raise TimeoutError("provider timed out")
        return _msg("ok")

    safe = resilient_model(model, CircuitBreaker("model", failure_threshold=2, cooldown_s=10, now=clock))
    for _ in range(2):
        with pytest.raises(TimeoutError):
            safe([], [])
    with pytest.raises(ModelUnavailable):
        safe([], [])
    assert len(calls) == 2  # the third call never reached the provider
    clock.t = 10
    assert safe([], []).content == "ok" and safe.breaker.state == CLOSED


# ---- end to end: no invented answers -------------------------------------------------------------------
def test_ehr_down_sends_the_case_to_staff_and_the_model_is_told_not_to_guess():
    seen = []
    steps = [_msg(calls=[_call("c1", "find_appointment_slots", specialty="Dermatology")]),
             _msg("I cannot check appointments right now; staff will contact you.")]
    h = Harness(steps=steps)
    inner = Flaky(h.tools, {"check_eligibility", "check_preauth", "find_appointment_slots", "book_appointment"})
    h.deps.agent.tools, _ = _resilient(inner)
    original = h.deps.agent.model
    h.deps.agent.model = lambda messages, specs: seen.append(messages) or original(messages, specs)
    r = h.start(_patient(h.tools))
    assert (r.route, r.status) == ("human_review", "awaiting_human_review")
    assert any("could not be reached (eligibility)" in x for x in r.reasons)
    assert r.eligibility["unavailable"] is True and h.tools.booked == {}
    assert all("book_appointment" not in names for names in h.offered)  # nothing is booked on a guess
    tool_results = [json.loads(m["content"]) for m in seen[-1] if m["role"] == "tool"]
    assert tool_results and all(t["unavailable"] and "Do not guess" in t["note"] for t in tool_results)
    assert len(h.tools.flags) == 1


def test_flaky_ehr_gives_the_same_route_as_a_healthy_one():
    healthy = Harness().start(_patient(CareFlowTools()), answer=False)
    h = Harness()
    h.deps.agent.tools, sleeps = _resilient(Flaky(h.tools, {"check_eligibility"}, failures=2))
    flaky = h.start(_patient(h.tools), answer=False)
    assert (flaky.route, flaky.eligibility) == (healthy.route, healthy.eligibility) and len(sleeps) == 2


def test_model_down_hands_the_case_to_staff_without_an_answer():
    h = Harness()
    breaker = CircuitBreaker("model", failure_threshold=1)
    breaker.record_failure()
    h.deps.agent.model = resilient_model(h.deps.agent.model, breaker)
    r = h.start(_patient(h.tools))
    assert (r.route, r.status) == ("human_review", "awaiting_human_review")
    assert h.offered == [] and "unavailable" in r.reasons[0]
    assert "staff" in r.response.lower() and h.tools.booked == {}
