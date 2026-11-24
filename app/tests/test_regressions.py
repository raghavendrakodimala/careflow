"""One test per defect found after the milestones were built, so each stays fixed."""

import json

import pytest

from scripts._common import VerificationReport
from src import guardrails
from src.agents import TeamResult
from src.config import get_settings
from src.domain import policies
from src.domain.policies import Route
from src.graph import MAX_SEARCHES, OTHER_PATIENT_ERROR, REPEATED_SEARCH_NOTE, SEARCH_LIMIT_NOTE, _has_search_run
from src.guardrails import ClinicalRisk
from src.llm_client import LLMClientError
from src.reliability import ModelUnavailableError
from src.store import RuntimeStore


def _make_call(name: str, **result) -> dict:
    return {"name": name, "arguments": "{}", "result": result, "agent": "referral_tracker"}


# ---- a booking that raised a staff flag is not an automatic outcome ---------------------------------
def test_a_booking_that_raised_a_staff_flag_is_routed_to_human_review():
    flagged = TeamResult(answer="You are booked.", tool_calls=[_make_call("book_appointment", ok=True, flag_id="FLG-1")])
    assert flagged.booking_flag_ids == ["FLG-1"] and flagged.route == Route.HUMAN_REVIEW


def test_a_plain_or_failed_booking_does_not_change_the_route():
    plain = TeamResult(answer="You are booked.", tool_calls=[_make_call("book_appointment", ok=True)])
    failed = TeamResult(answer="That slot has gone.", tool_calls=[_make_call("book_appointment", ok=False, flag_id="FLG-2")])
    assert plain.booking_flag_ids == [] and plain.route == Route.AUTO
    assert failed.booking_flag_ids == [] and failed.route == Route.AUTO


# ---- the hosted classifier's request limit follows the configured gate limit -------------------------
@pytest.mark.parametrize(("configured", "expected"), [(2.0, 2), (4.2, 5), (0.3, 1)])
def test_the_hosted_classifier_is_given_the_configured_time_limit(monkeypatch, configured, expected):
    seen = {}

    def fake_complete(messages, schema, settings, **options):
        seen["timeout"], seen["retries"] = settings.request_timeout_s, settings.transport_retries
        return ClinicalRisk(score=0.0)

    monkeypatch.setattr(guardrails, "complete_structured", fake_complete)
    settings = get_settings().model_copy(update={"careflow_guard_timeout_s": configured})
    guardrails.build_hosted_classifier(settings)("Is my plan active?")
    assert seen == {"timeout": expected, "retries": 0}


# ---- only searches that ran count toward the per-turn cap --------------------------------------------
def test_refused_and_repeated_searches_do_not_use_up_the_search_allowance():
    ran = _make_call("search_policies", ok=True, count=2, results=[])
    failed = _make_call("search_policies", ok=False, unavailable=True, error="index down")
    refused = [
        _make_call("search_policies", ok=False, error=SEARCH_LIMIT_NOTE, search_limit=MAX_SEARCHES),
        _make_call("search_policies", ok=False, error=OTHER_PATIENT_ERROR),
        _make_call("search_policies", ok=False, error="tool 'search_policies' is not available to this agent", tools=[]),
        _make_call("search_policies", ok=True, count=2, results=[], note=REPEATED_SEARCH_NOTE),
    ]
    assert _has_search_run(ran) and _has_search_run(failed)  # a search that reached the index counts, even if it failed
    assert not any(_has_search_run(call) for call in refused)
    assert not _has_search_run(_make_call("check_eligibility", ok=True))


# ---- an urgency the order does not know is rejected ---------------------------------------------------
def test_promoting_a_flag_to_an_unknown_urgency_is_an_error(tmp_path):
    store = RuntimeStore(tmp_path / "runtime.db")
    try:
        with pytest.raises(ValueError, match="unknown urgency 'Bogus'"):
            store.promote_flag("FLG-0001", "Bogus", policies.URGENCY_ORDER)
        with pytest.raises(KeyError):  # a known urgency still reports the unknown flag
            store.promote_flag("no-such-flag", policies.URGENCY_ORDER[0], policies.URGENCY_ORDER)
    finally:
        store.close()


# ---- a demo scenario that loses the model is a failed check, not a crash ------------------------------
@pytest.mark.parametrize("error", [LLMClientError("all models failed"), ModelUnavailableError("circuit open")])
def test_a_scenario_that_loses_the_model_is_recorded_as_failed_and_the_script_goes_on(error, capsys):
    report = VerificationReport()
    with report.attempt("team: both specialists"):
        raise error
    with report.attempt("next scenario"):
        report.record_check("next scenario", [])
    assert [name for name, problems in report.results if problems] == ["team: both specialists"]
    assert type(error).__name__ in json.dumps(report.results) and "1/2 checks passed" in report.build_summary()
    assert "[FAIL]" in capsys.readouterr().out


def test_other_errors_in_a_scenario_still_stop_the_script():
    report = VerificationReport()
    with pytest.raises(ZeroDivisionError), report.attempt("broken scenario"):
        raise ZeroDivisionError
