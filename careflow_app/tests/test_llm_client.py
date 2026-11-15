"""Offline tests for the LiteLLM wrapper. No network: litellm.completion is replaced by a fake."""

from types import SimpleNamespace

import litellm
import pytest

from careflow import llm_client
from careflow.config import Settings
from careflow.llm_client import LLMClientError, _strip_fences, complete_structured
from careflow.schemas import PatientRequest

GOOD = '{"urgency": "Routine", "seeks_clinical_advice": false, "specialty": "ENT"}'
BAD = '{"urgency": "Critical", "seeks_clinical_advice": false}'


def _resp(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _settings(**kw) -> Settings:
    base = dict(
        careflow_model="gemini/primary",
        careflow_fallback_models="gemini/backup",
        gemini_api_key="not-a-real-key",
        _env_file=None,
    )
    return Settings(**{**base, **kw})


def _install(monkeypatch, script):
    """script: list of str (returned as content) or Exception (raised), one per call."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        step = script[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        return _resp(step)

    monkeypatch.setattr(llm_client.litellm, "completion", fake)
    return calls


def _transient():
    return litellm.ServiceUnavailableError("503", llm_provider="gemini", model="m")


def test_valid_json_returns_model(monkeypatch):
    calls = _install(monkeypatch, [GOOD])
    out = complete_structured([{"role": "user", "content": "x"}], PatientRequest, _settings())
    assert out.specialty == "ENT" and len(calls) == 1


def test_invalid_json_is_retried_with_the_error_fed_back(monkeypatch):
    calls = _install(monkeypatch, [BAD, GOOD])
    out = complete_structured([{"role": "user", "content": "x"}], PatientRequest, _settings())
    assert out.urgency == "Routine"
    assert len(calls) == 2
    feedback = calls[1]["messages"][-1]["content"]
    assert "failed validation" in feedback and "urgency" in feedback


def test_fenced_json_is_accepted(monkeypatch):
    _install(monkeypatch, [f"```json\n{GOOD}\n```"])
    assert complete_structured([], PatientRequest, _settings()).specialty == "ENT"
    assert _strip_fences("```\n{}\n```") == "{}"
    assert _strip_fences("{}") == "{}"


def test_transient_error_moves_to_fallback_model(monkeypatch):
    calls = _install(monkeypatch, [_transient(), GOOD])
    out = complete_structured([], PatientRequest, _settings())
    assert out.specialty == "ENT"
    assert [c["model"] for c in calls] == ["gemini/primary", "gemini/backup"]


def test_all_models_failing_raises_client_error(monkeypatch):
    _install(monkeypatch, [_transient(), _transient()])
    with pytest.raises(LLMClientError, match="all models failed"):
        complete_structured([], PatientRequest, _settings())


def test_validation_retries_are_bounded(monkeypatch):
    # 3 attempts per model (1 + 2 retries), 2 models, all invalid.
    calls = _install(monkeypatch, [BAD] * 6)
    with pytest.raises(LLMClientError):
        complete_structured([], PatientRequest, _settings())
    assert len(calls) == 6


def test_provider_is_chosen_by_model_string_only():
    s = _settings(openrouter_api_key="or-key", ollama_host="http://h:1")
    assert s.provider_kwargs("gemini/x") == {"api_key": "not-a-real-key"}
    assert s.provider_kwargs("openrouter/a/b") == {"api_key": "or-key"}
    assert s.provider_kwargs("ollama/llama3.1:8b") == {"api_base": "http://h:1"}
    assert s.provider_kwargs("something/else") == {}


def test_model_order_dedupes_fallback():
    s = _settings(careflow_model="gemini/a", careflow_fallback_models="gemini/a, gemini/b")
    assert s.models == ["gemini/a", "gemini/b"]


def test_error_message_does_not_leak_the_key(monkeypatch):
    _install(monkeypatch, [_transient(), _transient()])
    with pytest.raises(LLMClientError) as exc:
        complete_structured([], PatientRequest, _settings())
    assert "not-a-real-key" not in str(exc.value)
