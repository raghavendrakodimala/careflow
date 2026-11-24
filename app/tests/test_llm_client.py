"""Offline tests for the LiteLLM wrapper. No network: litellm.completion is replaced by a fake."""

from types import SimpleNamespace

import litellm
import pytest

from src import llm_client
from src.config import Settings
from src.llm_client import LLMClientError, _strip_fences, complete_structured
from src.schemas import PatientRequest

GOOD = '{"urgency": "Routine", "seeks_clinical_advice": false, "specialty": "ENT"}'
BAD = '{"urgency": "Critical", "seeks_clinical_advice": false}'


def _make_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_settings(**overrides) -> Settings:
    base = dict(
        careflow_model="gemini/primary",
        careflow_fallback_models="gemini/backup",
        gemini_api_key="not-a-real-key",
        _env_file=None,
    )
    return Settings(**{**base, **overrides})


def _install_fake_completion(monkeypatch, script):
    """script: list of str (returned as content) or Exception (raised), one per call."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        step = script[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        return _make_response(step)

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)
    return calls


def _make_transient_error():
    return litellm.ServiceUnavailableError("503", llm_provider="gemini", model="m")


def test_valid_json_returns_model(monkeypatch):
    calls = _install_fake_completion(monkeypatch, [GOOD])
    out = complete_structured([{"role": "user", "content": "x"}], PatientRequest, _make_settings())
    assert out.specialty == "ENT" and len(calls) == 1


def test_invalid_json_is_retried_with_the_error_fed_back(monkeypatch):
    calls = _install_fake_completion(monkeypatch, [BAD, GOOD])
    out = complete_structured([{"role": "user", "content": "x"}], PatientRequest, _make_settings())
    assert out.urgency == "Routine"
    assert len(calls) == 2
    feedback = calls[1]["messages"][-1]["content"]
    assert "failed validation" in feedback and "urgency" in feedback


def test_fenced_json_is_accepted(monkeypatch):
    _install_fake_completion(monkeypatch, [f"```json\n{GOOD}\n```"])
    assert complete_structured([], PatientRequest, _make_settings()).specialty == "ENT"
    assert _strip_fences("```\n{}\n```") == "{}"
    assert _strip_fences("{}") == "{}"


def test_transient_error_moves_to_fallback_model(monkeypatch):
    calls = _install_fake_completion(monkeypatch, [_make_transient_error(), GOOD])
    out = complete_structured([], PatientRequest, _make_settings())
    assert out.specialty == "ENT"
    assert [call["model"] for call in calls] == ["gemini/primary", "gemini/backup"]


def test_all_models_failing_raises_client_error(monkeypatch):
    _install_fake_completion(monkeypatch, [_make_transient_error(), _make_transient_error()])
    with pytest.raises(LLMClientError, match="all models failed"):
        complete_structured([], PatientRequest, _make_settings())


def test_validation_retries_are_bounded(monkeypatch):
    # 3 attempts per model (1 + 2 retries), 2 models, all invalid.
    calls = _install_fake_completion(monkeypatch, [BAD] * 6)
    with pytest.raises(LLMClientError):
        complete_structured([], PatientRequest, _make_settings())
    assert len(calls) == 6


def test_provider_is_chosen_by_model_string_only():
    settings = _make_settings(openrouter_api_key="or-key", ollama_host="http://h:1")
    assert settings.get_provider_options("gemini/x") == {"api_key": "not-a-real-key"}
    assert settings.get_provider_options("openrouter/a/b") == {"api_key": "or-key"}
    assert settings.get_provider_options("ollama/llama3.1:8b") == {"api_base": "http://h:1"}
    assert settings.get_provider_options("something/else") == {}


def test_model_order_dedupes_fallback():
    settings = _make_settings(careflow_model="gemini/a", careflow_fallback_models="gemini/a, gemini/b")
    assert settings.models == ["gemini/a", "gemini/b"]


def test_error_message_does_not_leak_the_key(monkeypatch):
    _install_fake_completion(monkeypatch, [_make_transient_error(), _make_transient_error()])
    with pytest.raises(LLMClientError) as exc:
        complete_structured([], PatientRequest, _make_settings())
    assert "not-a-real-key" not in str(exc.value)


def _make_schema_unsupported_error():
    return litellm.UnsupportedParamsError("response_format is not supported", llm_provider="openrouter", model="m")


def test_rejected_native_schema_falls_back_to_schema_in_prompt_on_the_same_model(monkeypatch):
    # Without this fallback a provider with no native schema support fails outright.
    calls = _install_fake_completion(monkeypatch, [_make_schema_unsupported_error(), GOOD])
    out = complete_structured([{"role": "user", "content": "x"}], PatientRequest, _make_settings())
    assert out.specialty == "ENT"
    assert [call["model"] for call in calls] == ["gemini/primary", "gemini/primary"]
    assert "response_format" in calls[0] and "response_format" not in calls[1]
    assert "JSON Schema" in calls[1]["messages"][-1]["content"] and "urgency" in calls[1]["messages"][-1]["content"]


def test_schema_fallback_still_validates_and_retries(monkeypatch):
    calls = _install_fake_completion(monkeypatch, [_make_schema_unsupported_error(), BAD, GOOD])
    assert complete_structured([], PatientRequest, _make_settings()).urgency == "Routine"
    assert len(calls) == 3 and "failed validation" in calls[2]["messages"][-1]["content"]


def test_schema_worded_bad_request_also_triggers_fallback(monkeypatch):
    error = litellm.BadRequestError("Invalid JSON schema in response_format", model="m", llm_provider="openrouter")
    calls = _install_fake_completion(monkeypatch, [error, GOOD])
    assert complete_structured([], PatientRequest, _make_settings()).specialty == "ENT" and len(calls) == 2


def test_unrelated_bad_request_is_not_swallowed(monkeypatch):
    error = litellm.BadRequestError("invalid api key", model="m", llm_provider="gemini")
    _install_fake_completion(monkeypatch, [error])
    with pytest.raises(litellm.BadRequestError):
        complete_structured([], PatientRequest, _make_settings())


def test_complete_with_tools_falls_back_and_returns_message(monkeypatch):
    message = SimpleNamespace(content="", tool_calls=["call"])
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _make_transient_error()
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)
    out = llm_client.complete_with_tools([], [{"type": "function"}], _make_settings())
    assert out is message and [call["model"] for call in calls] == ["gemini/primary", "gemini/backup"]
    assert calls[0]["tools"] == [{"type": "function"}]


def test_complete_with_tools_all_models_failing_raises(monkeypatch):
    def raise_transient_error(**kwargs):
        raise _make_transient_error()

    monkeypatch.setattr(llm_client.litellm, "completion", raise_transient_error)
    with pytest.raises(LLMClientError, match="all models failed"):
        llm_client.complete_with_tools([], [], _make_settings())


def test_a_mistyped_mode_setting_is_rejected_with_the_allowed_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="off, jsonl, langfuse"):
        Settings(careflow_tracing="jsnol", _env_file=None)
    with pytest.raises(ValidationError, match="none, local, hosted"):
        Settings(careflow_guard_classifier="remote", _env_file=None)
    ok = Settings(careflow_tracing=" JSONL ", careflow_guard_classifier="", _env_file=None)
    assert (ok.careflow_tracing, ok.careflow_guard_classifier) == ("jsonl", "none")
