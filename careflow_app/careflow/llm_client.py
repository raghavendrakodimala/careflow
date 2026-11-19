"""Provider-agnostic structured-output client built on LiteLLM."""

from __future__ import annotations

import json
import re
from typing import TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

# Failures worth moving to the next model for; anything else is a real bug.
_TRANSIENT = (
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.APIConnectionError,
)


class LLMClientError(RuntimeError):
    pass


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


_SCHEMA_WORDS = re.compile(r"response_format|response_schema|json_schema|structured output|schema", re.IGNORECASE)


def _schema_rejected(exc: Exception) -> bool:
    """The provider or model cannot take a native response schema (several OpenRouter and Ollama models)."""
    if isinstance(exc, litellm.UnsupportedParamsError):
        return True
    return isinstance(exc, litellm.BadRequestError) and bool(_SCHEMA_WORDS.search(str(exc)))


def _schema_in_prompt(messages: list[dict], response_model: type[BaseModel]) -> list[dict]:
    """Fallback: describe the JSON schema in the prompt instead; the reply is still validated by Pydantic."""
    schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    note = {
        "role": "system",
        "content": f"Reply with one JSON object only, no prose, matching this JSON Schema:\n{schema}",
    }
    return [*messages, note]


def complete_with_tools(messages: list[dict], tools: list[dict], settings: Settings | None = None):
    """One tool-calling turn. Returns the assistant message (may hold tool_calls).

    Same provider rules as complete_structured: LiteLLM retries transient errors, then the next
    fallback model is tried. The model is chosen by config only.
    """
    settings = settings or get_settings()
    last_error: Exception | None = None
    for model in settings.models:
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0,
                timeout=settings.request_timeout_s,
                num_retries=settings.transport_retries,
                **settings.provider_kwargs(model),
            )
        except _TRANSIENT as exc:
            last_error = exc
            continue
        return resp.choices[0].message
    raise LLMClientError(f"all models failed ({', '.join(settings.models)}): {type(last_error).__name__}: {last_error}")


def complete_structured(
    messages: list[dict],
    response_model: type[T],
    settings: Settings | None = None,
    validation_retries: int = 2,
) -> T:
    """Ask the model for JSON matching `response_model`; return a validated object.

    - Transient provider errors: LiteLLM retries, then the next fallback model is tried.
    - Native schema rejected by the provider: the same model is asked again with the schema in the prompt.
    - Invalid output: the validation error is sent back and the model tries again.
    """
    settings = settings or get_settings()
    last_error: Exception | None = None

    for model in settings.models:
        convo = list(messages)
        native = True  # pass the Pydantic schema as response_format until the provider says it cannot
        attempts = 0
        while attempts <= validation_retries:
            schema_arg = {"response_format": response_model} if native else {}
            try:
                resp = litellm.completion(
                    model=model,
                    messages=convo,
                    temperature=0,
                    timeout=settings.request_timeout_s,
                    num_retries=settings.transport_retries,
                    **schema_arg,
                    **settings.provider_kwargs(model),
                )
            except _TRANSIENT as exc:
                last_error = exc
                break  # next model
            except litellm.BadRequestError as exc:
                if not (native and _schema_rejected(exc)):
                    raise
                last_error = exc
                native = False  # same model, schema in the prompt; does not use up a validation retry
                convo = _schema_in_prompt(convo, response_model)
                continue
            attempts += 1
            content = resp.choices[0].message.content or ""
            try:
                return response_model.model_validate_json(_strip_fences(content))
            except ValidationError as exc:
                last_error = exc
                convo = [
                    *convo,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "That JSON failed validation:\n"
                        f"{exc}\nReturn corrected JSON only.",
                    },
                ]
    raise LLMClientError(f"all models failed ({', '.join(settings.models)}): {type(last_error).__name__}: {last_error}")
