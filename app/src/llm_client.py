"""Provider-agnostic structured-output client built on LiteLLM."""

from __future__ import annotations

import contextvars
import json
import re
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings
from .state import Role

ParsedModel = TypeVar("ParsedModel", bound=BaseModel)

# Failures worth moving to the next model for (LiteLLM has already retried them `transport_retries` times);
# anything else is a real bug and propagates.
_TRANSIENT = (
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.APIConnectionError,
)


class LLMClientError(RuntimeError):
    """No configured model produced a usable reply: each failed transiently or, for structured output,
    kept returning JSON that did not validate."""


def _make_all_failed_error(settings: Settings, last_error: Exception | None) -> LLMClientError:
    return LLMClientError(
        f"all models failed ({', '.join(settings.models)}): {type(last_error).__name__}: {last_error}"
    )


def _strip_fences(text: str) -> str:
    """The JSON inside a Markdown code fence (```json ... ```), which some models add despite the prompt."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


# Words in a provider's 400 message that mean "the response schema was the problem", not the request.
_SCHEMA_WORDS = re.compile(r"response_format|response_schema|json_schema|structured output|schema", re.IGNORECASE)


def _is_schema_rejected(error: Exception) -> bool:
    """The provider or model cannot take a native response schema (several OpenRouter and Ollama models)."""
    if isinstance(error, litellm.UnsupportedParamsError):
        return True
    return isinstance(error, litellm.BadRequestError) and bool(_SCHEMA_WORDS.search(str(error)))


def _add_schema_to_prompt(messages: list[dict], response_model: type[BaseModel]) -> list[dict]:
    """Fallback: describe the JSON schema in the prompt instead; the reply is still validated by Pydantic."""
    schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    note = {
        "role": Role.SYSTEM.value,
        "content": f"Reply with one JSON object only, no prose, matching this JSON Schema:\n{schema}",
    }
    return [*messages, note]


# What the most recent call in this context used: {"model", "input_tokens", "output_tokens"}. Set here,
# read by the tracing wrappers (reliability.py), so the return values of the two functions below stay as
# they are. The model is the one that answered, which is a fallback when the first one failed.
LAST_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("careflow_last_call", default=None)


def _note_call(model: str, response: Any) -> None:
    """Record the answering model and its token usage in LAST_CALL (None where the provider reports none)."""
    usage = getattr(response, "usage", None)
    LAST_CALL.set({"model": model, "input_tokens": getattr(usage, "prompt_tokens", None),
                   "output_tokens": getattr(usage, "completion_tokens", None)})


def complete_with_tools(messages: list[dict], tools: list[dict], settings: Settings | None = None):
    """One tool-calling turn. Returns the assistant message (may hold tool_calls).

    Same provider rules as complete_structured: LiteLLM retries transient errors, then the next
    fallback model is tried. The model is chosen by config only.
    """
    settings = settings or get_settings()
    last_error: Exception | None = None
    for model in settings.models:
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0,
                timeout=settings.request_timeout_s,
                num_retries=settings.transport_retries,
                **settings.get_provider_options(model),
            )
        except _TRANSIENT as error:
            last_error = error
            continue
        _note_call(model, response)
        return response.choices[0].message
    raise _make_all_failed_error(settings, last_error)


def complete_structured(
    messages: list[dict],
    response_model: type[ParsedModel],
    settings: Settings | None = None,
    validation_retries: int = 2,
) -> ParsedModel:
    """Ask the model for JSON matching `response_model`; return a validated object.

    - Transient provider errors: LiteLLM retries, then the next fallback model is tried.
    - Native schema rejected by the provider: the same model is asked again with the schema in the prompt.
    - Invalid output: the validation error is sent back and the model tries again, up to
      `validation_retries` times per model; then the next model is tried.
    Raises LLMClientError when every model has failed.
    """
    settings = settings or get_settings()
    last_error: Exception | None = None

    for model in settings.models:
        conversation = list(messages)
        native = True  # pass the Pydantic schema as response_format until the provider says it cannot
        attempts = 0  # replies received from this model
        while attempts <= validation_retries:
            schema_argument = {"response_format": response_model} if native else {}
            try:
                response = litellm.completion(
                    model=model,
                    messages=conversation,
                    temperature=0,
                    timeout=settings.request_timeout_s,
                    num_retries=settings.transport_retries,
                    **schema_argument,
                    **settings.get_provider_options(model),
                )
            except _TRANSIENT as error:
                last_error = error
                break  # next model
            except litellm.BadRequestError as error:
                if not (native and _is_schema_rejected(error)):
                    raise
                last_error = error
                native = False  # same model, schema in the prompt; does not use up a validation retry
                conversation = _add_schema_to_prompt(conversation, response_model)
                continue
            attempts += 1
            _note_call(model, response)
            content = response.choices[0].message.content or ""
            try:
                return response_model.model_validate_json(_strip_fences(content))
            except ValidationError as error:
                last_error = error
                conversation = [
                    *conversation,
                    {"role": Role.ASSISTANT.value, "content": content},
                    {
                        "role": Role.USER.value,
                        "content": "That JSON failed validation:\n"
                        f"{error}\nReturn corrected JSON only.",
                    },
                ]
    raise _make_all_failed_error(settings, last_error)
