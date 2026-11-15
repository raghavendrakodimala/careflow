"""Provider-agnostic structured-output client built on LiteLLM."""

from __future__ import annotations

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


def complete_structured(
    messages: list[dict],
    response_model: type[T],
    settings: Settings | None = None,
    validation_retries: int = 2,
) -> T:
    """Ask the model for JSON matching `response_model`; return a validated object.

    - Transient provider errors: LiteLLM retries, then the next fallback model is tried.
    - Invalid output: the validation error is sent back and the model tries again.
    """
    settings = settings or get_settings()
    last_error: Exception | None = None

    for model in settings.models:
        convo = list(messages)
        for _ in range(validation_retries + 1):
            try:
                resp = litellm.completion(
                    model=model,
                    messages=convo,
                    response_format=response_model,
                    temperature=0,
                    timeout=settings.request_timeout_s,
                    num_retries=settings.transport_retries,
                    **settings.provider_kwargs(model),
                )
            except _TRANSIENT as exc:
                last_error = exc
                break  # next model
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
