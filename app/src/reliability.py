"""Reliability: retries, a circuit breaker and safe fallbacks around the two things that can fail
underneath an agent: its tools (the EHR over MCP, the policy index) and its model.

    tools = ResilientTools(tools, tracer=tracer)            # same interface as CareFlowTools
    model = make_resilient_model(model, breaker, tracer)    # same (messages, specs) -> message callable

Rules:
- Retry only what is safe to repeat. Each tool's ToolDefinition says whether it is idempotent;
  book_appointment is not, so a failed booking is never sent twice (it could book two slots).
- A business answer is not a failure. {"ok": False, "error": "slot already booked"} is the tool working.
  Only an exception, or a result marked "unavailable", counts against the breaker and is retried.
- One breaker per backend. After `failure_threshold` failures in a row it opens and calls fail at once,
  without waiting on a dead service; after `cooldown_s` one trial call is let through (half-open), and
  its outcome closes or re-opens the breaker.
- The fallback is never a guess. A tool that cannot be reached returns {"ok": False, "unavailable":
  True, ...}; the agent's prompt already says to tell the person and flag staff when a tool errors. A
  model that cannot be reached raises ModelUnavailableError; the triage workflow turns that into a staff handoff.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .config import DEFAULT_BREAKER_COOLDOWN_S, DEFAULT_BREAKER_FAILURES, DEFAULT_TOOL_ATTEMPTS
from .llm_client import LAST_CALL
from .telemetry import MODEL_SPAN, NULL_TRACER, SpanKind, Tracer
from .tool_contracts import TOOLS_BY_NAME, Backend, ToolName


class BreakerState(StrEnum):
    """CircuitBreaker.state. Each member equals its string value; a tool span and /health record the
    plain string."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


CLOSED, OPEN, HALF_OPEN = BreakerState.CLOSED, BreakerState.OPEN, BreakerState.HALF_OPEN  # module-level aliases
MODEL_BREAKER = "model"  # the name of the model's circuit breaker (the tools' breakers are named by backend)
# Which backend each remote tool depends on (from its definition). Tools not listed run in-process
# against the local store and get no retries or breaker.
BACKENDS = {tool.name: tool.backend for tool in TOOLS_BY_NAME.values() if tool.backend != Backend.LOCAL}
# Instructions to the agent, returned as the "note" of an unavailable tool result. A failed booking has its
# own: it is not retried, so whether it went through is unknown.
UNAVAILABLE_NOTE = ("This service cannot be reached right now. Do not guess the answer: tell the person you cannot "
                    "check this at the moment and use flag_for_human so staff follow up.")
BOOKING_NOTE = ("The booking request failed in transit and was not retried, so it may or may not have been made. "
                "Do not tell the person it is booked or try another slot; use flag_for_human so staff confirm.")


class ModelUnavailableError(RuntimeError):
    """The model's circuit breaker is open: recent calls failed and the cool-down has not passed."""


class CircuitBreaker:
    """Closed -> open after `failure_threshold` failures in a row -> half-open `cooldown_s` seconds later ->
    closed again on a successful trial call, or open again on a failed one. Callers ask allow() before a
    call and report its outcome with record_success() or record_failure()."""

    def __init__(self, name: str, failure_threshold: int = DEFAULT_BREAKER_FAILURES,
                 cooldown_s: float = DEFAULT_BREAKER_COOLDOWN_S, now: Callable[[], float] = time.monotonic):
        self.name, self.failure_threshold, self.cooldown_s, self.now = name, failure_threshold, cooldown_s, now
        self.failures = 0  # failures in a row since the last success
        self.opened_at: float | None = None  # when it last opened, on the `now` clock; None while closed
        self._trial = False  # a half-open trial call is in flight
        self._lock = threading.RLock()  # the API's worker threads share one breaker per backend

    @property
    def state(self) -> BreakerState:
        """CLOSED, OPEN or HALF_OPEN. Half-open is derived from the clock, not stored."""
        with self._lock:
            if self.opened_at is None:
                return BreakerState.CLOSED
            cooled = self.now() - self.opened_at >= self.cooldown_s
            return BreakerState.HALF_OPEN if cooled else BreakerState.OPEN

    def allow(self) -> bool:
        """May a call go through now? In half-open, only one trial at a time."""
        with self._lock:
            state = self.state
            if state == BreakerState.CLOSED:
                return True
            if state == BreakerState.HALF_OPEN and not self._trial:
                self._trial = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.failures, self.opened_at, self._trial = 0, None, False

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self._trial or self.failures >= self.failure_threshold:
                self.opened_at = self.now()  # open, or re-open after a failed trial
            self._trial = False


@dataclass
class RetryPolicy:
    """Exponential backoff for idempotent tool calls: by default 0.2 s, then 0.4 s, between three tries."""

    attempts: int = DEFAULT_TOOL_ATTEMPTS  # total tries, the first call included
    backoff_s: float = 0.2  # pause before the first retry
    factor: float = 2.0  # each later pause is this many times the one before
    sleep: Callable[[float], None] = time.sleep  # replaced in tests

    def wait(self, attempt: int) -> None:
        """Pause before retry number `attempt` (1 = the first retry)."""
        self.sleep(self.backoff_s * self.factor ** (attempt - 1))


def _build_breakers() -> dict[str, CircuitBreaker]:
    """A fresh breaker per backend. Local tools have none: they never count as unreachable."""
    return {backend: CircuitBreaker(backend) for backend in set(BACKENDS.values())}


@dataclass
class ResilientTools:
    """Any tools object (CareFlowTools, McpRoutedTools) with retries, breakers, fallbacks and tool spans."""

    inner: Any
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    breakers: dict[str, CircuitBreaker] = field(default_factory=_build_breakers)
    tracer: Tracer = NULL_TRACER

    def __getattr__(self, name: str) -> Any:  # store, clock, patients, list_available_tools, ...
        return getattr(self.inner, name)

    def _build_unavailable_result(self, name: str, why: str) -> dict:
        """The fallback result: no data, and a note telling the agent to say so and flag staff."""
        note = BOOKING_NOTE if name == ToolName.BOOK_APPOINTMENT else UNAVAILABLE_NOTE
        return {"ok": False, "unavailable": True, "error": f"{name} is temporarily unavailable ({why})", "note": note}

    def call(self, name: str, arguments: dict | str | None = None, context: dict | None = None) -> dict:
        """Run one tool call inside a tool span. Returns the tool's own result, or the unavailable fallback
        when the breaker is open or every allowed attempt failed; it never raises for a tool failure."""
        definition = TOOLS_BY_NAME.get(name)
        backend = BACKENDS.get(name, Backend.LOCAL.value)
        breaker = self.breakers.get(backend)
        idempotent = bool(definition and definition.idempotent)
        with self.tracer.open_span(f"tool.{name}", kind=SpanKind.TOOL, tool=name, backend=backend,
                                   idempotent=idempotent, **(arguments if isinstance(arguments, dict) else {})) as span:
            if breaker is not None and not breaker.allow():
                span.set(ok=False, unavailable=True, breaker=breaker.state.value, attempts=0)
                return self._build_unavailable_result(name, "circuit open")
            attempts = max(1, self.retry.attempts) if idempotent else 1  # always at least one try
            why = "no attempt made"
            for attempt in range(1, attempts + 1):
                try:
                    result = self.inner.call(name, arguments, context=context)
                    failed = bool(result.get("unavailable"))
                    why = str(result.get("error") or "unavailable")
                except Exception as error:
                    result, failed, why = None, True, type(error).__name__
                if not failed:
                    if breaker is not None:
                        breaker.record_success()
                    scalars = {key: value for key, value in result.items() if not isinstance(value, (dict, list))}
                    span.set(**{**scalars, "attempts": attempt,
                                "breaker": breaker.state.value if breaker else None})
                    return result
                if breaker is not None:
                    breaker.record_failure()
                    if not breaker.allow():  # opened during this call: stop retrying a dead service
                        break
                if attempt < attempts:
                    self.retry.wait(attempt)
            # `why` is an exception class name or a tool's own error text. Only a class name is recorded and
            # passed on; free text is replaced by a fixed word.
            span.set(ok=False, unavailable=True, attempts=attempt,
                     breaker=breaker.state.value if breaker else None,
                     error_type=why if why.isidentifier() else "unavailable")
            return self._build_unavailable_result(name, why if why.isidentifier() else "no response")


def _get_usage() -> dict:
    """The model that actually answered and its token counts, when the client recorded them."""
    return {key: value for key, value in (LAST_CALL.get() or {}).items() if value is not None}


def _call_model(breaker: CircuitBreaker, tracer: Tracer, span_attrs: Callable[[], dict],
                invoke: Callable[[], Any], describe: Callable[[Any], dict] | None = None) -> Any:
    """One model call behind `breaker`, inside a generation span. Raises ModelUnavailableError without calling
    when the breaker is open; any exception from `invoke` counts as a failure and is re-raised.
    `span_attrs` gives the span's opening attributes (evaluated only once the breaker lets the call
    through); `describe` turns the result into extra ones."""
    if not breaker.allow():
        raise ModelUnavailableError(f"model circuit is {breaker.state}")
    with tracer.open_span(MODEL_SPAN, kind=SpanKind.GENERATION, **span_attrs()) as span:
        LAST_CALL.set(None)  # so _get_usage() reports this call, not an earlier one
        try:
            result = invoke()
        except Exception:
            breaker.record_failure()
            raise
        breaker.record_success()
        span.set(**(describe(result) if describe else {}), **_get_usage())
        return result


def _list_tool_call_names(message: Any) -> dict:
    """Span attributes for a tool-calling reply: the names of the tools the model asked for."""
    return {"tool_calls": [tool_call.function.name for tool_call in (getattr(message, "tool_calls", None) or [])]}


def make_resilient_model(model: Callable[[list[dict], list[dict]], Any], breaker: CircuitBreaker | None = None,
                         tracer: Tracer = NULL_TRACER, name: str = "model") -> Callable[[list[dict], list[dict]], Any]:
    """The model callable with a breaker and a generation span. The provider client already retries and
    falls back across models (llm_client); the breaker adds failing fast once that has not been enough.
    The breaker is exposed as the returned callable's `breaker` attribute."""
    breaker = breaker or CircuitBreaker(MODEL_BREAKER)

    def call(messages: list[dict], specs: list[dict]) -> Any:
        return _call_model(
            breaker, tracer, lambda: {"model": name, "messages": len(messages), "tools_offered": len(specs)},
            lambda: model(messages, specs), _list_tool_call_names)

    call.breaker = breaker  # type: ignore[attr-defined]
    return call


def guard_model_call(function: Callable[..., Any], breaker: CircuitBreaker, tracer: Tracer = NULL_TRACER,
                     name: str = "model") -> Callable[..., Any]:
    """A structured model call (the intake parse, the Clinical Safety Reviewer) behind the same breaker and
    with the same generation span as the tool-calling model. When the breaker is open it raises
    ModelUnavailableError at once; both callers already fail closed on any error."""

    def call(*arguments: Any, **keyword_arguments: Any) -> Any:
        return _call_model(breaker, tracer, lambda: {"model": name},
                           lambda: function(*arguments, **keyword_arguments))

    return call
