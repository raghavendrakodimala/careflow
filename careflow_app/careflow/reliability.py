"""M7 reliability: retries, a circuit breaker and safe fallbacks around the two things that can fail
underneath an agent: its tools (the EHR over MCP, the policy index) and its model.

    tools = ResilientTools(tools, tracer=tracer)       # same interface as CareFlowTools
    model = resilient_model(model, breaker, tracer)    # same (messages, specs) -> message callable

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
  model that cannot be reached raises ModelUnavailable; the triage workflow turns that into a staff handoff.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm_client import LAST_CALL
from .telemetry import NULL_TRACER, Tracer
from .tool_contracts import TOOLS_BY_NAME

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"
# Which backend a tool depends on. Tools not listed run in-process against the local store.
BACKENDS = {
    "find_appointment_slots": "ehr", "book_appointment": "ehr", "check_eligibility": "ehr", "check_preauth": "ehr",
    "find_patient": "ehr", "search_policies": "policy_index",
}
UNAVAILABLE_NOTE = ("This service cannot be reached right now. Do not guess the answer: tell the person you cannot "
                    "check this at the moment and use flag_for_human so staff follow up.")
BOOKING_NOTE = ("The booking request failed in transit and was not retried, so it may or may not have been made. "
                "Do not tell the person it is booked or try another slot; use flag_for_human so staff confirm.")


class ModelUnavailable(RuntimeError):
    """The model's circuit breaker is open: recent calls failed and the cool-down has not passed."""


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, cooldown_s: float = 30.0,
                 now: Callable[[], float] = time.monotonic):
        self.name, self.failure_threshold, self.cooldown_s, self.now = name, failure_threshold, cooldown_s, now
        self.failures = 0
        self.opened_at: float | None = None
        self._trial = False  # a half-open trial call is in flight
        self._lock = threading.RLock()  # the API's worker threads share one breaker per backend

    @property
    def state(self) -> str:
        with self._lock:
            if self.opened_at is None:
                return CLOSED
            return HALF_OPEN if self.now() - self.opened_at >= self.cooldown_s else OPEN

    def allow(self) -> bool:
        """May a call go through now? In half-open, only one trial at a time."""
        with self._lock:
            state = self.state
            if state == CLOSED:
                return True
            if state == HALF_OPEN and not self._trial:
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
    attempts: int = 3
    backoff_s: float = 0.2
    factor: float = 2.0
    sleep: Callable[[float], None] = time.sleep

    def wait(self, attempt: int) -> None:
        """Pause before retry number `attempt` (1 = the first retry)."""
        self.sleep(self.backoff_s * self.factor ** (attempt - 1))


def _breakers() -> dict[str, CircuitBreaker]:
    return {backend: CircuitBreaker(backend) for backend in set(BACKENDS.values())}


@dataclass
class ResilientTools:
    """Any tools object (CareFlowTools, McpRoutedTools) with retries, breakers, fallbacks and tool spans."""

    inner: Any
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    breakers: dict[str, CircuitBreaker] = field(default_factory=_breakers)
    tracer: Tracer = NULL_TRACER

    def __getattr__(self, name: str) -> Any:  # store, clock, patients, available_tools, ...
        return getattr(self.inner, name)

    def _unavailable(self, name: str, why: str) -> dict:
        note = BOOKING_NOTE if name == "book_appointment" else UNAVAILABLE_NOTE
        return {"ok": False, "unavailable": True, "error": f"{name} is temporarily unavailable ({why})", "note": note}

    def call(self, name: str, arguments: dict | str | None = None, context: dict | None = None) -> dict:
        definition = TOOLS_BY_NAME.get(name)
        backend = BACKENDS.get(name, "local")
        breaker = self.breakers.get(backend)
        idempotent = bool(definition and definition.idempotent)
        with self.tracer.span(f"tool.{name}", kind="tool", tool=name, backend=backend, idempotent=idempotent,
                              **(arguments if isinstance(arguments, dict) else {})) as span:
            if breaker is not None and not breaker.allow():
                span.set(ok=False, unavailable=True, breaker=breaker.state, attempts=0)
                return self._unavailable(name, "circuit open")
            attempts = self.retry.attempts if idempotent else 1
            why = "no attempt made"
            for attempt in range(1, attempts + 1):
                try:
                    result = self.inner.call(name, arguments, context=context)
                    failed = bool(result.get("unavailable"))
                    why = str(result.get("error") or "unavailable")
                except Exception as exc:
                    result, failed, why = None, True, type(exc).__name__
                if not failed:
                    if breaker is not None:
                        breaker.record_success()
                    span.set(attempts=attempt, breaker=breaker.state if breaker else None,
                             **{k: v for k, v in result.items() if not isinstance(v, (dict, list))})
                    return result
                if breaker is not None:
                    breaker.record_failure()
                    if not breaker.allow():  # opened during this call: stop retrying a dead service
                        break
                if attempt < attempts:
                    self.retry.wait(attempt)
            span.set(ok=False, unavailable=True, attempts=attempt, breaker=breaker.state if breaker else None,
                     error_type=why if why.isidentifier() else "unavailable")
            return self._unavailable(name, why if why.isidentifier() else "no response")


def resilient_model(model: Callable[[list[dict], list[dict]], Any], breaker: CircuitBreaker | None = None,
                    tracer: Tracer = NULL_TRACER, name: str = "model") -> Callable[[list[dict], list[dict]], Any]:
    """The model callable with a breaker and a generation span. The provider client already retries and
    falls back across models (llm_client); the breaker adds failing fast once that has not been enough."""
    breaker = breaker or CircuitBreaker("model")

    def call(messages: list[dict], specs: list[dict]) -> Any:
        if not breaker.allow():
            raise ModelUnavailable(f"model circuit is {breaker.state}")
        with tracer.span("model.complete", kind="generation", model=name, messages=len(messages),
                         tools_offered=len(specs)) as span:
            LAST_CALL.set(None)
            try:
                message = model(messages, specs)
            except Exception:
                breaker.record_failure()
                raise
            breaker.record_success()
            span.set(tool_calls=[c.function.name for c in (getattr(message, "tool_calls", None) or [])], **_used())
            return message

    call.breaker = breaker  # type: ignore[attr-defined]
    return call


def _used() -> dict:
    """The model that actually answered and its token counts, when the client recorded them."""
    return {k: v for k, v in (LAST_CALL.get() or {}).items() if v is not None}


def guarded(fn: Callable[..., Any], breaker: CircuitBreaker, tracer: Tracer = NULL_TRACER,
            name: str = "model") -> Callable[..., Any]:
    """A structured model call (the intake parse, the Clinical Safety Reviewer) behind the same breaker and
    with the same generation span as the tool-calling model. When the breaker is open it raises
    ModelUnavailable at once; both callers already fail closed on any error."""

    def call(*args: Any, **kwargs: Any) -> Any:
        if not breaker.allow():
            raise ModelUnavailable(f"model circuit is {breaker.state}")
        with tracer.span("model.complete", kind="generation", model=name) as span:
            LAST_CALL.set(None)
            try:
                result = fn(*args, **kwargs)
            except Exception:
                breaker.record_failure()
                raise
            breaker.record_success()
            span.set(**_used())
            return result

    return call
