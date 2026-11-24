"""Tracing: one trace per request, with a span for each workflow step, agent, tool call and model call.

    tracer = build_tracer(settings)            # CAREFLOW_TRACING = off | jsonl | langfuse
    with tracer.open_trace(thread_id, "triage"):
        with tracer.open_span("tool.check_eligibility", kind="tool", patient_ref=ref) as span:
            ...
            span.set(ok=True)

Sinks: JSONL (out/traces.jsonl, one line per finished span; always available) and LangFuse (when
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set). A sink that fails never breaks a request.

Privacy (data-handling standard, minimum necessary): traces leave the process, so they hold structure,
not content. Only attribute names in SAFE_KEYS are recorded; everything else is dropped. That keeps out
the patient's message, the model's text, names, birth dates, member numbers and free-text reasons.
String values that are kept are still passed through `scrub`, which blanks member numbers and dates.
patient_ref is kept: it is the clinic's pseudonymous record ID and is what staff search a trace by.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .config import OUT_DIR, Settings, TracingMode, get_settings
from .schemas import MEMBER_NUMBER

TRACE_FILE = OUT_DIR / "traces.jsonl"


class SpanKind(StrEnum):
    """What a span times. A recorded span holds the plain string."""

    TRACE = "trace"          # the root span of one request
    SPAN = "span"            # a workflow step, or anything else
    AGENT = "agent"
    TOOL = "tool"
    GENERATION = "generation"  # a model call
    GUARDRAIL = "guardrail"


class SpanStatus(StrEnum):
    """How a span ended. A recorded span holds the plain string."""

    OK = "ok"
    ERROR = "error"
    PAUSED = "paused"  # stopped at the human-review pause: control flow, not an error


# Span names that more than one module opens or reads (scripts/trace_summary.py counts by them).
MODEL_SPAN = "model.complete"  # every model call
INPUT_GUARD_SPAN = "guard.input"  # the input gate of the chat path
AGENT_SPAN_PREFIX = "agent."  # followed by the agent's name
REVIEWER_SPAN = f"{AGENT_SPAN_PREFIX}clinical_safety_reviewer"  # every Clinical Safety Reviewer check

# Attribute names that may be recorded. Identifiers the clinic itself issues, codes, counts and outcomes.
SAFE_KEYS = frozenset({
    # request and workflow
    "thread_id", "channel", "route", "band", "case_status", "step", "reasons_count", "review_id", "flag_id",
    "action", "reviewer_id", "repeat", "topic", "safe", "category", "vetoed", "agents", "agent",
    # tools
    "tool", "backend", "ok", "found", "count", "abstained", "duplicate", "unavailable", "attempts", "idempotent",
    "patient_ref", "referral_id", "slot_id", "appointment_id", "procedure_code", "plan_code", "specialty", "mode",
    "limit", "urgency", "clinical", "telehealth", "self_pay", "eligibility_status", "covered", "preauth_required",
    "payment", "breaker",
    # model
    "model", "messages", "tools_offered", "tool_calls", "input_tokens", "output_tokens",
    # errors
    "error_type",
})
_MEMBER = re.compile(rf"\b{MEMBER_NUMBER}\b", re.I)  # an insurance member number: three letters, 6 to 10 digits
# A date that could be a birth date: ISO (2001-04-09) or day/month/year in either order with / . or -
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")
# Exception class names, matched by name so this module need not import LangGraph. Its human-review pause
# is control flow, not an error.
_PAUSE = ("GraphInterrupt",)
_MAX_TEXT = 200  # characters kept of a string value
_MAX_ITEMS = 50  # items kept of a list, tuple or set


def scrub(value: Any) -> Any:
    """A value that is safe to store: member numbers and dates blanked, long text cut, containers walked.
    A nested dict keeps only its SAFE_KEYS; an object of any other type is replaced by its type name."""
    if isinstance(value, str):
        return _DATE.sub("[date]", _MEMBER.sub("[member]", value))[:_MAX_TEXT]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [scrub(item) for item in list(value)[:_MAX_ITEMS]]
    if isinstance(value, dict):
        return {str(key): scrub(item) for key, item in value.items() if key in SAFE_KEYS}
    return type(value).__name__


def filter_safe_attributes(attributes: dict) -> dict:
    """The attributes that may be recorded: SAFE_KEYS only, values scrubbed, None values left out."""
    return {key: scrub(value) for key, value in attributes.items() if key in SAFE_KEYS and value is not None}


@dataclass
class Span:
    """One timed step of a request. `attributes` only ever holds values that went through filter_safe_attributes."""

    trace_id: str
    span_id: str
    parent_id: str | None  # None for the root span of a trace
    name: str
    kind: str  # a SpanKind value
    started: float  # seconds, from the tracer's clock (time.time by default)
    ended: float | None = None
    status: str = SpanStatus.OK.value  # a SpanStatus value
    attributes: dict = field(default_factory=dict)
    handle: Any = None  # a sink's own object for this span (LangFuse)

    def set(self, **attributes) -> None:
        """Add attributes while the span is open; names outside SAFE_KEYS are dropped silently."""
        self.attributes.update(filter_safe_attributes(attributes))

    def to_dict(self) -> dict:
        """The span as one flat record (a JSONL line): the fixed fields, then the attributes."""
        return {"trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id, "name": self.name,
                "kind": self.kind, "status": self.status, "started": round(self.started, 3),
                "duration_ms": round(((self.ended or self.started) - self.started) * 1000, 1), **self.attributes}


class Sink(Protocol):
    """Where spans go. on_start is called when a span opens, on_end once it is finished; either may raise
    (the tracer swallows it)."""

    def on_start(self, span: Span, parent: Span | None) -> None: ...
    def on_end(self, span: Span) -> None: ...
    def close(self) -> None: ...


class MemorySink:
    """Keeps finished spans in a list (tests)."""

    def __init__(self):
        self.spans: list[dict] = []

    def on_start(self, span: Span, parent: Span | None) -> None:
        pass

    def on_end(self, span: Span) -> None:
        self.spans.append(span.to_dict())

    def close(self) -> None:
        pass


class JsonlSink:
    """Appends each finished span as one JSON line. The file is opened per span, so it can be read or
    rotated while the process runs."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or TRACE_FILE)
        self._lock = threading.Lock()  # one writer at a time: lines from concurrent requests never interleave

    def on_start(self, span: Span, parent: Span | None) -> None:
        pass

    def on_end(self, span: Span) -> None:
        line = json.dumps(span.to_dict(), ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(line + "\n")

    def close(self) -> None:
        pass


class LangfuseSink:
    """Mirrors each span into LangFuse as it opens and closes. `client` is a langfuse.Langfuse (or a fake)."""

    # Span kind -> LangFuse observation type; any other kind is sent as a plain "span".
    KINDS: ClassVar[dict[str, str]] = {
        kind.value: kind.value for kind in (SpanKind.TOOL, SpanKind.GENERATION, SpanKind.AGENT, SpanKind.GUARDRAIL)}

    def __init__(self, client: Any):
        self.client = client

    def on_start(self, span: Span, parent: Span | None) -> None:
        owner = parent.handle if parent is not None and parent.handle is not None else self.client
        span.handle = owner.start_observation(name=span.name, as_type=self.KINDS.get(span.kind, SpanKind.SPAN.value),
                                              metadata={"trace_id": span.trace_id, **span.attributes})

    def on_end(self, span: Span) -> None:
        if span.handle is None:
            return
        span.handle.update(metadata={"trace_id": span.trace_id, **span.attributes},
                           level="ERROR" if span.status == SpanStatus.ERROR else "DEFAULT")
        span.handle.end()
        if span.parent_id is None:  # the request is over: send its trace now
            self.client.flush()

    def close(self) -> None:
        self.client.flush()


class Tracer:
    """Opens spans and hands them to the sinks. With no sinks it is disabled: spans are still yielded, so
    callers need no special case, but nothing is recorded or nested."""

    def __init__(self, sinks: list[Sink] | None = None, now: Callable[[], float] = time.time):
        self.sinks, self.now = list(sinks or []), now
        # The open span of the current thread or task; a new span becomes its child.
        self._current: ContextVar[Span | None] = ContextVar("careflow_span", default=None)

    @property
    def enabled(self) -> bool:
        return bool(self.sinks)

    def _emit(self, method: str, *arguments) -> None:
        """Call `method` on every sink, ignoring a sink that fails."""
        for sink in self.sinks:
            with contextlib.suppress(Exception):  # tracing must never break a request
                getattr(sink, method)(*arguments)

    @contextmanager
    def open_span(self, name: str, kind: str = SpanKind.SPAN, trace_id: str | None = None,
                  **attributes) -> Iterator[Span]:
        """Open a span under the current one (or start a new trace if there is none). An exception passing
        through marks it "error" and records the exception's type name only, then is re-raised."""
        parent = self._current.get()
        span = Span(
            trace_id=trace_id or (parent.trace_id if parent else uuid.uuid4().hex), span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None, name=name,
            kind=str(kind),  # the plain string, also when the caller passed a SpanKind member
            started=self.now(), attributes=filter_safe_attributes(attributes),
        )
        if not self.enabled:
            yield span
            return
        self._emit("on_start", span, parent)
        token = self._current.set(span)
        try:
            yield span
        except BaseException as error:
            if type(error).__name__ in _PAUSE:
                span.status = SpanStatus.PAUSED.value
            else:
                span.status = SpanStatus.ERROR.value
                span.attributes["error_type"] = type(error).__name__
            raise
        finally:
            self._current.reset(token)
            span.ended = self.now()
            self._emit("on_end", span)

    def open_trace(self, trace_id: str, name: str, **attributes):
        """The root span of one request. Inside an open trace it is an ordinary child span."""
        if self._current.get() is not None:
            return self.open_span(name, **attributes)
        return self.open_span(name, kind=SpanKind.TRACE, trace_id=trace_id, thread_id=trace_id, **attributes)

    def close(self) -> None:
        self._emit("close")


NULL_TRACER = Tracer()  # the disabled tracer: the default wherever a tracer is optional


def build_tracer(settings: Settings | None = None, path: Path | str | None = None) -> Tracer:
    """The tracer the settings ask for. `langfuse` without keys (or without the package) falls back to JSONL."""
    settings = settings or get_settings()
    mode = (settings.careflow_tracing or TracingMode.OFF).lower()
    if mode == TracingMode.OFF:
        return NULL_TRACER
    sinks: list[Sink] = [JsonlSink(path)]
    if mode == TracingMode.LANGFUSE and settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse

            sinks.append(LangfuseSink(Langfuse(
                public_key=settings.langfuse_public_key.get_secret_value(),
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                host=settings.langfuse_host or None,
            )))
        except Exception:
            pass  # JSONL still records everything
    return Tracer(sinks)
