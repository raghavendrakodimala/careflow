"""M7 tracing: one trace per request, with a span for each workflow step, agent, tool call and model call.

    tracer = default_tracer(settings)          # CAREFLOW_TRACING = off | jsonl | langfuse
    with tracer.trace(thread_id, "triage"):
        with tracer.span("tool.check_eligibility", kind="tool", patient_ref=ref) as span:
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

import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .config import OUT_DIR, Settings, get_settings

TRACE_FILE = OUT_DIR / "traces.jsonl"

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
_MEMBER = re.compile(r"\b[A-Z]{3}\d{6,10}\b", re.I)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")
_PAUSE = ("GraphInterrupt",)  # LangGraph's human-review pause is control flow, not an error


def scrub(value: Any) -> Any:
    """A value that is safe to store: member numbers and dates blanked, long text cut, containers walked."""
    if isinstance(value, str):
        return _DATE.sub("[date]", _MEMBER.sub("[member]", value))[:200]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [scrub(v) for v in list(value)[:50]]
    if isinstance(value, dict):
        return {str(k): scrub(v) for k, v in value.items() if k in SAFE_KEYS}
    return type(value).__name__


def safe_attrs(attrs: dict) -> dict:
    return {k: scrub(v) for k, v in attrs.items() if k in SAFE_KEYS and v is not None}


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str  # trace | span | agent | tool | generation | guardrail
    started: float
    ended: float | None = None
    status: str = "ok"  # ok | error | paused
    attrs: dict = field(default_factory=dict)
    handle: Any = None  # a sink's own object for this span (LangFuse)

    def set(self, **attrs) -> None:
        self.attrs.update(safe_attrs(attrs))

    def as_dict(self) -> dict:
        return {"trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id, "name": self.name,
                "kind": self.kind, "status": self.status, "started": round(self.started, 3),
                "duration_ms": round(((self.ended or self.started) - self.started) * 1000, 1), **self.attrs}


class Sink(Protocol):
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
        self.spans.append(span.as_dict())

    def close(self) -> None:
        pass


class JsonlSink:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or TRACE_FILE)
        self._lock = threading.Lock()

    def on_start(self, span: Span, parent: Span | None) -> None:
        pass

    def on_end(self, span: Span) -> None:
        line = json.dumps(span.as_dict(), ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def close(self) -> None:
        pass


class LangfuseSink:
    """Mirrors each span into LangFuse as it opens and closes. `client` is a langfuse.Langfuse (or a fake)."""

    KINDS = {"tool": "tool", "generation": "generation", "agent": "agent", "guardrail": "guardrail"}

    def __init__(self, client: Any):
        self.client = client

    def on_start(self, span: Span, parent: Span | None) -> None:
        owner = parent.handle if parent is not None and parent.handle is not None else self.client
        span.handle = owner.start_observation(name=span.name, as_type=self.KINDS.get(span.kind, "span"),
                                              metadata={"trace_id": span.trace_id, **span.attrs})

    def on_end(self, span: Span) -> None:
        if span.handle is None:
            return
        span.handle.update(metadata={"trace_id": span.trace_id, **span.attrs},
                           level="ERROR" if span.status == "error" else "DEFAULT")
        span.handle.end()
        if span.parent_id is None:
            self.client.flush()

    def close(self) -> None:
        self.client.flush()


class Tracer:
    def __init__(self, sinks: list[Sink] | None = None, now: Callable[[], float] = time.time):
        self.sinks, self.now = list(sinks or []), now
        self._current: ContextVar[Span | None] = ContextVar("careflow_span", default=None)

    @property
    def enabled(self) -> bool:
        return bool(self.sinks)

    def _emit(self, method: str, *args) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, method)(*args)
            except Exception:  # tracing must never break a request
                pass

    @contextmanager
    def span(self, name: str, kind: str = "span", trace_id: str | None = None, **attrs) -> Iterator[Span]:
        parent = self._current.get()
        span = Span(
            trace_id=trace_id or (parent.trace_id if parent else uuid.uuid4().hex), span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None, name=name, kind=kind, started=self.now(),
            attrs=safe_attrs(attrs),
        )
        if not self.enabled:
            yield span
            return
        self._emit("on_start", span, parent)
        token = self._current.set(span)
        try:
            yield span
        except BaseException as exc:
            if type(exc).__name__ in _PAUSE:
                span.status = "paused"
            else:
                span.status = "error"
                span.attrs["error_type"] = type(exc).__name__
            raise
        finally:
            self._current.reset(token)
            span.ended = self.now()
            self._emit("on_end", span)

    def trace(self, trace_id: str, name: str, **attrs):
        """The root span of one request. Inside an open trace it is an ordinary child span."""
        if self._current.get() is not None:
            return self.span(name, **attrs)
        return self.span(name, kind="trace", trace_id=trace_id, thread_id=trace_id, **attrs)

    def close(self) -> None:
        self._emit("close")


NULL_TRACER = Tracer()


def default_tracer(settings: Settings | None = None, path: Path | str | None = None) -> Tracer:
    """The tracer the settings ask for. `langfuse` without keys (or without the package) falls back to JSONL."""
    settings = settings or get_settings()
    mode = (settings.careflow_tracing or "off").lower()
    if mode == "off":
        return NULL_TRACER
    sinks: list[Sink] = [JsonlSink(path)]
    if mode == "langfuse" and settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse

            sinks.append(LangfuseSink(Langfuse(
                public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host or None,
            )))
        except Exception:
            pass  # JSONL still records everything
    return Tracer(sinks)
