"""The CareFlow HTTP service (FastAPI). Thin adapters over the workflow and the agent team.

    uvicorn src.api.app:app            (from app/)

    GET  /health                             store and checkpointer reachable
    POST /v1/triage                          run one incoming message through the triage workflow
    GET  /v1/reviews                         the staff queue: cases waiting for a decision
    GET  /v1/reviews/{thread_id}             what staff are asked to decide for one case
    POST /v1/reviews/{thread_id}/decision    record the decision and finish the case
    POST /v1/chat                            one conversational turn with the agent team

Resources are opened once at start-up (runtime store, session checkpointer, the EHR MCP server, the
policy index if it has been built, the tracer, and the local input-gate classifier, which is loaded and
warmed so the first message is not the slow one) and closed at shutdown. Tests pass their own through
create_app(), so no model, index or child process is needed.

Access: if CAREFLOW_API_KEY is set, every /v1 request must send it as X-API-Key. This service has no
user login. `patient_ref` in a request is trusted as already verified by the calling channel (a patient
portal session, a staff console); a caller without one is identified from full name plus member number.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..agent import PatientMismatchError, close_checkpointer, open_session_checkpointer
from ..agents import Reviewer, run_team
from ..config import Settings, get_settings
from ..guard_model import build_classifier
from ..guardrails import Classifier
from ..schemas import DEFAULT_CHANNEL, Channel
from ..state import build_thread_config
from ..store import RuntimeStore
from ..tool_contracts import CONTACT_ID, ToolName
from ..tools import CareFlowTools
from ..workflows import (
    AlreadyTriagedError,
    NotAwaitingReviewError,
    ReviewDecision,
    TriageDependencies,
    get_pending_review,
    resume_triage,
    start_triage,
)

# Chat sessions and triage cases share one checkpoint store with different state shapes. A chat thread
# is stored under this prefix and a triage or review request may not use it, so neither can read the other.
log = logging.getLogger(__name__)

CHAT_PREFIX = "chat:"
# The 503 detail when the chat path fails (the model or a dependency is down).
UNAVAILABLE = ("The assistant cannot answer right now. Your message has not been lost; please try again shortly or "
               "call the clinic.")
# The staff flag raised when a chat turn fails. It names the kind of failure and nothing from the request.
UNANSWERED_REASON = "The assistant could not answer a chat message ({error}); the patient was asked to try again or call."


def _flag_unanswered_chat(tools: Any, thread_id: str, patient_ref: str | None, error: Exception) -> None:
    """Tell staff that a chat message went unanswered, so a failed turn is a handoff and not only an error.

    One flag per conversation and kind of failure (the store de-duplicates within a contact), so a caller
    who retries does not fill the queue. Best effort: if the flag cannot be raised the 503 still goes out,
    and only the type of the second failure is logged.
    """
    reason = UNANSWERED_REASON.format(error=type(error).__name__)
    context = {CONTACT_ID: CHAT_PREFIX + thread_id}
    try:
        result = tools.call(ToolName.FLAG_FOR_HUMAN.value, {"reason": reason, "patient_ref": patient_ref}, context=context)
        if not result.get("ok") and patient_ref:  # an unknown reference must not stop the handoff
            tools.call(ToolName.FLAG_FOR_HUMAN.value, {"reason": reason}, context=context)
    except Exception as flag_error:
        log.error("staff flag for failed chat on thread %s was not raised: %s", thread_id, type(flag_error).__name__)


# Request limits. A thread_id becomes a checkpoint key, so its characters are restricted.
MAX_MESSAGE_CHARS = 8000
MAX_THREAD_ID_CHARS = 64
THREAD_ID_PATTERN = r"^[A-Za-z0-9_.:-]+$"


def _reject_chat_thread(thread_id: str | None) -> None:
    """Refuse (422) a triage or review request whose thread_id is in the chat sessions' namespace."""
    if thread_id and thread_id.lower().startswith(CHAT_PREFIX):
        raise HTTPException(status_code=422, detail=f"thread_id may not start with {CHAT_PREFIX!r}")


# The request models carry no docstrings: pydantic would publish them as schema descriptions in /openapi.json.
# Shared settings: an unknown field is a 422, and surrounding whitespace is stripped from strings.
class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TriageRequest(_Body):
    raw_text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    channel: Channel = DEFAULT_CHANNEL
    patient_ref: str | None = Field(default=None, max_length=32)  # trusted as verified by the calling channel
    received_at: str | None = Field(default=None, max_length=40)  # ISO timestamp; the server's clock if omitted
    thread_id: str | None = Field(default=None, min_length=1, max_length=MAX_THREAD_ID_CHARS, pattern=THREAD_ID_PATTERN)


class ChatRequest(_Body):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    channel: Channel = DEFAULT_CHANNEL
    patient_ref: str | None = Field(default=None, max_length=32)  # trusted as verified by the calling channel
    thread_id: str | None = Field(default=None, min_length=1, max_length=MAX_THREAD_ID_CHARS, pattern=THREAD_ID_PATTERN)


@dataclass
class Resources:
    """What the service holds open for its lifetime; one instance per app, on app.state.resources."""

    dependencies: TriageDependencies
    checkpointer: Any
    classifier: Classifier | None = None
    settings: Settings | None = None
    owned: bool = False  # opened by the app itself, so closed by it too

    @property
    def reviewer(self) -> Reviewer | None:
        return self.dependencies.review

    def close(self) -> None:
        """Close everything, tolerating an MCP client that fails to stop."""
        tools = self.dependencies.tools
        # The tools may be an McpRoutedTools, directly or as the `inner` of the resilience wrapper.
        for closer in (getattr(tools, "close", None), getattr(getattr(tools, "inner", None), "close", None)):
            if closer:  # the MCP client, if the EHR tools are served over MCP
                try:
                    closer()
                except Exception:
                    log.warning("a tools backend did not close cleanly", exc_info=True)
        self.dependencies.agent.tracer.close()
        close_checkpointer(self.checkpointer)
        tools.store.close()


def build_default_resources(settings: Settings | None = None) -> Resources:
    """Production wiring: the durable store and checkpointer, the EHR over MCP, policy search if built."""
    settings = settings or get_settings()
    try:  # imported here so that a missing index or retrieval dependency only switches policy search off
        from ..rag.retriever import open_policy_search

        retriever = open_policy_search(settings)
        policy_index = retriever if retriever.count() else None
    except Exception:
        log.warning("policy search is off (no usable index): policy questions go to staff", exc_info=True)
        policy_index = None
    tools: Any = CareFlowTools(store=RuntimeStore(), policy_index=policy_index)
    if settings.careflow_api_use_mcp:
        try:
            from ..mcp import McpRoutedTools

            tools = McpRoutedTools(tools)
        except Exception:  # the same tools then run in-process
            log.warning("the EHR MCP server did not start: its tools run in-process", exc_info=True)
    classifier = build_classifier(settings)
    warm = getattr(classifier, "warm", None)
    if warm:  # load the local model now: a first message that waited for it would time out and be refused
        try:
            warm()
        except Exception:  # /health reports it; every message is then refused by the gate (fail closed)
            log.error("the local clinical classifier did not load: the input gate will refuse every message",
                      exc_info=True)
    return Resources(TriageDependencies.build_default(tools, settings), open_session_checkpointer(), classifier, settings, owned=True)


def _describe_classifier(classifier) -> str:
    """none | hosted | local:ready | local:not loaded, for /health."""
    if classifier is None:
        return "none"
    if not hasattr(classifier, "warm"):
        return "hosted"
    return "local:ready" if getattr(classifier, "_weights", None) is not None else "local:not loaded"


def _describe_backends(tools) -> dict:
    """For /health: is the EHR served over MCP (start-up falls back to in-process tools if the server will
    not start), and the state of each circuit breaker."""
    client = getattr(getattr(tools, "inner", tools), "client", None)
    thread = getattr(client, "_thread", None)
    mcp_state = "off" if client is None else ("running" if thread is None or thread.is_alive() else "stopped")
    # str(): the response holds the plain string of a BreakerState
    return {"mcp": mcp_state,
            "breakers": {name: str(breaker.state)
                         for name, breaker in (getattr(tools, "breakers", None) or {}).items()}}


def _build_triage_payload(result) -> dict:
    """A TriageResult as the response body: every field, with the audit trail reduced to its step names."""
    data = asdict(result)
    data.pop("audit")
    return {**data, "steps": result.steps}


def create_app(resources: Resources | None = None) -> FastAPI:
    """The app. `resources` are the caller's own (tests) and are left open at shutdown; without them the
    production ones are opened at start-up and closed at shutdown."""
    # The route handlers carry comments, not docstrings: FastAPI would publish a docstring in /openapi.json.

    @asynccontextmanager
    async def manage_resources(app: FastAPI):
        app.state.resources = resources or build_default_resources()
        try:
            yield
        finally:
            if app.state.resources.owned:
                app.state.resources.close()

    app = FastAPI(title="CareFlow", version="1.0", lifespan=manage_resources)

    def get_resources(request: Request) -> Resources:
        return request.app.state.resources

    def check_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        """The API-key check for /v1 routes; no key configured means no check. Compared in constant time."""
        settings = get_resources(request).settings or get_settings()
        if settings.careflow_api_key and not secrets.compare_digest(
                (x_api_key or "").encode(), settings.careflow_api_key.get_secret_value().encode()):
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_request(request: Request, validation_error: RequestValidationError) -> JSONResponse:
        # Which field was wrong and why, but not the rejected value: FastAPI's default body returns the
        # input, which for an over-long or mis-shaped request is the patient's whole message.
        problems = [{key: error[key] for key in ("type", "loc", "msg") if key in error}
                    for error in validation_error.errors()]
        return JSONResponse(status_code=422, content={"detail": problems})

    @app.get("/health")
    def health(request: Request) -> dict:
        held_resources = get_resources(request)
        try:  # one read from the store and one from the checkpointer: 503 if either cannot be reached
            # The patients are cached at start-up, so the query below is the store check.
            patients = len(held_resources.dependencies.tools.patients)
            pending_reviews = len(held_resources.dependencies.tools.store.list_pending_reviews())
            held_resources.checkpointer.get_tuple(build_thread_config("health"))
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"not ready: {type(error).__name__}") from error
        return {"status": "ok", "patients": patients, "pending_reviews": pending_reviews,
                "policy_search": ToolName.SEARCH_POLICIES in held_resources.dependencies.tools.list_available_tools(),
                "tracing": held_resources.dependencies.agent.tracer.enabled,
                "guard_classifier": _describe_classifier(held_resources.classifier),
                **_describe_backends(held_resources.dependencies.tools)}

    @app.post("/v1/triage", dependencies=[Depends(check_api_key)])
    def triage(body: TriageRequest, request: Request) -> dict:
        held_resources = get_resources(request)
        _reject_chat_thread(body.thread_id)
        record = {"channel": body.channel,
                  "received_at": body.received_at or held_resources.dependencies.tools.clock.now().isoformat(),
                  "raw_text": body.raw_text, "patient_ref": body.patient_ref}
        try:
            result = start_triage(record, held_resources.dependencies, held_resources.checkpointer, thread_id=body.thread_id)
        except AlreadyTriagedError as error:
            raise HTTPException(status_code=409, detail=f"thread {error} already holds a case") from error
        return _build_triage_payload(result)

    @app.get("/v1/reviews", dependencies=[Depends(check_api_key)])
    def reviews(request: Request) -> dict:
        pending = get_resources(request).dependencies.tools.store.list_pending_reviews()
        return {"count": len(pending), "reviews": pending}

    @app.get("/v1/reviews/{thread_id}", dependencies=[Depends(check_api_key)])
    def review(thread_id: str, request: Request) -> dict:
        held_resources = get_resources(request)
        _reject_chat_thread(thread_id)
        payload = get_pending_review(thread_id, held_resources.dependencies, held_resources.checkpointer)
        if payload is None:
            raise HTTPException(status_code=404, detail="no case is waiting for review on this thread")
        return payload

    @app.post("/v1/reviews/{thread_id}/decision", dependencies=[Depends(check_api_key)])
    def decide(thread_id: str, body: ReviewDecision, request: Request) -> dict:
        held_resources = get_resources(request)
        _reject_chat_thread(thread_id)
        try:
            result = resume_triage(thread_id, body, held_resources.dependencies, held_resources.checkpointer)
        except NotAwaitingReviewError as error:
            raise HTTPException(status_code=409, detail="this thread is not waiting for a decision") from error
        except ValueError as error:  # a re-band the policy does not allow
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _build_triage_payload(result)

    @app.post("/v1/chat", dependencies=[Depends(check_api_key)])
    def chat(body: ChatRequest, request: Request) -> dict:
        held_resources = get_resources(request)
        settings = held_resources.settings or get_settings()
        thread_id = body.thread_id or uuid.uuid4().hex
        try:
            team = run_team(
                body.message, held_resources.dependencies.agent, held_resources.reviewer, patient_ref=body.patient_ref,
                channel=body.channel,
                # Always stored, also when the caller sent no thread_id: the id returned below must lead
                # back to this turn, or a caller identified here would have to identify again.
                thread_id=CHAT_PREFIX + thread_id, checkpointer=held_resources.checkpointer, guard=True,
                classifier=held_resources.classifier, guard_time_limit_s=settings.careflow_guard_timeout_s,
            )
        except PatientMismatchError as error:
            raise HTTPException(status_code=409,
                                detail="this conversation belongs to another patient; start a new one") from error
        except Exception as error:  # the model is down: say so, never guess
            # Logged with the thread and the error's type only. The error's own text and traceback are left
            # out: a provider error can quote the request, which holds the message and the patient (data handling).
            log.error("chat failed on thread %s: %s", thread_id, type(error).__name__)
            _flag_unanswered_chat(held_resources.dependencies.agent.tools, thread_id, body.patient_ref, error)
            raise HTTPException(status_code=503, detail=UNAVAILABLE) from error
        # The caller's own thread_id is returned, without the internal chat prefix.
        return {"thread_id": thread_id, "answer": team.answer, "route": team.route, "agents": team.agents,
                "tools_used": team.tools_used, "vetoed": team.vetoed, "guard": team.guard, "verdict": team.verdict,
                "patient_ref": team.patient_ref}

    return app


app = create_app()
