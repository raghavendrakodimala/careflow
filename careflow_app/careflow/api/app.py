"""M8: the CareFlow HTTP service (FastAPI). Thin adapters over the workflow and the agent team.

    uvicorn careflow.api.app:app            (from careflow_app/)

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

import secrets
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ..agent import PatientMismatch, close_checkpointer, session_checkpointer
from ..agents import Reviewer, run_team
from ..config import Settings, get_settings
from ..guard_model import build_classifier
from ..guardrails import Classifier
from ..schemas import Channel
from ..store import RuntimeStore
from ..tools import CareFlowTools
from ..workflows import (AlreadyTriaged, NotAwaitingReview, ReviewDecision, TriageDeps, pending_review, resume_triage,
                         start_triage)

# Chat sessions and triage cases share one checkpoint store with different state shapes. A chat thread
# is stored under this prefix and a triage or review request may not use it, so neither can read the other.
CHAT_PREFIX = "chat:"


def _not_a_chat_thread(thread_id: str | None) -> None:
    if thread_id and thread_id.lower().startswith(CHAT_PREFIX):
        raise HTTPException(status_code=422, detail=f"thread_id may not start with {CHAT_PREFIX!r}")


UNAVAILABLE = "The assistant cannot answer right now. Your message has not been lost; please try again shortly or call the clinic."


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TriageRequest(_Body):
    raw_text: str = Field(min_length=1, max_length=8000)
    channel: Channel = "web_form"
    patient_ref: str | None = Field(default=None, max_length=32)
    received_at: str | None = Field(default=None, max_length=40)
    thread_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class ChatRequest(_Body):
    message: str = Field(min_length=1, max_length=8000)
    channel: Channel = "web_form"
    patient_ref: str | None = Field(default=None, max_length=32)
    thread_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


@dataclass
class Resources:
    deps: TriageDeps
    checkpointer: Any
    classifier: Classifier | None = None
    settings: Settings | None = None
    owned: bool = False  # opened by the app itself, so closed by it too

    @property
    def reviewer(self) -> Reviewer | None:
        return self.deps.review

    def close(self) -> None:
        tools = self.deps.tools
        for closer in (getattr(tools, "close", None), getattr(getattr(tools, "inner", None), "close", None)):
            if closer:  # the MCP client, if the EHR tools are served over MCP
                try:
                    closer()
                except Exception:
                    pass
        self.deps.agent.tracer.close()
        close_checkpointer(self.checkpointer)
        tools.store.close()


def default_resources(settings: Settings | None = None) -> Resources:
    """Production wiring: the durable store and checkpointer, the EHR over MCP, policy search if built."""
    settings = settings or get_settings()
    policy_index = None
    try:
        from ..rag.retriever import open_policy_search

        retriever = open_policy_search(settings)
        policy_index = retriever if retriever.count() else None
    except Exception:
        policy_index = None  # no index yet: policy questions are handed to staff
    tools: Any = CareFlowTools(store=RuntimeStore(), policy_index=policy_index)
    if settings.careflow_api_use_mcp:
        try:
            from ..mcp import McpRoutedTools

            tools = McpRoutedTools(tools)
        except Exception:
            pass  # the server could not start: the same tools run in-process
    classifier = build_classifier(settings)
    warm = getattr(classifier, "warm", None)
    if warm:  # load the local model now: a first message that waited for it would time out and be refused
        try:
            warm()
        except Exception:
            pass  # /health reports it; every message is then refused by the gate (fail closed)
    return Resources(TriageDeps.default(tools, settings), session_checkpointer(), classifier, settings, owned=True)


def _classifier_state(classifier) -> str:
    """none | hosted | local:ready | local:not loaded, for /health."""
    if classifier is None:
        return "none"
    if not hasattr(classifier, "warm"):
        return "hosted"
    return "local:ready" if getattr(classifier, "_weights", None) is not None else "local:not loaded"


def _backends(tools) -> dict:
    """For /health: is the EHR served over MCP (start-up falls back to in-process tools if the server will
    not start), and the state of each circuit breaker."""
    client = getattr(getattr(tools, "inner", tools), "client", None)
    thread = getattr(client, "_thread", None)
    mcp = "off" if client is None else ("running" if thread is None or thread.is_alive() else "stopped")
    return {"mcp": mcp, "breakers": {name: b.state for name, b in (getattr(tools, "breakers", None) or {}).items()}}


def _triage_payload(result) -> dict:
    data = asdict(result)
    data.pop("audit")
    return {**data, "steps": result.steps}


def create_app(resources: Resources | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.resources = resources or default_resources()
        try:
            yield
        finally:
            if app.state.resources.owned:
                app.state.resources.close()

    app = FastAPI(title="CareFlow", version="1.0", lifespan=lifespan)

    def res(request: Request) -> Resources:
        return request.app.state.resources

    def authorised(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        settings = res(request).settings or get_settings()
        if settings.careflow_api_key and not secrets.compare_digest(
                (x_api_key or "").encode(), settings.careflow_api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")

    @app.get("/health")
    def health(request: Request) -> dict:
        r = res(request)
        try:
            patients = len(r.deps.tools.patients)
            r.checkpointer.get_tuple({"configurable": {"thread_id": "health"}})
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"not ready: {type(exc).__name__}") from exc
        return {"status": "ok", "patients": patients, "pending_reviews": len(r.deps.tools.store.pending_reviews()),
                "policy_search": "search_policies" in r.deps.tools.available_tools(),
                "tracing": r.deps.agent.tracer.enabled, "guard_classifier": _classifier_state(r.classifier),
                **_backends(r.deps.tools)}

    @app.post("/v1/triage", dependencies=[Depends(authorised)])
    def triage(body: TriageRequest, request: Request) -> dict:
        r = res(request)
        _not_a_chat_thread(body.thread_id)
        record = {"channel": body.channel, "received_at": body.received_at or r.deps.tools.clock.now().isoformat(),
                  "raw_text": body.raw_text, "patient_ref": body.patient_ref}
        try:
            result = start_triage(record, r.deps, r.checkpointer, thread_id=body.thread_id)
        except AlreadyTriaged as exc:
            raise HTTPException(status_code=409, detail=f"thread {exc} already holds a case") from exc
        return _triage_payload(result)

    @app.get("/v1/reviews", dependencies=[Depends(authorised)])
    def reviews(request: Request) -> dict:
        pending = res(request).deps.tools.store.pending_reviews()
        return {"count": len(pending), "reviews": pending}

    @app.get("/v1/reviews/{thread_id}", dependencies=[Depends(authorised)])
    def review(thread_id: str, request: Request) -> dict:
        r = res(request)
        _not_a_chat_thread(thread_id)
        payload = pending_review(thread_id, r.deps, r.checkpointer)
        if payload is None:
            raise HTTPException(status_code=404, detail="no case is waiting for review on this thread")
        return payload

    @app.post("/v1/reviews/{thread_id}/decision", dependencies=[Depends(authorised)])
    def decide(thread_id: str, body: ReviewDecision, request: Request) -> dict:
        r = res(request)
        _not_a_chat_thread(thread_id)
        try:
            result = resume_triage(thread_id, body, r.deps, r.checkpointer)
        except NotAwaitingReview as exc:
            raise HTTPException(status_code=409, detail="this thread is not waiting for a decision") from exc
        except ValueError as exc:  # a re-band the policy does not allow
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _triage_payload(result)

    @app.post("/v1/chat", dependencies=[Depends(authorised)])
    def chat(body: ChatRequest, request: Request) -> dict:
        r = res(request)
        settings = r.settings or get_settings()
        thread_id = body.thread_id or uuid.uuid4().hex
        try:
            team = run_team(
                body.message, r.deps.agent, r.reviewer, patient_ref=body.patient_ref, channel=body.channel,
                # Always stored, also when the caller sent no thread_id: the id returned below must lead
                # back to this turn, or a caller identified here would have to identify again.
                thread_id=CHAT_PREFIX + thread_id, checkpointer=r.checkpointer, guard=True,
                classifier=r.classifier, guard_time_limit_s=settings.careflow_guard_timeout_s,
            )
        except PatientMismatch as exc:
            raise HTTPException(status_code=409, detail="this conversation belongs to another patient; start a new one") from exc
        except Exception as exc:  # the model is down: say so, never guess
            raise HTTPException(status_code=503, detail=UNAVAILABLE) from exc
        return {"thread_id": thread_id, "answer": team.answer, "route": team.route, "agents": team.agents,
                "tools_used": team.tools_used, "vetoed": team.vetoed, "guard": team.guard, "verdict": team.verdict,
                "patient_ref": team.patient_ref}

    return app


app = create_app()
