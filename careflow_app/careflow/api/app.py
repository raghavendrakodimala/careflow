"""M8: the CareFlow HTTP service (FastAPI). Thin adapters over the workflow and the agent team.

    uvicorn careflow.api.app:app            (from careflow_app/)

    GET  /health                             store and checkpointer reachable
    POST /v1/triage                          run one incoming message through the triage workflow
    GET  /v1/reviews                         the staff queue: cases waiting for a decision
    GET  /v1/reviews/{thread_id}             what staff are asked to decide for one case
    POST /v1/reviews/{thread_id}/decision    record the decision and finish the case
    POST /v1/chat                            one conversational turn with the agent team

Resources are opened once at start-up (runtime store, session checkpointer, the EHR MCP server, the
policy index if it has been built, the tracer) and closed at shutdown. Tests pass their own through
create_app(), so no model, index or child process is needed.

Access: if CAREFLOW_API_KEY is set, every /v1 request must send it as X-API-Key. This service has no
user login. `patient_ref` in a request is trusted as already verified by the calling channel (a patient
portal session, a staff console); a caller without one is identified from full name plus member number.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ..agent import close_checkpointer, session_checkpointer
from ..agents import Reviewer, run_team
from ..config import Settings, get_settings
from ..guardrails import Classifier, default_classifier
from ..schemas import Channel
from ..store import RuntimeStore
from ..tools import CareFlowTools
from ..workflows import (AlreadyTriaged, NotAwaitingReview, ReviewDecision, TriageDeps, pending_review, resume_triage,
                         start_triage)

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
    return Resources(TriageDeps.default(tools, settings), session_checkpointer(), default_classifier(settings),
                     settings, owned=True)


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
        if settings.careflow_api_key and x_api_key != settings.careflow_api_key:
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
                "tracing": r.deps.agent.tracer.enabled}

    @app.post("/v1/triage", dependencies=[Depends(authorised)])
    def triage(body: TriageRequest, request: Request) -> dict:
        r = res(request)
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
        payload = pending_review(thread_id, r.deps, r.checkpointer)
        if payload is None:
            raise HTTPException(status_code=404, detail="no case is waiting for review on this thread")
        return payload

    @app.post("/v1/reviews/{thread_id}/decision", dependencies=[Depends(authorised)])
    def decide(thread_id: str, body: ReviewDecision, request: Request) -> dict:
        r = res(request)
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
                thread_id=thread_id, checkpointer=r.checkpointer if body.thread_id else None, guard=True,
                classifier=r.classifier, guard_time_limit_s=settings.careflow_guard_timeout_s,
            )
        except Exception as exc:  # the model is down: say so, never guess
            raise HTTPException(status_code=503, detail=UNAVAILABLE) from exc
        return {"thread_id": thread_id, "answer": team.answer, "route": team.route, "agents": team.agents,
                "tools_used": team.tools_used, "vetoed": team.vetoed, "guard": team.guard, "verdict": team.verdict,
                "patient_ref": team.patient_ref}

    return app


app = create_app()
