"""The CareFlow agent: a thin, stable wrapper over the graph in graph.py.

run_agent() keeps one simple signature for callers (the demo scripts, tests, the team in agents.py). The
graph is where the structure lives; see graph.py.

Memory:
- session memory: pass a checkpointer and a thread_id; turns on the same thread share history.
  open_session_checkpointer() gives the persistent one (out/sessions.db), which survives a restart.
- long-term patient memory: on by default (AgentDependencies.build_default); read from and written to the runtime store
  for the verified patient only.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from .config import SESSIONS_DB, Settings
from .graph import MAX_STEPS, AgentDependencies, AgentNode, build_agent_graph, list_turn_tool_calls
from .llm_client import LLMClientError
from .prompts import SYSTEM_PROMPT
from .schemas import DEFAULT_CHANNEL
from .state import CONFIGURABLE, THREAD_ID, Role, build_thread_config
from .store import BUSY_TIMEOUT_S
from .tool_contracts import CONTACT_ID
from .tools import CareFlowTools

DEFAULT_THREAD = "default"  # the conversation used when a checkpointer is given without a thread_id


class PatientMismatchError(ValueError):
    """A turn named a different patient from the one this conversation is already verified for."""


@dataclass
class AgentResult:
    """One turn of one agent: what the patient is told and what the graph did to get there."""

    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # {name, arguments, result}
    steps: int = 0
    stopped_early: bool = False  # hit the step cap without a final answer
    patient_ref: str | None = None  # verified patient for the thread after this turn
    grounding: dict = field(default_factory=dict)  # groundedness check of this answer (graph `ground` node)

    @property
    def tools_used(self) -> list[str]:
        return [call["name"] for call in self.tool_calls]


def open_session_checkpointer(path: Path | str | None = None) -> SqliteSaver:
    """Persistent session memory: LangGraph checkpoints in SQLite (default out/sessions.db)."""
    path = Path(path or SESSIONS_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False, timeout=BUSY_TIMEOUT_S)
    # Same settings as the runtime store: the human-review step and the API workers share this file.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_S * 1000}")
    return SqliteSaver(connection)


def close_checkpointer(checkpointer: Any) -> None:
    """Close an open_session_checkpointer() (its SQLite connection). Safe on None and on in-memory savers."""
    connection = getattr(checkpointer, "conn", None)
    if connection is not None:
        connection.close()


def run_agent(
    user_message: str,
    tools: CareFlowTools | None = None,
    settings: Settings | None = None,
    max_steps: int = MAX_STEPS,
    patient_ref: str | None = None,
    model: Any = None,
    checkpointer: Any = None,
    thread_id: str | None = None,
    channel: str = DEFAULT_CHANNEL,
    dependencies: AgentDependencies | None = None,
    tool_names: set[str] | None = None,
    system_prompt: str | None = None,
) -> AgentResult:
    """Answer one request.

    `patient_ref`, if the caller has already verified it (an authenticated channel), is the thread's
    patient; otherwise the agent identifies the caller with find_patient. `model` replaces the LLM call
    (tests). `dependencies` replaces all dependencies (e.g. memory=None to switch long-term memory off). `tool_names`
    limits the tools offered (the triage workflow withholds booking while a case waits for staff).
    `system_prompt` replaces the default prompt (one prompt per specialist). With a `checkpointer` the turn
    is added to the conversation stored under `thread_id`; PatientMismatchError is raised if that conversation is
    already verified for a different patient.
    """
    # A falsy or blank reference means "not verified", the same as None.
    patient_ref = str(patient_ref).strip().upper() or None if patient_ref else None
    dependencies = dependencies or AgentDependencies.build_default(tools, settings)
    if model is not None:
        dependencies.model = model
    graph = build_agent_graph(dependencies, system_prompt or SYSTEM_PROMPT, tool_names=tool_names,
                              checkpointer=checkpointer, max_steps=max_steps)

    context = f"\n\n(Caller context: patient_ref={patient_ref})" if patient_ref else ""
    # One contact = one conversation (thread). Without a checkpointer each call is its own contact, so it
    # gets its own contact_id: staff flags de-duplicate within a contact, never across contacts.
    # Recursion limit: a step is up to three nodes (memory, model, tools); the rest is headroom for the
    # closing nodes (ground or escalate, then record).
    config: dict = {"recursion_limit": max_steps * 3 + 8, CONFIGURABLE: {CONTACT_ID: uuid.uuid4().hex}}
    before: list[dict] = []
    if checkpointer is None and thread_id:
        # Run as a step of the triage workflow: the same contact as the caller's thread, stated
        # explicitly so flags de-duplicate and memory is recorded against that thread.
        config[CONFIGURABLE] = {CONTACT_ID: thread_id, THREAD_ID: thread_id}
    if checkpointer is not None:
        config.update(build_thread_config(thread_id or DEFAULT_THREAD))
        saved = graph.get_state(config).values
        before = saved.get("messages", [])
        # The thread holds one patient's conversation. A different patient_ref on it would put that
        # history in front of another patient (data-handling standard 4.1): refuse the turn.
        if patient_ref and saved.get("patient_ref") and saved["patient_ref"] != patient_ref:
            raise PatientMismatchError(thread_id or DEFAULT_THREAD)

    turn: dict = {
        "messages": [{"role": Role.USER.value, "content": user_message + context}],
        "steps": 0,
        "escalated": False,
        "channel": channel,
        "draft": None,
        "grounding": {},
    }
    if patient_ref:  # only set it when given, so a patient verified earlier on the thread is kept
        turn["patient_ref"] = patient_ref
    state = graph.invoke(turn, config)
    messages = state["messages"]
    return AgentResult(
        answer=(messages[-1].get("content") or "").strip(),
        tool_calls=list_turn_tool_calls(messages, len(before)),
        steps=state.get("steps", 0),
        stopped_early=bool(state.get("escalated")),
        patient_ref=state.get("patient_ref"),
        grounding=state.get("grounding") or {},
    )


def get_thread_patient(dependencies: AgentDependencies, checkpointer: Any, thread_id: str | None) -> str | None:
    """The patient a checkpointed conversation is verified for, or None."""
    if checkpointer is None:
        return None
    graph = build_agent_graph(dependencies, checkpointer=checkpointer)
    return graph.get_state(build_thread_config(thread_id or DEFAULT_THREAD)).values.get("patient_ref")


def redact_answers(dependencies: AgentDependencies, checkpointer: Any, thread_id: str | None, text: str, count: int = 1) -> None:
    """Replace the last `count` stored answers on a thread with `text` (a vetoed reply; see state.add_messages),
    so the next turn's model never reads the withheld wording."""
    if checkpointer is None:
        return
    graph = build_agent_graph(dependencies, checkpointer=checkpointer)
    graph.update_state(build_thread_config(thread_id or DEFAULT_THREAD),
                       {"messages": [{"role": Role.REDACT.value, "content": text, "count": count}]},
                       as_node=AgentNode.RECORD.value)


__all__ = [
    "SYSTEM_PROMPT",
    "AgentResult",
    "LLMClientError",
    "PatientMismatchError",
    "close_checkpointer",
    "get_thread_patient",
    "open_session_checkpointer",
    "redact_answers",
    "run_agent",
]
