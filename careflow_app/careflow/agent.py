"""The CareFlow agent: a thin, stable wrapper over the graph in graph.py.

run_agent() keeps one simple signature for callers (the demo scripts, tests, M8's API). The graph is
where the structure lives; see graph.py for how later milestones extend it.

Memory (M3):
- session memory: pass a checkpointer and a thread_id; turns on the same thread share history.
  session_checkpointer() gives the persistent one (out/sessions.db), which survives a restart.
- long-term patient memory: on by default (Deps.default); read from and written to the runtime store
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
from .graph import MAX_STEPS, Deps, build_agent_graph, turn_tool_calls
from .llm_client import LLMClientError
from .prompts import SYSTEM_PROMPT
from .state import REDACT
from .tools import CareFlowTools


class PatientMismatch(ValueError):
    """A turn named a different patient from the one this conversation is already verified for."""


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # {name, arguments, result}
    steps: int = 0
    stopped_early: bool = False  # hit the step cap without a final answer
    patient_ref: str | None = None  # verified patient for the thread after this turn
    grounding: dict = field(default_factory=dict)  # M4: groundedness check of this answer (graph `ground` node)

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


def session_checkpointer(path: Path | str | None = None) -> SqliteSaver:
    """Persistent session memory: LangGraph checkpoints in SQLite (default out/sessions.db)."""
    path = Path(path or SESSIONS_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
    # Same settings as the runtime store: M5's human-review step and M8's API workers share this file.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return SqliteSaver(conn)


def close_checkpointer(checkpointer: Any) -> None:
    """Close a session_checkpointer() (its SQLite connection). Safe on None and on in-memory savers."""
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        conn.close()


def run_agent(
    user_message: str,
    tools: CareFlowTools | None = None,
    settings: Settings | None = None,
    max_steps: int = MAX_STEPS,
    patient_ref: str | None = None,
    model: Any = None,
    checkpointer: Any = None,
    thread_id: str | None = None,
    channel: str = "web_form",
    deps: Deps | None = None,
    tool_names: set[str] | None = None,
    system_prompt: str | None = None,
) -> AgentResult:
    """Answer one request.

    `patient_ref`, if the caller has already verified it (an authenticated channel), is the thread's
    patient; otherwise the agent identifies the caller with find_patient. `model` replaces the LLM call
    (tests). `deps` replaces all dependencies (e.g. memory=None to switch long-term memory off). `tool_names`
    limits the tools offered (M5 withholds booking while a case waits for staff). `system_prompt`
    replaces the default prompt (M6: one prompt per specialist).
    """
    patient_ref = str(patient_ref).strip().upper() or None if patient_ref else None
    deps = deps or Deps.default(tools, settings)
    if model is not None:
        deps.model = model
    graph = build_agent_graph(deps, system_prompt or SYSTEM_PROMPT, tool_names=tool_names,
                              checkpointer=checkpointer, max_steps=max_steps)

    context = f"\n\n(Caller context: patient_ref={patient_ref})" if patient_ref else ""
    # One contact = one conversation (thread). Without a checkpointer each call is its own contact, so it
    # gets its own contact_id: staff flags de-duplicate within a contact, never across contacts.
    config: dict = {"recursion_limit": max_steps * 3 + 8, "configurable": {"contact_id": uuid.uuid4().hex}}
    before: list[dict] = []
    if checkpointer is None and thread_id:
        # Run as a step of a larger workflow (M5): the same contact as the caller's thread, stated
        # explicitly so flags de-duplicate and memory is recorded against that thread.
        config["configurable"] = {"contact_id": thread_id, "thread_id": thread_id}
    if checkpointer is not None:
        config["configurable"] = {"thread_id": thread_id or "default"}
        saved = graph.get_state(config).values
        before = saved.get("messages", [])
        # The thread holds one patient's conversation. A different patient_ref on it would put that
        # history in front of another patient (data-handling standard 4.1): refuse the turn.
        if patient_ref and saved.get("patient_ref") and saved["patient_ref"] != patient_ref:
            raise PatientMismatch(thread_id or "default")

    turn: dict = {
        "messages": [{"role": "user", "content": user_message + context}],
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
        tool_calls=turn_tool_calls(messages, len(before)),
        steps=state.get("steps", 0),
        stopped_early=bool(state.get("escalated")),
        patient_ref=state.get("patient_ref"),
        grounding=state.get("grounding") or {},
    )


def thread_patient(deps: Deps, checkpointer: Any, thread_id: str | None) -> str | None:
    """The patient a checkpointed conversation is verified for, or None."""
    if checkpointer is None:
        return None
    graph = build_agent_graph(deps, checkpointer=checkpointer)
    return graph.get_state({"configurable": {"thread_id": thread_id or "default"}}).values.get("patient_ref")


def redact_answers(deps: Deps, checkpointer: Any, thread_id: str | None, text: str, count: int = 1) -> None:
    """Replace the last `count` stored answers on a thread with `text` (a vetoed reply; see state.add_messages)."""
    if checkpointer is None:
        return
    graph = build_agent_graph(deps, checkpointer=checkpointer)
    graph.update_state({"configurable": {"thread_id": thread_id or "default"}},
                       {"messages": [{"role": REDACT, "content": text, "count": count}]}, as_node="record")


__all__ = ["AgentResult", "PatientMismatch", "run_agent", "session_checkpointer", "close_checkpointer", "thread_patient",
           "redact_answers", "LLMClientError", "SYSTEM_PROMPT"]
