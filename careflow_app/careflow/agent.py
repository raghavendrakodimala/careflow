"""M2: a single scheduling / insurance-lookup agent. A thin, stable wrapper over the graph in graph.py.

run_agent() keeps one simple signature for callers (the demo script, tests, M8's API). The graph is
where the structure lives; see graph.py for how later milestones extend it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .graph import MAX_STEPS, Deps, build_agent_graph
from .llm_client import LLMClientError
from .prompts import SYSTEM_PROMPT
from .tools import CareFlowTools


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # {name, arguments, result}
    steps: int = 0
    stopped_early: bool = False  # hit the step cap without a final answer

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


def _trace(messages: list[dict], start: int) -> list[dict]:
    """Rebuild {name, arguments, result} for every tool call made this turn."""
    results = {m["tool_call_id"]: json.loads(m["content"]) for m in messages[start:] if m["role"] == "tool"}
    return [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"], "result": results[c["id"]]}
        for m in messages[start:]
        for c in m.get("tool_calls", [])
        if c["id"] in results
    ]


def run_agent(
    user_message: str,
    tools: CareFlowTools | None = None,
    settings: Settings | None = None,
    max_steps: int = MAX_STEPS,
    patient_ref: str | None = None,
    model: Any = None,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> AgentResult:
    """Answer one request. `patient_ref`, if the caller already knows it, is given as context.

    `model` replaces the LLM call (tests). With `checkpointer` and `thread_id`, turns on the same
    thread share history (session memory, M3).
    """
    deps = Deps.default(tools, settings)
    if model is not None:
        deps.model = model
    graph = build_agent_graph(deps, checkpointer=checkpointer, max_steps=max_steps)

    context = f"\n\n(Caller context: patient_ref={patient_ref})" if patient_ref else ""
    config: dict = {"recursion_limit": max_steps * 2 + 6}
    before: list[dict] = []
    if checkpointer is not None:
        config["configurable"] = {"thread_id": thread_id or "default"}
        before = graph.get_state(config).values.get("messages", [])

    state = graph.invoke(
        {
            "messages": [{"role": "user", "content": user_message + context}],
            "patient_ref": patient_ref,
            "steps": 0,
            "escalated": False,
        },
        config,
    )
    messages = state["messages"]
    return AgentResult(
        answer=(messages[-1].get("content") or "").strip(),
        tool_calls=_trace(messages, len(before)),
        steps=state.get("steps", 0),
        stopped_early=bool(state.get("escalated")),
    )


__all__ = ["AgentResult", "run_agent", "LLMClientError", "SYSTEM_PROMPT"]
