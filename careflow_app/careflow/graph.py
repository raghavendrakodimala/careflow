"""M2 agent as a LangGraph graph: model -> tools -> model ... -> END, with a step-cap escalation.

Built to be extended, not edited:
  M3  add a node before `model` that loads patient memory; pass a checkpointer for session memory
  M4  add a tool to tools.py (no graph change)
  M5  call this graph (or its pieces) as a step of the intake -> eligibility -> routing workflow
  M6  call build_agent_graph() once per agent with its own system_prompt and tool_names
  M7  wrap Deps.model and Deps.tools.call with retries / a circuit breaker
  M8  insert a guard node between `model` and END
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm_client import complete_with_tools
from .prompts import SYSTEM_PROMPT
from .state import CareFlowState
from .tools import TOOL_SPECS, CareFlowTools

MAX_STEPS = 6
ESCALATION_REASON = "Assistant could not finish the request within its step limit"
ESCALATION_ANSWER = "I could not complete this request, so I have passed it to clinic staff who will follow up."


@dataclass
class Deps:
    """Everything the graph needs from outside. The single seam for tests and for M7 hardening."""

    tools: CareFlowTools
    model: Callable[[list[dict], list[dict]], Any]  # (messages, tool_specs) -> assistant message

    @classmethod
    def default(cls, tools: CareFlowTools | None = None, settings: Settings | None = None) -> Deps:
        return cls(
            tools=tools or CareFlowTools(),
            model=lambda messages, specs: complete_with_tools(messages, specs, settings),
        )


def _call_as_dict(call: Any) -> dict:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
    }


def build_agent_graph(
    deps: Deps,
    system_prompt: str = SYSTEM_PROMPT,
    tool_names: set[str] | None = None,
    checkpointer: Any = None,
    max_steps: int = MAX_STEPS,
):
    specs = [s for s in TOOL_SPECS if tool_names is None or s["function"]["name"] in tool_names]
    allowed = {s["function"]["name"] for s in specs}

    def model_node(state: CareFlowState) -> dict:
        msg = deps.model([{"role": "system", "content": system_prompt}, *state["messages"]], specs)
        calls = getattr(msg, "tool_calls", None) or []
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if calls:
            entry["tool_calls"] = [_call_as_dict(c) for c in calls]
        return {"messages": [entry], "steps": state.get("steps", 0) + 1}

    def tools_node(state: CareFlowState) -> dict:
        out = []
        for call in state["messages"][-1]["tool_calls"]:
            name = call["function"]["name"]
            if name in allowed:
                result = deps.tools.call(name, call["function"]["arguments"])
            else:  # the model asked for a tool this agent does not have
                result = {"ok": False, "error": f"tool {name!r} is not available to this agent", "tools": sorted(allowed)}
            out.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
        return {"messages": out}

    def escalate_node(state: CareFlowState) -> dict:
        """Step cap reached: never leave the patient with silence. Hand to staff and say so."""
        # Goes through deps.tools.call, the same seam as model-requested tools, so M7 tracing and
        # resilience wrappers see this handoff too.
        ref = state.get("patient_ref")
        args = {"reason": ESCALATION_REASON, "urgency": "Routine", "patient_ref": ref}
        result = deps.tools.call("flag_for_human", args)
        if not result.get("ok"):
            # e.g. the caller's patient_ref is unknown. The handoff must still reach staff, so flag it
            # without the reference and keep the unverified value in the reason.
            args = {"reason": f"{ESCALATION_REASON} (unverified patient reference: {ref})", "urgency": "Routine"}
            result = deps.tools.call("flag_for_human", args)
        call = {
            "id": "escalate-1",
            "type": "function",
            "function": {"name": "flag_for_human", "arguments": json.dumps(args)},
        }
        return {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "escalate-1", "content": json.dumps(result)},
                {"role": "assistant", "content": ESCALATION_ANSWER},
            ],
            "escalated": True,
        }

    def after_model(state: CareFlowState) -> str:
        return "tools" if state["messages"][-1].get("tool_calls") else END

    def after_tools(state: CareFlowState) -> str:
        return "escalate" if state.get("steps", 0) >= max_steps else "model"

    g = StateGraph(CareFlowState)
    g.add_node("model", model_node)
    g.add_node("tools", tools_node)
    g.add_node("escalate", escalate_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
    g.add_conditional_edges("tools", after_tools, {"model": "model", "escalate": "escalate"})
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer)
