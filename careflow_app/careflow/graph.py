"""The CareFlow agent as a LangGraph graph.

    START -> memory -> model -> tools -> memory -> model ... -> record -> END
                                  tools -> escalate (step cap) -> record -> END

  memory    M3: rebuilds the verified patient's long-term memory note from the runtime store
  model     calls the LLM with the system prompt, the memory note and a rolling window of recent turns
  tools     runs the requested tools; a successful find_patient sets the thread's verified patient_ref
  escalate  the step cap was reached: flag staff and say so, never leave silence
  record    M3: appends one interaction for this turn to the patient's history

Built to be extended, not edited:
  M4  improve search_policies (hybrid + rerank) in rag/ (no graph change)
  M5  call this graph (or its pieces) as a step of the intake -> eligibility -> routing workflow
  M6  call build_agent_graph() once per agent with its own system_prompt and tool_names
  M7  wrap Deps.model and Deps.tools.call with retries / a circuit breaker
  M8  insert a guard node between `model` and `record`
Session memory is the checkpointer (one thread_id per conversation); see agent.run_agent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm_client import complete_with_tools
from .memory import PRIORITY_MARK, PatientMemory, summarise_turn
from .prompts import SYSTEM_PROMPT
from .state import CareFlowState
from .tools import TOOL_SPECS, CareFlowTools

MAX_STEPS = 6
HISTORY_TURNS = 8  # user turns the model sees; older turns stay in the checkpoint but are not sent
ESCALATION_REASON = "Assistant could not finish the request within its step limit"
ESCALATION_ANSWER = "I could not complete this request, so I have passed it to clinic staff who will follow up."
OTHER_PATIENT_ERROR = (
    "This conversation is verified for one patient, so only that patient's records can be used. Do not share "
    "anything about anyone else; another person must contact the clinic themselves."
)


@dataclass
class Deps:
    """Everything the graph needs from outside. The single seam for tests and for M7 hardening."""

    tools: CareFlowTools
    model: Callable[[list[dict], list[dict]], Any]  # (messages, tool_specs) -> assistant message
    memory: PatientMemory | None = None  # M3 long-term patient memory; None turns it off

    @classmethod
    def default(cls, tools: CareFlowTools | None = None, settings: Settings | None = None) -> Deps:
        tools = tools or CareFlowTools()
        return cls(
            tools=tools,
            model=lambda messages, specs: complete_with_tools(messages, specs, settings),
            memory=PatientMemory(tools.store, tools.clock),
        )


def _call_as_dict(call: Any) -> dict:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
    }


def _thread_id(config: RunnableConfig | None) -> str | None:
    """The conversation's thread_id (None when run without a checkpointer: each call is its own contact)."""
    return ((config or {}).get("configurable") or {}).get("thread_id")


def _contact_id(config: RunnableConfig | None) -> str | None:
    """The contact a call belongs to: the conversation's thread_id, else the per-call contact_id run_agent sets."""
    conf = (config or {}).get("configurable") or {}
    return conf.get("thread_id") or conf.get("contact_id")


def _other_patients(tools: CareFlowTools, raw_arguments: Any, bound: str) -> set[str]:
    """Patients other than `bound` that a tool call's arguments point at."""
    try:
        args = json.loads(raw_arguments or "{}") if isinstance(raw_arguments, str) else dict(raw_arguments or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()  # malformed arguments: the tool's own validation reports them
    return tools.patients_referenced(args) - {bound} if isinstance(args, dict) else set()


_MARKED = re.compile(r"(?i)^\s*(repeat\s*[-–]\s*priority|repeat)\b")


def _prepare_flag(
    memory: PatientMemory | None, raw: Any, patient_ref: str, done: list[dict], thread_id: str | None
) -> Any:
    """Complete a staff flag on a verified conversation from the records, not from the model's judgement.

    - patient_ref: the verified patient, if the model left it out (a ticket staff cannot attribute is useless).
    - Repeat rules: "Repeat - Priority" (urgency at least Priority) for the third same-issue contact within
      72 hours (OP-CS-014 8.2), "REPEAT" for a same-issue contact within 24 hours (triage SOP 7.3). The
      issue is clinical for a clinical flag, otherwise what this turn's tools worked on; with no known
      issue the reason is left as the model wrote it (the prompt rule still applies)."""
    try:
        args = json.loads(raw or "{}") if isinstance(raw, str) else dict(raw or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw
    if not isinstance(args, dict) or not isinstance(args.get("reason"), str):
        return raw  # malformed: the tool's validation reports it
    if not args.get("patient_ref"):
        args["patient_ref"] = patient_ref
    issue = "clinical_question" if args.get("clinical") else summarise_turn(done)["topic"]
    marker = memory.repeat_marker(patient_ref, issue, exclude_thread=thread_id) if memory else None
    if marker:
        reason = _MARKED.sub("", args["reason"]).lstrip(" :-–")
        args["reason"] = f"{marker}: {reason}"
        if marker == PRIORITY_MARK and args.get("urgency", "Routine") == "Routine":
            args["urgency"] = "Priority"
    return json.dumps(args)


def _turn_start(messages: list[dict]) -> int:
    """Index of the latest user message: where the current turn begins."""
    return max((i for i, m in enumerate(messages) if m["role"] == "user"), default=0)


def recent_turns(messages: list[dict], turns: int) -> list[dict]:
    """The last `turns` user turns. Cuts only at a user message, so a tool call is never separated
    from its result."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if turns and len(starts) > turns:
        return messages[starts[-turns]:]
    return messages


def turn_tool_calls(messages: list[dict], start: int | None = None) -> list[dict]:
    """{name, arguments, result} for every tool call from `start` (default: the current turn)."""
    start = _turn_start(messages) if start is None else start
    results = {m["tool_call_id"]: json.loads(m["content"]) for m in messages[start:] if m["role"] == "tool"}
    return [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"], "result": results[c["id"]]}
        for m in messages[start:]
        for c in m.get("tool_calls", [])
        if c["id"] in results
    ]


def build_agent_graph(
    deps: Deps,
    system_prompt: str = SYSTEM_PROMPT,
    tool_names: set[str] | None = None,
    checkpointer: Any = None,
    max_steps: int = MAX_STEPS,
    history_turns: int = HISTORY_TURNS,
):
    if tool_names is None:  # everything this configuration can run (no policy index -> no policy search)
        tool_names = deps.tools.available_tools()
    specs = [s for s in TOOL_SPECS if s["function"]["name"] in tool_names]
    allowed = {s["function"]["name"] for s in specs}

    def memory_node(state: CareFlowState, config: RunnableConfig) -> dict:
        if deps.memory is None:
            return {}
        # This conversation's own contact row is not an "earlier contact" for the repeat rules.
        return {"patient_memory": deps.memory.context(state.get("patient_ref"), exclude_thread=_thread_id(config))}

    def model_node(state: CareFlowState) -> dict:
        prefix = [{"role": "system", "content": system_prompt}]
        if state.get("patient_memory"):
            prefix.append({"role": "system", "content": state["patient_memory"]})
        msg = deps.model([*prefix, *recent_turns(state["messages"], history_turns)], specs)
        calls = getattr(msg, "tool_calls", None) or []
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        if calls:
            entry["tool_calls"] = [_call_as_dict(c) for c in calls]
        return {"messages": [entry], "steps": state.get("steps", 0) + 1}

    def tools_node(state: CareFlowState, config: RunnableConfig) -> dict:
        out, update = [], {}
        bound = state.get("patient_ref")  # the verified patient, if any
        done = turn_tool_calls(state["messages"])  # this turn's earlier tool calls, for the repeat marker
        for call in state["messages"][-1]["tool_calls"]:
            name, raw = call["function"]["name"], call["function"]["arguments"]
            if name not in allowed:  # the model asked for a tool this agent does not have
                result = {"ok": False, "error": f"tool {name!r} is not available to this agent", "tools": sorted(allowed)}
            elif bound and (others := _other_patients(deps.tools, raw, bound)):
                # Deterministic, not left to the prompt: a verified conversation only touches its own
                # patient's records (data-handling standard, minimum necessary).
                result = {"ok": False, "error": OTHER_PATIENT_ERROR}
            else:
                if name == "flag_for_human" and bound:
                    raw = _prepare_flag(deps.memory, raw, bound, done, _thread_id(config))
                result = deps.tools.call(name, raw, context={"contact_id": _contact_id(config)})
                done.append({"name": name, "arguments": raw, "result": result})
                if name == "find_patient" and result.get("found"):
                    if bound and result["patient_ref"] != bound:
                        result = {"ok": False, "error": OTHER_PATIENT_ERROR}
                    else:
                        update["patient_ref"] = bound = result["patient_ref"]  # verified: memory loads next
            out.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
        return {"messages": out, **update}

    def escalate_node(state: CareFlowState, config: RunnableConfig) -> dict:
        """Step cap reached: never leave the patient with silence. Hand to staff and say so."""
        # Goes through deps.tools.call, the same seam as model-requested tools, so M7 tracing and
        # resilience wrappers see this handoff too.
        ref = state.get("patient_ref")
        args = {"reason": ESCALATION_REASON, "urgency": "Routine", "patient_ref": ref}
        context = {"contact_id": _contact_id(config)}
        result = deps.tools.call("flag_for_human", args, context=context)
        if not result.get("ok"):
            # e.g. the caller's patient_ref is unknown. The handoff must still reach staff, so flag it
            # without the reference and keep the unverified value in the reason.
            args = {"reason": f"{ESCALATION_REASON} (unverified patient reference: {ref})", "urgency": "Routine"}
            result = deps.tools.call("flag_for_human", args, context=context)
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

    def record_node(state: CareFlowState, config: RunnableConfig) -> dict:
        if deps.memory is not None:
            messages = state["messages"]
            deps.memory.record_turn(
                state.get("patient_ref"),
                turn_tool_calls(messages),
                channel=state.get("channel") or "web_form",
                thread_id=_thread_id(config),
                answer=messages[-1].get("content") or "",
            )
        return {}

    def after_model(state: CareFlowState) -> str:
        return "tools" if state["messages"][-1].get("tool_calls") else "record"

    def after_tools(state: CareFlowState) -> str:
        return "escalate" if state.get("steps", 0) >= max_steps else "memory"

    g = StateGraph(CareFlowState)
    g.add_node("memory", memory_node)
    g.add_node("model", model_node)
    g.add_node("tools", tools_node)
    g.add_node("escalate", escalate_node)
    g.add_node("record", record_node)
    g.add_edge(START, "memory")
    g.add_edge("memory", "model")
    g.add_conditional_edges("model", after_model, {"tools": "tools", "record": "record"})
    g.add_conditional_edges("tools", after_tools, {"memory": "memory", "escalate": "escalate"})
    g.add_edge("escalate", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=checkpointer)
