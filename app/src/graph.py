"""The CareFlow agent as a LangGraph graph.

    START -> memory -> model -> tools -> memory -> model ... -> ground -> record -> END
                                  tools -> escalate (step cap) -> record -> END

  memory    rebuilds the verified patient's long-term memory note from the runtime store
  model     calls the LLM with the system prompt, the memory note and a rolling window of recent turns
  tools     runs the requested tools; a successful find_patient sets the thread's verified patient_ref
  ground    the final answer arrives as a draft; if the conversation window holds a policy search or
            the answer names a corpus document, every figure and citation in it must be supported
            (grounding.py). Supported: the draft is the answer.
            Not supported: it is withheld, staff are flagged and the patient is told so
  escalate  the step cap was reached: flag staff and say so, never leave silence
  record    appends one interaction for this turn to the patient's history

The graph is configured, not edited: build_agent_graph() is called once per agent with its own
system_prompt and tool_names (agents.py), and everything from outside comes through AgentDependencies, where the
retries, circuit breakers and tracing are wrapped around the model and the tools.
Session memory is the checkpointer (one thread_id per conversation); see agent.run_agent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .domain import policies
from .domain.policies import UrgencyBand
from .grounding import GroundingResult, check_answer, has_document_name, list_corpus_documents
from .llm_client import complete_with_tools
from .memory import PRIORITY_MARK, PatientMemory, summarise_turn
from .prompts import (
    DEFLECTION_MARK,  # noqa: F401 - re-exported: the evaluation script reads the marks from here
    DEFLECTION_SCRIPT,
    EMERGENCY_MARK,  # noqa: F401
    EMERGENCY_STATEMENT,
    SYSTEM_PROMPT,
    has_emergency_statement,
    is_deflection,
)
from .reliability import BACKENDS, MODEL_BREAKER, CircuitBreaker, ResilientTools, RetryPolicy, make_resilient_model
from .schemas import DEFAULT_CHANNEL
from .state import CONFIGURABLE, THREAD_ID, CareFlowState, Role
from .store import FlagCategory
from .telemetry import NULL_TRACER, Tracer, build_tracer
from .tool_contracts import CONTACT_ID, ToolName
from .tools import TOOL_SPECS, CareFlowTools

MAX_STEPS = 6  # model calls per turn; a turn still asking for tools after this many is handed to staff
HISTORY_TURNS = 8  # user turns the model sees; older turns stay in the checkpoint but are not sent
MAX_SEARCHES = 3  # policy searches per turn. Rewording a search that did not answer rarely helps.
# Fixed staff-flag reasons and patient replies for the handoffs the graph makes itself.
ESCALATION_REASON = "Assistant could not finish the request within its step limit"
ESCALATION_ANSWER = "I could not complete this request, so I have passed it to clinic staff who will follow up."
GROUNDING_REASON = "Policy answer withheld: not supported by the clinic's documents"
GROUNDING_ANSWER = (
    "I could not confirm that answer in the clinic's policy documents, so I have not given it. I have passed "
    "your question to clinic staff, who will follow up."
)
CLINICAL_TICKET_REASON = "Clinical question deflected; Clinical Routing Ticket created automatically (OP-CS-014 9.6)"
OTHER_PATIENT_ERROR = (
    "This conversation is verified for one patient, so only that patient's records can be used. Do not share "
    "anything about anyone else; another person must contact the clinic themselves."
)
# Tool results that tell the model to stop searching (returned in place of running the search again).
REPEATED_SEARCH_NOTE = ("You already ran this exact search in this turn and these are the same passages. Do not "
                        "search for it again: answer from the passages and tool results you have, use another "
                        "tool (check_preauth or calculate_copay give co-pays and costs), or say you do not have "
                        "the information and offer to pass the question to staff.")
SEARCH_LIMIT_NOTE = (f"You have already searched the policies {MAX_SEARCHES} times in this turn, so this search was "
                     "not run. Answer from the passages and tool results you have, use another tool "
                     "(check_preauth or calculate_copay give co-pays and costs), or say you do not have the "
                     "information and offer to pass the question to staff.")
# A repeat marker already at the start of a flag reason (written by the model), replaced by the one from the records.
_MARKED = re.compile(r"(?i)^\s*(repeat\s*[-–]\s*priority|repeat)\b")


class AgentNode(StrEnum):
    """The nodes of the agent graph (see the module docstring). LangGraph is given the plain names."""

    MEMORY = "memory"
    MODEL = "model"
    TOOLS = "tools"
    GROUND = "ground"
    ESCALATE = "escalate"
    RECORD = "record"


@dataclass
class AgentDependencies:
    """Everything the graph needs from outside. The single seam for tests and for the resilience wrappers."""

    tools: CareFlowTools
    model: Callable[[list[dict], list[dict]], Any]  # (messages, tool_specs) -> assistant message
    memory: PatientMemory | None = None  # long-term patient memory; None turns it off
    tracer: Tracer = NULL_TRACER  # spans for steps, agents, tools and model calls

    @classmethod
    def build_default(cls, tools: CareFlowTools | None = None, settings: Settings | None = None) -> AgentDependencies:
        """Production wiring: tools and model wrapped with retries, circuit breakers and spans."""
        tools = tools or CareFlowTools()
        active_settings = settings or get_settings()
        tracer = build_tracer(active_settings)
        if not isinstance(tools, ResilientTools):
            tools = ResilientTools(
                tools, retry=RetryPolicy(attempts=active_settings.careflow_tool_retries), tracer=tracer,
                breakers={backend: CircuitBreaker(backend, active_settings.careflow_breaker_failures,
                                                  active_settings.careflow_breaker_cooldown_s)
                          for backend in set(BACKENDS.values())},
            )
        model = make_resilient_model(
            lambda messages, specs: complete_with_tools(messages, specs, settings),
            CircuitBreaker(MODEL_BREAKER, active_settings.careflow_breaker_failures,
                           active_settings.careflow_breaker_cooldown_s),
            tracer, name=active_settings.careflow_model,
        )
        return cls(tools=tools, model=model, memory=PatientMemory(tools.store, tools.clock), tracer=tracer)


def _convert_call_to_dict(call: Any) -> dict:
    """A tool call from the model's reply (an SDK object) as the plain dict stored in the message history."""
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
    }


def _build_flag_messages(call_id: str, arguments: str, result: dict) -> list[dict]:
    """A flag_for_human call the graph made itself, as an assistant tool call plus its result, so the trace,
    memory and AgentResult.tool_calls see it like a call the model asked for."""
    call = {"id": call_id, "type": "function",
            "function": {"name": ToolName.FLAG_FOR_HUMAN.value, "arguments": arguments}}
    return [{"role": Role.ASSISTANT.value, "content": "", "tool_calls": [call]},
            {"role": Role.TOOL.value, "tool_call_id": call_id, "content": json.dumps(result)}]


def _load_arguments(raw: Any) -> Any:
    """Tool-call arguments (a JSON string or a mapping) as Python data; None if they cannot be read."""
    try:
        return json.loads(raw or "{}") if isinstance(raw, str) else dict(raw or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _get_thread_id(config: RunnableConfig | None) -> str | None:
    """The conversation's thread_id (None when run without a checkpointer: each call is its own contact)."""
    return ((config or {}).get(CONFIGURABLE) or {}).get(THREAD_ID)


def _get_contact_id(config: RunnableConfig | None) -> str | None:
    """The contact a call belongs to: the conversation's thread_id, else the per-call contact_id run_agent sets."""
    configurable = (config or {}).get(CONFIGURABLE) or {}
    return configurable.get(THREAD_ID) or configurable.get(CONTACT_ID)


def _find_other_patients(tools: CareFlowTools, raw_arguments: Any, bound: str) -> set[str]:
    """Patients other than `bound` that a tool call's arguments point at."""
    arguments = _load_arguments(raw_arguments)
    if not isinstance(arguments, dict):
        return set()  # malformed arguments: the tool's own validation reports them
    return tools.find_referenced_patients(arguments) - {bound}


def _prepare_flag(
    memory: PatientMemory | None, raw: Any, patient_ref: str, done: list[dict], thread_id: str | None
) -> Any:
    """Complete a staff flag on a verified conversation from the records, not from the model's judgement.

    - patient_ref: the verified patient, if the model left it out (a ticket staff cannot attribute is useless).
    - Repeat rules: "Repeat - Priority" (urgency at least Priority) for the third same-issue contact within
      72 hours (OP-CS-014 8.2), "REPEAT" for a same-issue contact within 24 hours (triage SOP 7.3). The
      issue is clinical for a clinical flag, otherwise what this turn's tools worked on; with no known
      issue the reason is left as the model wrote it (the prompt rule still applies).

    `done` is this turn's tool calls so far (list_turn_tool_calls). Returns the arguments as a JSON string, or
    `raw` unchanged when it cannot be read."""
    arguments = _load_arguments(raw)
    if not isinstance(arguments, dict) or not isinstance(arguments.get("reason"), str):
        return raw  # malformed: the tool's validation reports it
    if not arguments.get("patient_ref"):
        arguments["patient_ref"] = patient_ref
    issue = policies.CLINICAL_ISSUE if arguments.get("clinical") else summarise_turn(done)["topic"]
    marker = memory.find_repeat_marker(patient_ref, issue, exclude_thread=thread_id) if memory else None
    if marker:
        reason = _MARKED.sub("", arguments["reason"]).lstrip(" :-–")
        arguments["reason"] = f"{marker}: {reason}"
        if marker == PRIORITY_MARK and arguments.get("urgency", policies.DEFAULT_BAND) == UrgencyBand.ROUTINE:
            arguments["urgency"] = UrgencyBand.PRIORITY.value
    return json.dumps(arguments)


def _has_search_run(call: dict) -> bool:
    """True for a policy search that reached the index. A search refused before it ran, or answered from
    an identical earlier search, does not use up the turn's allowance."""
    result = call["result"]
    refused = ("search_limit" in result or "tools" in result or result.get("error") == OTHER_PATIENT_ERROR
               or result.get("note") == REPEATED_SEARCH_NOTE)
    return call["name"] == ToolName.SEARCH_POLICIES and not refused


def _find_same_search(done: list[dict], raw: Any) -> dict | None:
    """The result of an identical, successful search_policies call earlier in this turn, if there was one."""
    arguments = _load_arguments(raw)
    if not isinstance(arguments, dict):
        return None
    for call in done:
        if call["name"] == ToolName.SEARCH_POLICIES and call["result"].get("ok") and _load_arguments(call["arguments"]) == arguments:
            return call["result"]
    return None


def _mark_hcp_flag(raw: Any, asked: str | None) -> Any:
    """A clinical flag the model raised, marked "Caller states HCP" when the caller's message (`asked`) says
    so (OP-CS-014 8.3). Returns `raw` unchanged when nothing is added."""
    arguments = _load_arguments(raw)
    if not isinstance(arguments, dict) or not arguments.get("clinical") or not isinstance(arguments.get("reason"), str):
        return raw
    marked = policies.mark_hcp(arguments["reason"], asked)
    return raw if marked == arguments["reason"] else json.dumps({**arguments, "reason": marked})


def _find_turn_start(messages: list[dict]) -> int:
    """Index of the latest user message: where the current turn begins."""
    return max((index for index, message in enumerate(messages) if message["role"] == Role.USER), default=0)


def _get_caller_text(messages: list[dict]) -> str | None:
    """What the caller wrote in the current turn."""
    return messages[_find_turn_start(messages)].get("content")


def get_recent_turns(messages: list[dict], turns: int) -> list[dict]:
    """The last `turns` user turns. Cuts only at a user message, so a tool call is never separated
    from its result."""
    starts = [index for index, message in enumerate(messages) if message["role"] == Role.USER]
    if turns and len(starts) > turns:
        return messages[starts[-turns]:]
    return messages


def list_turn_tool_calls(messages: list[dict], start: int | None = None) -> list[dict]:
    """{name, arguments, result} for every tool call from `start` (default: the current turn)."""
    start = _find_turn_start(messages) if start is None else start
    results = {message["tool_call_id"]: json.loads(message["content"])
               for message in messages[start:] if message["role"] == Role.TOOL}
    return [
        {"name": call["function"]["name"], "arguments": call["function"]["arguments"], "result": results[call["id"]]}
        for message in messages[start:]
        for call in message.get("tool_calls", [])
        if call["id"] in results
    ]


def build_agent_graph(
    dependencies: AgentDependencies,
    system_prompt: str = SYSTEM_PROMPT,
    tool_names: set[str] | None = None,
    checkpointer: Any = None,
    max_steps: int = MAX_STEPS,
    history_turns: int = HISTORY_TURNS,
):
    """Compile one agent: the model is offered only `tool_names` (default: every tool this configuration can
    run) under `system_prompt`, sees the last `history_turns` user turns, and is handed to staff once it has
    made `max_steps` model calls in a turn without a final answer. With a `checkpointer` the conversation is
    stored per thread_id."""
    if tool_names is None:  # everything this configuration can run (no policy index -> no policy search)
        tool_names = dependencies.tools.list_available_tools()
    specs = [spec for spec in TOOL_SPECS if spec["function"]["name"] in tool_names]
    allowed = {spec["function"]["name"] for spec in specs}
    documents = list_corpus_documents()  # for the phantom-citation check

    def memory_node(state: CareFlowState, config: RunnableConfig) -> dict:
        """Refresh the memory note. Runs before every model call, so a patient verified by find_patient in
        this turn has their note in front of the model on its next step."""
        if dependencies.memory is None:
            return {}
        # This conversation's own contact row is not an "earlier contact" for the repeat rules.
        return {"patient_memory": dependencies.memory.build_context(state.get("patient_ref"), exclude_thread=_get_thread_id(config))}

    def model_node(state: CareFlowState) -> dict:
        """One model call. Tool calls go into the history; a final answer is only a draft until `ground`."""
        prefix = [{"role": Role.SYSTEM.value, "content": system_prompt}]
        if state.get("patient_memory"):
            prefix.append({"role": Role.SYSTEM.value, "content": state["patient_memory"]})
        reply = dependencies.model([*prefix, *get_recent_turns(state["messages"], history_turns)], specs)
        calls = getattr(reply, "tool_calls", None) or []
        entry: dict = {"role": Role.ASSISTANT.value, "content": reply.content or ""}
        steps = state.get("steps", 0) + 1
        if calls:
            entry["tool_calls"] = [_convert_call_to_dict(call) for call in calls]
            return {"messages": [entry], "steps": steps, "draft": None}
        return {"draft": entry, "steps": steps}  # a final answer is committed by `ground`, after the check

    def _check_grounding(state: CareFlowState, content: str) -> GroundingResult:
        """The grounding check of a draft answer against everything the model was shown. A draft that neither
        follows a policy search nor names a corpus document is not checked (an unchecked result is grounded)."""
        # Sources: the same window model_node sends, not just this turn.
        window = get_recent_turns(state["messages"], history_turns)
        seen = list_turn_tool_calls(window, 0)
        # Checked when the model could be answering from policy passages (a search anywhere in the window,
        # so a follow-up that reuses an earlier search is checked too) or names a corpus document.
        searches = [call for call in seen if call["name"] == ToolName.SEARCH_POLICIES]
        if not searches and not has_document_name(content, documents):
            return GroundingResult()
        return check_answer(
            content,
            passages=[passage for call in searches for passage in call["result"].get("results", [])],
            # Only the scripts the model must say word for word, not the whole system prompt: its internal
            # rules (24 / 72 hours, OP-CS-014) would otherwise ground any answer.
            trusted_sources=[DEFLECTION_SCRIPT, EMERGENCY_STATEMENT, state.get("patient_memory") or "",
                             *(message["content"] for message in window
                               if message["role"] == Role.ASSISTANT and message.get("content"))],
            tool_results=[call["result"] for call in seen if call["name"] != ToolName.SEARCH_POLICIES],
            user_messages=[message["content"] for message in window if message["role"] == Role.USER],
            known_documents=documents,
        )

    def ground_node(state: CareFlowState, config: RunnableConfig) -> dict:
        """Commit the draft as the answer if it is supported; otherwise withhold it and hand to staff."""
        draft = state["draft"]
        result = _check_grounding(state, draft.get("content") or "")
        if result.grounded:
            final = [draft]
        else:
            # Withhold the answer: hand to staff and say so (an unsupported coverage figure is worse than
            # none). The reason names no withheld figure: the flag call is part of the history the model
            # sees next turn. Which figures failed is in state["grounding"] (AgentResult.grounding).
            final = [*_raise_staff_flag(state, config, GROUNDING_REASON, policies.DEFAULT_BAND, False, "ground-1"),
                     {"role": Role.ASSISTANT.value, "content": GROUNDING_ANSWER}]
        return {"messages": _add_clinical_ticket(state, config, final), "draft": None, "grounding": result.to_dict()}

    def _raise_staff_flag(state: CareFlowState, config: RunnableConfig, reason: str, urgency: str, clinical: bool,
                          call_id: str) -> list[dict]:
        """A flag raised by the graph itself, as history messages (see _build_flag_messages). It goes through
        dependencies.tools.call, like a tool the model asked for, so tracing and the resilience wrappers see it."""
        arguments: dict = {"reason": reason, "urgency": urgency, **({"clinical": True} if clinical else {})}
        raw: Any = arguments
        bound = state.get("patient_ref")
        if bound:  # patient_ref and the repeat rules (OP-CS-014 8.2) from the records
            raw = _prepare_flag(dependencies.memory, json.dumps(arguments), bound, list_turn_tool_calls(state["messages"]),
                                _get_thread_id(config))
        context = {CONTACT_ID: _get_contact_id(config)}
        flag = dependencies.tools.call(ToolName.FLAG_FOR_HUMAN.value, raw, context=context)
        if not flag.get("ok") and bound:
            # e.g. an unknown caller-supplied patient_ref: the patient is told staff will follow up, so the
            # ticket must exist. Flag without the reference, keeping it in the reason.
            raw = {**arguments, "reason": f"{reason} (unverified patient reference: {bound})"}
            flag = dependencies.tools.call(ToolName.FLAG_FOR_HUMAN.value, raw, context=context)
        return _build_flag_messages(call_id, raw if isinstance(raw, str) else json.dumps(raw), flag)

    def _add_clinical_ticket(state: CareFlowState, config: RunnableConfig, final: list[dict]) -> list[dict]:
        """OP-CS-014 9.6: where a message triggers deflection, the system must create the Clinical Routing
        Ticket automatically. If the answer deflects (or gives the emergency statement) and no clinical flag
        was raised this turn, raise it here: the model can say the script without calling flag_for_human.
        Returns `final` with the ticket's messages in front of it, or unchanged."""
        answer = final[-1].get("content")
        emergency = has_emergency_statement(answer)
        if not (emergency or is_deflection(answer)):
            return final
        raised = any(call["name"] == ToolName.FLAG_FOR_HUMAN and call["result"].get("ok")
                     and call["result"].get("category") == FlagCategory.CLINICAL
                     for call in list_turn_tool_calls(state["messages"]))
        if raised:
            return final
        urgency = UrgencyBand.EMERGENT.value if emergency else policies.DEFAULT_BAND
        reason = policies.mark_hcp(CLINICAL_TICKET_REASON, _get_caller_text(state["messages"]))  # OP-CS-014 8.3
        return [*_raise_staff_flag(state, config, reason, urgency, True, "ticket-1"), *final]

    def tools_node(state: CareFlowState, config: RunnableConfig) -> dict:
        """Run the tool calls of the last model message, one result message per call. A call is refused
        without running when the tool is not this agent's, it points at another patient, or it is one policy
        search too many; a staff flag is completed from the records before it is raised."""
        out, update = [], {}
        bound = state.get("patient_ref")  # the verified patient, if any
        done = list_turn_tool_calls(state["messages"])  # this turn's earlier tool calls, for the repeat marker
        searches = sum(_has_search_run(earlier_call) for earlier_call in done)  # searches that ran this turn
        for call in state["messages"][-1]["tool_calls"]:
            name, raw = call["function"]["name"], call["function"]["arguments"]
            if name not in allowed:  # the model asked for a tool this agent does not have
                result = {"ok": False, "error": f"tool {name!r} is not available to this agent", "tools": sorted(allowed)}
            elif bound and _find_other_patients(dependencies.tools, raw, bound):
                # Deterministic, not left to the prompt: a verified conversation only touches its own
                # patient's records (data-handling standard, minimum necessary).
                result = {"ok": False, "error": OTHER_PATIENT_ERROR}
            elif name == ToolName.SEARCH_POLICIES and (earlier := _find_same_search(done, raw)) is not None:
                # The same search again returns the same passages. Say so, or the model can repeat it until
                # the step limit.
                result = {**earlier, "note": REPEATED_SEARCH_NOTE}
            elif name == ToolName.SEARCH_POLICIES and searches >= MAX_SEARCHES:
                result = {"ok": False, "error": SEARCH_LIMIT_NOTE, "search_limit": MAX_SEARCHES}
            else:
                if name == ToolName.FLAG_FOR_HUMAN:
                    raw = _mark_hcp_flag(raw, _get_caller_text(state["messages"]))
                if name == ToolName.FLAG_FOR_HUMAN and bound:
                    raw = _prepare_flag(dependencies.memory, raw, bound, done, _get_thread_id(config))
                result = dependencies.tools.call(name, raw, context={CONTACT_ID: _get_contact_id(config)})
                searches += name == ToolName.SEARCH_POLICIES
                done.append({"name": name, "arguments": raw, "result": result})
                if name == ToolName.FIND_PATIENT and result.get("found"):
                    if bound and result["patient_ref"] != bound:
                        result = {"ok": False, "error": OTHER_PATIENT_ERROR}
                    else:
                        update["patient_ref"] = bound = result["patient_ref"]  # verified: memory loads next
            out.append({"role": Role.TOOL.value, "tool_call_id": call["id"], "content": json.dumps(result)})
        return {"messages": out, **update}

    def escalate_node(state: CareFlowState, config: RunnableConfig) -> dict:
        """Step cap reached: never leave the patient with silence. Hand to staff and say so."""
        # Unlike _raise_staff_flag, the reference is passed as given and no repeat marker is added.
        patient_ref = state.get("patient_ref")
        arguments = {"reason": ESCALATION_REASON, "urgency": policies.DEFAULT_BAND, "patient_ref": patient_ref}
        context = {CONTACT_ID: _get_contact_id(config)}
        result = dependencies.tools.call(ToolName.FLAG_FOR_HUMAN.value, arguments, context=context)
        if not result.get("ok") and patient_ref:
            # e.g. the caller's patient_ref is unknown. The handoff must still reach staff, so flag it
            # without the reference and keep the unverified value in the reason.
            arguments = {"reason": f"{ESCALATION_REASON} (unverified patient reference: {patient_ref})",
                    "urgency": policies.DEFAULT_BAND}
            result = dependencies.tools.call(ToolName.FLAG_FOR_HUMAN.value, arguments, context=context)
        return {
            "messages": [*_build_flag_messages("escalate-1", json.dumps(arguments), result),
                         {"role": Role.ASSISTANT.value, "content": ESCALATION_ANSWER}],
            "escalated": True,
        }

    def record_node(state: CareFlowState, config: RunnableConfig) -> dict:
        """Write this turn to the patient's long-term history (the memory itself skips an unverified caller)."""
        if dependencies.memory is not None:
            messages = state["messages"]
            dependencies.memory.record_turn(
                state.get("patient_ref"),
                list_turn_tool_calls(messages),
                channel=state.get("channel") or DEFAULT_CHANNEL,
                thread_id=_get_thread_id(config),
                answer=messages[-1].get("content") or "",
            )
        return {}

    # The node names as the plain strings LangGraph stores in checkpoints.
    memory, model, tools = AgentNode.MEMORY.value, AgentNode.MODEL.value, AgentNode.TOOLS.value
    ground, escalate, record = AgentNode.GROUND.value, AgentNode.ESCALATE.value, AgentNode.RECORD.value

    def after_model(state: CareFlowState) -> str:
        return ground if state.get("draft") else tools

    def after_tools(state: CareFlowState) -> str:
        return escalate if state.get("steps", 0) >= max_steps else memory

    graph = StateGraph(CareFlowState)
    graph.add_node(memory, memory_node)
    graph.add_node(model, model_node)
    graph.add_node(tools, tools_node)
    graph.add_node(escalate, escalate_node)
    graph.add_node(ground, ground_node)
    graph.add_node(record, record_node)
    graph.add_edge(START, memory)
    graph.add_edge(memory, model)
    graph.add_conditional_edges(model, after_model, {tools: tools, ground: ground})
    graph.add_edge(ground, record)
    graph.add_conditional_edges(tools, after_tools, {memory: memory, escalate: escalate})
    graph.add_edge(escalate, record)
    graph.add_edge(record, END)
    return graph.compile(checkpointer=checkpointer)
