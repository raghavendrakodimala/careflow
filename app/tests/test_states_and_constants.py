"""The enums and named constants behind the states and categories.

Three things are proved here:
- an enum never changes what is stored or sent: after a triage run and a chat turn, every string in the
  graph state, the checkpoints, the SQLite rows, the trace spans and the API responses is exactly `str`;
- data written before the enums existed (plain strings) still works: a case paused then can be listed,
  fetched and completed now;
- each `Literal` that had to stay a Literal (it is part of a schema sent to a model or published by the API)
  lists exactly its enum's values, and no message changed because an enum was formatted with repr().
"""

import dataclasses
import json
import re
import sqlite3
from enum import Enum, StrEnum
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from src import config, guardrails, memory, reliability, schemas, state, store, telemetry, tool_contracts
from src.agent import close_checkpointer, open_session_checkpointer
from src.agents import INSURANCE, REFERRAL, REVIEWER, SafetyCategory, SafetyVerdict, TeamResult, run_team
from src.api.app import CHAT_PREFIX, Resources, create_app
from src.config import Settings
from src.domain import policies
from src.graph import AgentNode, build_agent_graph
from src.guardrails import ClinicalRisk
from src.mcp import EHR_TOOLS
from src.rag.chunker import SOURCES, load_corpus_chunks
from src.reliability import BreakerState, CircuitBreaker, ModelUnavailableError, ResilientTools, make_resilient_model
from src.state import Role, build_thread_config
from src.telemetry import JsonlSink, MemorySink, SpanKind, SpanStatus, Tracer
from src.tool_contracts import TOOL_SPECS, TOOLS, Backend, ToolName
from src.workflows import NODES, build_triage_graph, get_pending_review, resume_triage
from src.workflows.state import CaseStatus, ReviewAction, ReviewDecision
from src.workflows.triage import TriageNode

from .test_workflow import Harness, _find_patient, _make_call, _make_message

ENUM_REPR = re.compile(r"<\w+\.\w+: ")  # what repr() of an enum member looks like, e.g. <Route.AUTO: 'auto'>


def not_plain(value, path="$"):
    """The path of every string that is not exactly `str` (an enum member or any other subclass), of every
    enum member, and of every string that holds an enum's repr(), anywhere inside `value`."""
    if isinstance(value, Enum):
        yield f"{path} = {value!r}"
    elif isinstance(value, str):
        if type(value) is not str:
            yield f"{path} is {type(value).__name__}"
        elif ENUM_REPR.search(value):
            yield f"{path} holds an enum repr: {value[:80]}"
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from not_plain(key, f"{path}.<key>")
            yield from not_plain(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            yield from not_plain(item, f"{path}[{index}]")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from not_plain(getattr(value, field.name), f"{path}.{field.name}")
    elif hasattr(value, "model_dump"):
        yield from not_plain(value.model_dump(), path)


def assert_plain(value, what: str) -> None:
    found = list(not_plain(value))
    assert not found, f"{what}: {found[:5]}"


def test_the_checker_itself_catches_members_and_reprs():
    assert list(not_plain({"route": "auto", "n": 1, "nested": [("a", None)]})) == []
    assert len(list(not_plain({"route": policies.Route.AUTO}))) == 1
    assert len(list(not_plain({policies.Route.AUTO: "x"}))) == 1
    assert len(list(not_plain([f"{policies.Route.AUTO!r}"]))) == 1
    assert len(list(not_plain(TeamResult(answer="", agents=[Role.USER])))) == 1


# ---- the enums and the names they replaced ----------------------------------------------------------
def test_members_equal_hash_and_format_as_their_values():
    assert policies.Route.AUTO == "auto" and policies.Route.REFUSE == "refuse"
    assert {policies.Route.HUMAN_REVIEW: 1}["human_review"] == 1 and {"open": 1}[BreakerState.OPEN] == 1
    assert f"{policies.Route.AUTO}" == "auto" and str(policies.UrgencyBand.EMERGENT) == "Emergent"
    assert json.dumps({"route": policies.Route.REFUSE, "role": Role.USER}) == '{"route": "refuse", "role": "user"}'
    assert type(policies.Route.AUTO.value) is str and type(str(policies.Route.AUTO)) is str


def test_the_earlier_constant_names_still_hold_the_same_values():
    assert (policies.ROUTE_AUTO, policies.ROUTE_HUMAN, policies.ROUTE_REFUSE) == ("auto", "human_review", "refuse")
    assert (reliability.CLOSED, reliability.OPEN, reliability.HALF_OPEN) == ("closed", "open", "half_open")
    from src.workflows.state import AWAITING, COMPLETED

    assert (AWAITING, COMPLETED) == ("awaiting_human_review", "completed")
    assert state.REDACT == "redact" and policies.CLINICAL_ISSUE == "clinical_question"
    assert memory.NO_ISSUE == "general_enquiry"
    assert policies.URGENCY_ORDER == ("Routine", "Priority", "Urgent", "Emergent")
    assert memory.OPEN_REFERRAL_STATUSES == ("open", "accepted", "scheduled")
    assert guardrails.TIME_LIMIT_S == config.GUARD_TIME_LIMIT_S == 2.0
    assert Settings.model_fields["careflow_guard_timeout_s"].default == guardrails.TIME_LIMIT_S
    breaker = CircuitBreaker("x")
    assert (breaker.failure_threshold, breaker.cooldown_s) == (
        Settings.model_fields["careflow_breaker_failures"].default,
        Settings.model_fields["careflow_breaker_cooldown_s"].default) == (3, 30.0)
    assert reliability.RetryPolicy().attempts == Settings.model_fields["careflow_tool_retries"].default == 3


def values(enum: type[StrEnum]) -> tuple[str, ...]:
    return tuple(member.value for member in enum)


def test_every_literal_that_stayed_a_literal_lists_its_enum():
    # These fields are part of a JSON schema sent to a model or published by the API, so they stay Literals.
    assert get_args(schemas.Urgency) == values(policies.UrgencyBand)
    assert get_args(schemas.Channel) == values(schemas.ContactChannel)
    assert get_args(schemas.MissingField) == values(schemas.MissingFieldName)
    mode = tool_contracts.FindAppointmentSlotsInput.model_fields["mode"].annotation
    assert get_args(get_args(mode)[0]) == values(policies.AppointmentMode)
    assert get_args(SafetyVerdict.model_fields["category"].annotation) == values(SafetyCategory)
    assert get_args(ReviewDecision.model_fields["action"].annotation) == values(ReviewAction)
    # The classifier may only answer with three of the gate's categories.
    assert set(get_args(ClinicalRisk.model_fields["category"].annotation)) <= set(values(guardrails.GuardCategory))


def test_enums_agree_with_the_tables_built_from_them():
    assert [tool.name for tool in TOOLS] == list(values(ToolName))
    assert {tool.backend for tool in TOOLS} == set(values(Backend))
    assert set(reliability.BACKENDS.values()) == {"ehr", "policy_index"}
    assert set(EHR_TOOLS) == {tool.name for tool in TOOLS if tool.backend == Backend.EHR}
    assert values(TriageNode) == NODES
    assert set(memory.TOPIC_SUMMARY) == set(values(policies.ContactTopic))
    assert list(values(memory.ContactOutcome)) == memory.OUTCOME_RANK
    assert f"agent.{REVIEWER.name}" == telemetry.REVIEWER_SPAN
    # A reviewed flag's status is built from the action: there is a FlagStatus for each.
    assert {f"reviewed_{action}" for action in values(ReviewAction)} <= set(values(store.FlagStatus))
    assert config._MODES == {"careflow_tracing": ("off", "jsonl", "langfuse"),
                             "careflow_guard_classifier": ("none", "local", "hosted"),
                             "careflow_policy_source": ("pdf", "markdown")}
    assert (Settings().careflow_tracing, Settings().careflow_policy_source) != ("", "")


def test_the_trusted_context_key_is_the_flag_tools_parameter():
    import inspect

    from src.tools import CareFlowTools

    assert tool_contracts.CONTACT_ID in inspect.signature(CareFlowTools.flag_for_human).parameters
    assert tool_contracts.CONTACT_ID not in tool_contracts.FlagForHumanInput.model_fields  # never the model's to set


def test_what_is_handed_to_models_and_frameworks_is_plain_text():
    assert_plain(TOOL_SPECS, "tool specs")
    assert_plain([(tool.name, tool.backend, tool.description) for tool in TOOLS], "tool definitions")
    assert_plain([EHR_TOOLS, NODES, SOURCES, policies.URGENCY_ORDER, policies.DEFAULT_BAND, schemas.DEFAULT_CHANNEL,
                  list(INSURANCE.tools), list(REFERRAL.tools), config._MODES, telemetry.LangfuseSink.KINDS,
                  reliability.BACKENDS, [node.value for node in AgentNode]], "module-level tables")
    assert_plain(Settings(careflow_tracing="", careflow_guard_classifier=" LOCAL ").model_dump(), "settings")
    assert_plain(dataclasses.asdict(guardrails.ALLOWED), "the default guard result")
    assert_plain(state.build_thread_config("t"), "run config")


# ---- no message changed -----------------------------------------------------------------------------
def test_messages_that_list_allowed_values_read_as_before():
    from src.tools import CareFlowTools

    tools = CareFlowTools()
    assert tools.flag_for_human("x", urgency="High")["error"] == (
        "urgency must be one of ['Routine', 'Priority', 'Urgent', 'Emergent']")
    assert tools.find_appointment_slots("Cardiology", mode="video")["error"] == "mode must be 'in_person' or 'telehealth'"
    assert tools.find_appointment_slots("Cardiology", mode="telehealth")["ok"] is True
    with pytest.raises(ValueError, match=re.escape("source must be one of ('markdown', 'pdf')")):
        load_corpus_chunks(source="docx")
    with pytest.raises(ValueError, match=re.escape("must be one of off, jsonl, langfuse (got 'loud')")):
        Settings(careflow_tracing="loud")
    breaker = CircuitBreaker("model", failure_threshold=1)
    breaker.record_failure()
    with pytest.raises(ModelUnavailableError, match=r"^model circuit is open$"):
        make_resilient_model(lambda messages, specs: None, breaker)([], [])
    assert policies.validate_reband("Emergent", "Urgent", "note") == (
        "an Emergent case cannot be lowered into a referral band (referral policy 3.3)")
    assert policies.decide_booking_payment("lapsed", True, "S1").staff_reason == (
        "Self-pay booking S1 (coverage lapsed); payment due at check-in")


# ---- nothing stored or sent is an enum ---------------------------------------------------------------
UNSAFE = SafetyVerdict(safe=False, category="medication", reason="names a dose")


def _traced(harness: Harness) -> MemorySink:
    """Put the production wrappers (retries, breakers, spans) around the harness's tools and model."""
    sink = MemorySink()
    tracer = Tracer([sink])
    harness.dependencies.agent.tracer = tracer
    harness.dependencies.agent.tools = ResilientTools(harness.tools, tracer=tracer)
    harness.dependencies.agent.model = make_resilient_model(harness._call_model, tracer=tracer)
    return sink


def _triage_cases():
    """name -> (harness arguments, reviewer, raw text, whether the caller is a known patient)."""
    flag = _make_message(calls=[_make_call("c1", "flag_for_human", reason="Please call me about my bill", urgency="Priority")])
    looping = [_make_message(calls=[_make_call(f"c{index}", "check_eligibility", patient_ref="MHP-P-00001")]) for index in range(8)]
    return {
        "auto": ({}, None, "I'd like a dermatology appointment.", True),
        "urgent": ({"parsed": {"urgency": "Urgent"}}, None, "I need to be seen soon.", True),
        "emergent": ({"parsed": {"urgency": "Emergent"}}, None, "I need to be seen now.", True),
        "clinical by parse": ({"parsed": {"seeks_clinical_advice": True}}, None, "Tell me about my tablets.", True),
        "blocked at the gate": ({}, None, "I have chest pain. Should I take aspirin?", True),
        "unreadable": ({"parse_error": RuntimeError("no model")}, None, "hello", False),
        "unknown caller": ({}, None, "I'd like a dermatology appointment.", False),
        "no specialty": ({"parsed": {"specialty": None}}, None, "Can someone help me?", True),
        "assistant flags": ({"steps": [flag, _make_message("Staff will call you.")]}, None, "Please call me about my bill.", True),
        "step cap": ({"steps": looping}, None, "Is my plan active?", True),
        "vetoed": ({"steps": [_make_message("Take two tablets twice a day.")]}, lambda message, reply: UNSAFE,
                   "I'd like a dermatology appointment.", True),
        "model down": ({"steps": []}, None, "I'd like a dermatology appointment.", True),
    }


@pytest.mark.parametrize("name", list(_triage_cases()))
def test_a_triage_run_stores_and_returns_only_plain_strings(name, tmp_path):
    kwargs, reviewer, text, known = _triage_cases()[name]
    harness = Harness(**kwargs)
    if name == "model down":
        harness.dependencies.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
    harness.checkpointer = open_session_checkpointer(tmp_path / "sessions.db")  # the real SQLite checkpointer
    sink = _traced(harness) if name != "model down" else MemorySink()
    harness.dependencies.review = reviewer
    try:
        from src.workflows import start_triage

        record = {"channel": "sms", "received_at": "2026-08-01T09:00:00Z", "raw_text": text,
                  "patient_ref": _find_patient(harness.tools) if known else None}
        result = start_triage(record, harness.dependencies, harness.checkpointer, thread_id="case-1")
        graph = build_triage_graph(harness.dependencies, harness.checkpointer)
        config_ = build_thread_config("case-1")
        assert_plain(result, "TriageResult")
        assert_plain(graph.get_state(config_).values, "state values")
        saved = harness.checkpointer.get_tuple(config_)
        assert_plain([saved.checkpoint, saved.metadata, saved.pending_writes], "checkpoint")
        if result.status == CaseStatus.AWAITING_HUMAN_REVIEW:
            assert_plain(get_pending_review("case-1", harness.dependencies, harness.checkpointer), "review payload")
            assert_plain(harness.tools.store.list_pending_reviews(), "review queue")
            done = resume_triage("case-1", {"reviewer_id": "RN-1", "action": "approve"}, harness.dependencies, harness.checkpointer)
            assert done.status == "completed"
            assert_plain(done, "TriageResult after the decision")
            assert_plain(graph.get_state(config_).values, "state values after the decision")
        else:
            assert (result.status, result.route) == ("completed", "auto")
        for snapshot in graph.get_state_history(config_):
            assert_plain([snapshot.values, snapshot.next, snapshot.metadata], "state history")
        assert_plain(sink.spans, "trace spans")
        assert {span["kind"] for span in sink.spans} <= set(values(SpanKind))
        assert {span["status"] for span in sink.spans} <= set(values(SpanStatus))
        _assert_rows_are_text(harness.tools.store.connection, "runtime store")
        _assert_no_class_paths(harness.checkpointer.conn)
    finally:
        close_checkpointer(harness.checkpointer)
    expected = {"auto": "auto", "urgent": "human_review", "emergent": "refuse", "clinical by parse": "refuse",
                "blocked at the gate": "refuse", "unreadable": "refuse", "unknown caller": "human_review",
                "no specialty": "human_review", "assistant flags": "human_review", "step cap": "human_review",
                "vetoed": "refuse", "model down": "human_review"}
    assert result.route == expected[name]


def _assert_rows_are_text(connection: sqlite3.Connection, what: str) -> None:
    """Every value in the tables the application writes is stored as text (or NULL), and reads back as str."""
    for table in ("staff_flags", "reviews", "patient_interactions", "appointments", "appointment_slots"):
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        for column in columns:
            kinds = {row[0] for row in connection.execute(f'SELECT DISTINCT typeof("{column}") FROM {table}')}
            assert kinds <= {"text", "null"}, f"{what}: {table}.{column} holds {kinds}"
        rows = [dict(zip(columns, row, strict=True)) for row in connection.execute(f"SELECT * FROM {table}")]
        assert_plain(rows, f"{what}: {table}")
    stored = {table: {row[0] for row in connection.execute(f"SELECT DISTINCT {column} FROM {table}")}
              for table, column in (("staff_flags", "status"), ("staff_flags", "category"), ("reviews", "status"),
                                    ("reviews", "route"), ("appointment_slots", "status"))}
    assert stored["staff_flags"] <= set(values(store.FlagCategory)) | set(values(store.FlagStatus))
    assert stored["reviews"] <= set(values(store.ReviewStatus)) | set(values(policies.Route))
    assert stored["appointment_slots"] <= set(values(store.SlotStatus))


def _assert_no_class_paths(connection: sqlite3.Connection) -> None:
    """A checkpoint that held an enum member would name its class (module path and class name) in the saved
    bytes. None of the application's modules or enum classes may appear there."""
    enum_names = [candidate.__name__.encode() for module in (policies, schemas, state, store, telemetry, tool_contracts,
                                                       reliability, guardrails, memory, config)
                  for candidate in vars(module).values() if isinstance(candidate, type) and issubclass(candidate, Enum)]
    enum_names += [b"AgentNode", b"TriageNode", b"CaseStatus", b"ReviewAction", b"SafetyCategory"]
    tables = [table_row[0] for table_row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert tables
    for table in tables:
        for row in connection.execute(f"SELECT * FROM {table}"):
            for value in row:
                blob = value.encode() if isinstance(value, str) else value
                if isinstance(blob, bytes):
                    assert b"src." not in blob and b"enum" not in blob, f"{table} names a class path"
                    assert not [enum_name for enum_name in enum_names if enum_name in blob], f"{table} names an enum class"


def _chat_cases(tools):
    ref = _find_patient(tools, status="lapsed")
    slot = tools.find_appointment_slots("Dermatology", patient_ref=ref)["slots"][0]["slot_id"]
    booking = [_make_message(calls=[_make_call("c1", "check_eligibility", patient_ref=ref)]),
               _make_message(calls=[_make_call("c2", "find_appointment_slots", specialty="Dermatology", patient_ref=ref)]),
               _make_message("Text beside a call.", calls=[_make_call("c3", "book_appointment", patient_ref=ref, slot_id=slot,
                                                        self_pay=True)]),
               _make_message("You are booked; payment is due at check-in.")]
    flagging = [_make_message(calls=[_make_call("c1", "flag_for_human", reason="Wants a call about a statement")]),
                _make_message("Staff will call you.")]
    identify = [_make_message(calls=[_make_call("c1", "find_patient", given_name="No", family_name="Body", birth_date="1980-01-01")]),
                _make_message("I could not find your record.")]
    return ref, {
        "booking": (booking, None, "Please book me a dermatology appointment, I will pay myself."),
        "flag": (flagging, None, "Can someone call me about my statement?"),
        "identify": (identify, None, "Is my plan active?"),
        "deflection": ([_make_message(guardrails.DEFLECTION_SCRIPT)], None, "Is my plan active?"),
        "blocked": ([_make_message("never reached")], None, "I have chest pain, what should I do?"),
        "vetoed": ([_make_message("Take two tablets twice a day.")], lambda message, reply: UNSAFE, "Is my plan active?"),
        "step cap": ([_make_message(calls=[_make_call(f"c{index}", "check_eligibility", patient_ref=ref)]) for index in range(8)], None,
                     "Is my plan active?"),
    }


@pytest.mark.parametrize("name", ["booking", "flag", "identify", "deflection", "blocked", "vetoed", "step cap"])
def test_a_chat_turn_stores_and_returns_only_plain_strings(name, tmp_path):
    harness = Harness()
    ref, cases = _chat_cases(harness.tools)
    harness.steps, reviewer, message = cases[name]
    sink = _traced(harness)
    checkpointer = open_session_checkpointer(tmp_path / "sessions.db")
    thread = CHAT_PREFIX + "t1"
    try:
        routes = []
        for turn in range(2):  # the second turn reads the first one back from the checkpoint
            harness.offered.clear()
            team = run_team(message, harness.dependencies.agent, reviewer, patient_ref=None if name == "identify" else ref,
                            channel="email", thread_id=thread, checkpointer=checkpointer, guard=True,
                            classifier=lambda text: ClinicalRisk(score=0.0))
            assert_plain(team, f"TeamResult, turn {turn}")
            assert_plain({"route": team.route, "tools_used": team.tools_used, "flags": team.flags}, "team properties")
            routes.append(team.route)
            graph = build_agent_graph(harness.dependencies.agent, checkpointer=checkpointer)
            saved = checkpointer.get_tuple(build_thread_config(thread))
            if name != "blocked":  # a blocked message never reaches an agent, so nothing is stored
                snapshot = graph.get_state(build_thread_config(thread))
                assert_plain([snapshot.values, snapshot.next, snapshot.metadata], f"state, turn {turn}")
                assert {saved_message["role"] for saved_message in snapshot.values["messages"]} <= {"user", "assistant", "tool"}
                assert_plain([saved.checkpoint, saved.metadata, saved.pending_writes], "checkpoint")
            else:
                assert saved is None
        # The self-pay booking raises a staff flag of its own, so staff see it.
        assert routes[0] == {"booking": "human_review", "flag": "human_review", "identify": "auto", "deflection": "refuse",
                             "blocked": "refuse", "vetoed": "refuse", "step cap": "human_review"}[name]
        if name == "booking":  # booked once, as self-pay, with the staff flag the booking raises
            booked = [appointment for appointment in harness.tools.store.list_rows("appointments") if appointment["slot_id"]]
            assert [(appointment["status"], appointment["payment"]) for appointment in booked] == [("scheduled", "self_pay")]
            assert [(flag["status"], flag["category"], flag["urgency"]) for flag in harness.tools.flags] == [
                ("awaiting_staff", "administrative", "Routine")]
        if name == "vetoed":  # the reducer's instruction is consumed, never stored
            assert all(saved_message["role"] != "redact" for saved_message in snapshot.values["messages"])
        assert_plain(sink.spans, "trace spans")
        assert {span["kind"] for span in sink.spans} <= set(values(SpanKind))
        assert_plain(harness.tools.flags, "flags")
        _assert_rows_are_text(harness.tools.store.connection, "runtime store")
        _assert_no_class_paths(checkpointer.conn)
    finally:
        close_checkpointer(checkpointer)


def test_what_the_store_hands_back_is_plain_text_even_when_given_members():
    runtime_store = store.RuntimeStore(":memory:")
    try:
        flag, duplicate = runtime_store.add_flag(None, "reason", policies.DEFAULT_BAND, "2026-08-01T09:00")
        review = runtime_store.add_review("t1", policies.Route.REFUSE.value, ["why"], "2026-08-01T09:00", flag_id=flag["flag_id"])
        decided = runtime_store.decide_review(review["review_id"], "RN-1", ReviewAction.APPROVE.value, "2026-08-01T09:05")
        slot = runtime_store.list_rows("appointment_slots")[0]["slot_id"]
        appointment, _ = runtime_store.book(slot, "MHP-P-00001", policies.PaymentKind.SELF_PAY.value)
        assert_plain([flag, review, decided, appointment, runtime_store.get_flag(flag["flag_id"])], "store results")
        assert (flag["status"], flag["category"], duplicate) == ("awaiting_staff", "administrative", False)
        assert (review["status"], decided["status"], decided["action"]) == ("pending", "decided", "approve")
        assert runtime_store.get_flag(flag["flag_id"])["status"] == "reviewed_approve"
        assert (appointment["status"], runtime_store.find_one("appointment_slots", "slot_id", slot)["status"]) == ("scheduled", "booked")
        _assert_rows_are_text(runtime_store.connection, "store")
    finally:
        runtime_store.close()


def test_memory_summaries_and_rows_are_plain_text():
    from src.tools import CareFlowTools

    tools = CareFlowTools()
    patient_memory = memory.PatientMemory(tools.store, tools.clock)
    clinical = [{"name": "flag_for_human", "arguments": {"clinical": True},
                 "result": {"ok": True, "flag_id": "FLAG-0001", "urgency": "Routine", "category": "clinical"}}]
    search = [{"name": "search_policies", "arguments": "{}", "result": {"ok": True, "count": 0}}]
    turns = [memory.summarise_turn(calls) for calls in ([], clinical, search)]
    assert_plain(turns, "turn summaries")
    assert [(turn["topic"], turn["outcome"]) for turn in turns] == [
        ("general_enquiry", "resolved"), ("clinical_question", "escalated_to_clinician"), ("policy_question", "resolved")]
    first = patient_memory.record_turn("MHP-P-00001", clinical, thread_id="t1")
    merged = patient_memory.record_turn("MHP-P-00001", search, thread_id="t1")
    admin = {"flag_id": "FLAG-0002", "urgency": "Routine", "category": "administrative"}
    by_member = patient_memory.record_outcome("MHP-P-00001", policies.ContactTopic.BILLING_QUESTION, admin, thread_id="t2")
    assert_plain([first, merged, by_member, patient_memory.recall("MHP-P-00001")], "memory rows")
    assert (merged["topic"], merged["outcome"]) == ("clinical_question", "escalated_to_clinician")
    assert (by_member["topic"], by_member["outcome"]) == ("billing_question", "callback_scheduled")
    _assert_rows_are_text(tools.store.connection, "memory")


def test_jsonl_trace_lines_hold_only_plain_strings(tmp_path):
    harness = Harness(parsed={"urgency": "Urgent"})
    path = tmp_path / "traces.jsonl"
    tracer = Tracer([JsonlSink(path)])
    harness.dependencies.agent.tracer = tracer
    harness.dependencies.agent.tools = ResilientTools(harness.tools, tracer=tracer)
    result = harness.start(_find_patient(harness.tools), thread_id="case-1")
    harness.resume(result.thread_id)
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    assert len(lines) > 10 and not ENUM_REPR.search(text)
    assert_plain(lines, "trace lines")
    assert {"trace", "span", "tool"} <= {line["kind"] for line in lines}
    assert {"ok", "paused"} == {line["status"] for line in lines}
    assert {line["breaker"] for line in lines if "breaker" in line} == {"closed"}
    root = next(line for line in lines if line["kind"] == "trace" and line["name"] == "triage")
    assert (root["route"], root["case_status"], root["band"]) == ("human_review", "awaiting_human_review", "Urgent")


def _client(harness: Harness, **settings) -> TestClient:
    harness.dependencies.review = lambda message, reply: SafetyVerdict(safe=True)
    resources = Resources(harness.dependencies, harness.checkpointer, classifier=lambda text: ClinicalRisk(score=0.0),
                          settings=Settings(careflow_tracing="off", **settings))
    return TestClient(create_app(resources))


def test_api_responses_hold_only_plain_strings():
    harness = Harness(parsed={"urgency": "Urgent"},
                steps=[_make_message(calls=[_make_call("c1", "check_eligibility", patient_ref="MHP-P-00001")]), _make_message("Your plan is active.")])
    harness.dependencies.agent.tools = ResilientTools(harness.tools)  # so /health reports the breakers
    ref = _find_patient(harness.tools)
    with _client(harness) as client:
        responses = [
            client.get("/health"),
            client.post("/v1/triage", json={"raw_text": "I need to be seen soon.", "patient_ref": ref, "thread_id": "c-1"}),
            client.post("/v1/triage", json={"raw_text": "Should I take aspirin for this?", "thread_id": "c-2"}),
            client.get("/v1/reviews"),
            client.get("/v1/reviews/c-1"),
            client.post("/v1/reviews/c-1/decision", json={"reviewer_id": "RN-1", "action": "approve"}),
            client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": "MHP-P-00001"}),
            client.post("/v1/chat", json={"message": "I have chest pain, what should I do?"}),
            client.post("/v1/triage", json={"raw_text": "x", "thread_id": "chat:1"}),
            client.post("/v1/triage", json={"raw_text": "x", "channel": "pigeon"}),
        ]
    for response in responses:
        assert not ENUM_REPR.search(response.text), response.text
        assert_plain(response.json(), response.request.url.path)
    health, urgent, clinical, queue, payload, decided, chat, emergency, chat_thread, bad_channel = (response.json() for response in responses)
    assert health["breakers"] == {"ehr": "closed", "policy_index": "closed"}
    assert (urgent["route"], urgent["status"], urgent["band"]) == ("human_review", "awaiting_human_review", "Urgent")
    assert (clinical["route"], clinical["guard"]["category"], clinical["guard"]["layer"]) == ("refuse", "clinical", "rules")
    assert queue["count"] == 2 and {review["status"] for review in queue["reviews"]} == {"pending"}
    assert payload["route"] == "human_review" and (decided["status"], decided["decision"]["action"]) == ("completed", "approve")
    assert (chat["route"], chat["tools_used"], chat["verdict"]["category"]) == ("auto", ["check_eligibility"], "none")
    assert (emergency["route"], emergency["guard"]["category"]) == ("refuse", "emergency")
    assert chat_thread["detail"] == "thread_id may not start with 'chat:'"
    assert responses[-1].status_code == 422 and bad_channel["detail"][0]["loc"] == ["body", "channel"]


# ---- data written before the enums existed ------------------------------------------------------------
def _paused_by_an_older_version(harness: Harness, thread_id: str, ref: str) -> None:
    """A case waiting for staff, written the way the code wrote it before the enums: literal strings in the
    saved state, a staff_flags row and a reviews row. Nothing here goes through today's enum members."""
    connection = harness.tools.store.connection
    connection.execute("INSERT INTO staff_flags (flag_id, patient_ref, reason, urgency, status, created_at, category, contact_id) "
                 "VALUES ('FLAG-0001', ?, 'Triage review needed: Urgent band: live clinician contact within 1 hour "
                 "(triage SOP 4.8)', 'Urgent', 'awaiting_staff', '2026-08-01T09:00', 'administrative', ?)", (ref, thread_id))
    connection.execute("INSERT INTO reviews (review_id, thread_id, flag_id, patient_ref, route, reasons, requested_at, status) "
                 "VALUES ('REV-0001', ?, 'FLAG-0001', ?, 'human_review', "
                 "'Urgent band: live clinician contact within 1 hour (triage SOP 4.8)', '2026-08-01T09:00', 'pending')",
                 (thread_id, ref))
    old_state = {
        "record": {"channel": "web_form", "received_at": "2026-08-01T09:00:00Z", "raw_text": "I need to be seen soon.",
                   "patient_ref": ref},
        "answer": True,
        "guard": {"blocked": False, "category": "none", "layer": "none", "reason": "", "band": None, "score": None,
                  "latency_ms": 0.0},
        "intake": {"patient_name": None, "symptoms": [], "urgency": "Urgent", "specialty": "Dermatology",
                   "insurance_plan": None, "insurance_member_number": None, "seeks_clinical_advice": False,
                   "missing_fields": ["insurance_id"], "channel": None},
        "patient_ref": ref, "band": "Urgent", "repeat": None, "topic": "appointment_booking",
        "eligibility": {"status": "active", "plan_code": "MERIDIAN-GOLD", "new_patient": False, "payment": "insurance",
                        "unavailable": False},
        "plan": {"specialty": "Dermatology", "sign_off_required": False, "referral": None, "preauth": None,
                 "payment": "insurance"},
        "route": "human_review",
        "reasons": ["Urgent band: live clinician contact within 1 hour (triage SOP 4.8)"],
        "response": "Your request has been marked urgent and passed to our clinical staff, who will contact you. "
                    "If you think this is an emergency, call 911 now.",
        "agent": {}, "verdict": None, "flag_id": "FLAG-0001", "review_id": "REV-0001", "decision": None,
        "status": "awaiting_human_review",
        "audit": [{"step": step, "at": "2026-08-01T09:00"} for step in
                  ("guard", "intake", "hydrate", "eligibility", "routing", "team", "decide")],
    }
    graph = build_triage_graph(harness.dependencies, harness.checkpointer)
    config_ = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config_, old_state, as_node="decide")  # saved as the `decide` node left it
    graph.invoke(None, config_)  # ... and stopped at the review pause, where the older version left it


@pytest.mark.parametrize("durable", [False, True])
def test_a_case_paused_with_plain_strings_can_be_listed_fetched_and_completed(durable, tmp_path):
    tools = None
    if durable:  # file-backed store and SQLite checkpoints, closed and reopened: as after an upgrade and restart
        from src.tools import CareFlowTools

        tools = CareFlowTools(store=store.RuntimeStore(tmp_path / "runtime.db"))
    harness = Harness(tools=tools)
    if durable:
        harness.checkpointer = open_session_checkpointer(tmp_path / "sessions.db")
    ref = _find_patient(harness.tools)
    _paused_by_an_older_version(harness, "old-case", ref)
    if durable:
        close_checkpointer(harness.checkpointer)
        harness.tools.store.close()
        harness = Harness(tools=CareFlowTools(store=store.RuntimeStore(tmp_path / "runtime.db")))
        harness.checkpointer = open_session_checkpointer(tmp_path / "sessions.db")
    try:
        with _client(harness) as client:
            queue = client.get("/v1/reviews").json()  # listed
            assert queue["count"] == 1 and queue["reviews"][0]["thread_id"] == "old-case"
            assert (queue["reviews"][0]["status"], queue["reviews"][0]["route"]) == ("pending", "human_review")
            payload = client.get("/v1/reviews/old-case").json()  # fetched
            assert (payload["review_id"], payload["flag_id"], payload["route"], payload["band"]) == (
                "REV-0001", "FLAG-0001", "human_review", "Urgent")
            again = client.post("/v1/triage", json={"raw_text": "hello", "thread_id": "old-case"})
            assert again.status_code == 409  # the old status string still marks the thread as taken
            lowered = client.post("/v1/reviews/old-case/decision",
                                  json={"reviewer_id": "RN-0042", "action": "approve", "urgency": "Routine"})
            assert lowered.status_code == 422 and "recorded reason" in lowered.json()["detail"]
            done = client.post("/v1/reviews/old-case/decision",  # completed
                               json={"reviewer_id": "RN-0042", "action": "reject", "note": "Seen in clinic."})
            body = done.json()
            assert done.status_code == 200 and (body["status"], body["route"], body["band"]) == (
                "completed", "human_review", "Urgent")
            assert body["steps"] == ["guard", "intake", "hydrate", "eligibility", "routing", "team", "decide", "finalize"]
            assert client.get("/v1/reviews").json()["count"] == 0
            assert client.get("/v1/reviews/old-case").status_code == 404
        row, flag = harness.tools.store.get_review("REV-0001"), harness.tools.store.get_flag("FLAG-0001")
        assert (row["status"], row["action"], row["reviewer_id"], row["note"]) == (
            "decided", "reject", "RN-0042", "Seen in clinic.")
        assert flag["status"] == "reviewed_reject"
        assert get_pending_review("old-case", harness.dependencies, harness.checkpointer) is None
        assert_plain(build_triage_graph(harness.dependencies, harness.checkpointer).get_state(build_thread_config("old-case")).values, "state")
    finally:
        if durable:
            close_checkpointer(harness.checkpointer)
            harness.tools.store.close()


def test_a_conversation_stored_with_plain_role_strings_continues(tmp_path):
    harness = Harness(steps=[_make_message("Your plan is active.")])
    checkpointer = open_session_checkpointer(tmp_path / "sessions.db")
    thread = CHAT_PREFIX + "old"
    try:
        graph = build_agent_graph(harness.dependencies.agent, checkpointer=checkpointer)
        call = {"id": "c1", "type": "function", "function": {"name": "check_eligibility", "arguments": "{}"}}
        old = [{"role": "user", "content": "Is my plan active?"},
               {"role": "assistant", "content": "", "tool_calls": [call]},
               {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": False, "error": "give patient_ref"})},
               {"role": "assistant", "content": "I need your details first."}]
        graph.update_state({"configurable": {"thread_id": thread}},
                           {"messages": old, "patient_ref": "MHP-P-00001", "steps": 2, "escalated": False,
                            "channel": "web_form", "draft": None, "grounding": {}}, as_node="record")
        team = run_team("And now?", harness.dependencies.agent, None, thread_id=thread, checkpointer=checkpointer)
        assert (team.answer, team.patient_ref, team.route) == ("Your plan is active.", "MHP-P-00001", "auto")
        messages = graph.get_state(build_thread_config(thread)).values["messages"]
        assert messages[:4] == old and [saved_message["role"] for saved_message in messages[4:]] == ["user", "assistant"]
        # The model was sent the old turns as they were stored.
        assert harness.offered and len(messages) == 6
        # A veto recorded in the old form (role "redact") still replaces the stored answer.
        graph.update_state({"configurable": {"thread_id": thread}},
                           {"messages": [{"role": "redact", "content": "[withheld]", "count": 1}]}, as_node="record")
        assert graph.get_state(build_thread_config(thread)).values["messages"][-1] == {"role": "assistant", "content": "[withheld]"}
    finally:
        close_checkpointer(checkpointer)
