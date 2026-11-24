"""The HTTP service beyond test_api.py: the production wiring (build_default_resources) on databases in tmp_path,
/health in each state, the request and response contract, what is logged when a turn fails, and the
service end to end with the EHR tools really served by the MCP child process. The model, the intake
parser and the reviewer are still fakes (the Harness of test_workflow.py), so nothing calls a provider."""

import contextlib
import logging
import sqlite3
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src import intake
from src.agent import open_session_checkpointer
from src.agents import SafetyVerdict
from src.api.app import UNAVAILABLE, Resources, build_default_resources, create_app
from src.config import Settings
from src.guard_model import LocalClinicalClassifier
from src.guardrails import ClinicalRisk
from src.llm_client import LLMClientError
from src.mcp import EHR_TOOLS, McpRoutedTools, McpUnavailableError
from src.prompts import DEFLECTION_SCRIPT
from src.reliability import OPEN, CircuitBreaker, ResilientTools
from src.schemas import PatientRequest
from src.store import RuntimeStore
from src.telemetry import MemorySink, Tracer
from src.tools import CareFlowTools

from .test_guard_model import _make_encoder, _write_weights
from .test_workflow import Harness, _find_patient, _make_call, _make_message

API_LOG = "src.api.app"
HEALTH_KEYS = {"status", "patients", "pending_reviews", "policy_search", "tracing", "guard_classifier", "mcp", "breakers"}
TRIAGE_KEYS = {"thread_id", "status", "route", "reasons", "response", "band", "patient_ref", "eligibility", "plan",
               "agent", "verdict", "guard", "flag_id", "review_id", "decision", "steps"}
CHAT_KEYS = {"thread_id", "answer", "route", "agents", "tools_used", "vetoed", "guard", "verdict", "patient_ref"}
REVIEW_KEYS = {"review_id", "flag_id", "route", "reasons", "band", "patient_ref", "plan", "response_sent"}
GUARD_KEYS = {"blocked", "category", "layer", "reason", "band", "score", "latency_ms"}


def _build_settings(**overrides) -> Settings:
    """Settings that read no .env file: tracing off, rules-only input gate, no API key, EHR in-process."""
    values = {"careflow_tracing": "off", "careflow_guard_classifier": "none", "careflow_api_key": "",
              "careflow_api_use_mcp": False, **overrides}
    return Settings(_env_file=None, **values)


def _raise_no_index(settings=None, path=None):
    raise FileNotFoundError("the policy index has not been built")


def _isolate(patch, folder, policy_search=_raise_no_index) -> None:
    """Point the production wiring at databases in `folder`, and at `policy_search` for the policy index.
    The paths are module attributes read when a store or checkpointer is opened (see src/config.py)."""
    patch.setattr("src.store.RUNTIME_DB", folder / "runtime.db")
    patch.setattr("src.agent.SESSIONS_DB", folder / "sessions.db")
    patch.setattr("src.rag.retriever.open_policy_search", policy_search)


def _close(resources: Resources) -> None:
    with contextlib.suppress(Exception):  # teardown: already closed by the test is fine
        resources.close()


@pytest.fixture()
def production(monkeypatch, tmp_path):
    """production(**settings) -> build_default_resources() on databases in tmp_path; closed at teardown.
    They are `owned`, so an app built over them closes them when its TestClient block ends."""
    opened = []

    def _open(policy_search=_raise_no_index, **overrides) -> Resources:
        _isolate(monkeypatch, tmp_path, policy_search)
        opened.append(build_default_resources(_build_settings(**overrides)))
        return opened[-1]

    yield _open
    for resources in opened:
        _close(resources)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The production wiring with the EHR tools served by the MCP child process, started once."""
    with pytest.MonkeyPatch.context() as patch:
        _isolate(patch, tmp_path_factory.mktemp("served"))
        resources = build_default_resources(_build_settings(careflow_api_use_mcp=True))
    resources.owned = False  # shared by several apps in this module: closed here, not by the first app to shut down
    yield resources
    _close(resources)


def _script(resources: Resources, steps=None, parsed=None) -> Harness:
    """Replace the model, the intake parser and the reviewer of `resources` with the Harness's fakes."""
    harness = Harness(tools=resources.dependencies.tools, steps=steps, parsed=parsed)
    resources.dependencies.agent.model, resources.dependencies.parse = harness._call_model, harness._parse
    resources.dependencies.review = lambda message, reply: SafetyVerdict(safe=True)
    return harness


def _open_client(harness: Harness, classifier=None, **settings) -> TestClient:
    """An app over the Harness's in-memory store, like test_api._open_client but with settings that read no file."""
    harness.dependencies.review = lambda message, reply: SafetyVerdict(safe=True)
    return TestClient(create_app(Resources(harness.dependencies, harness.checkpointer, classifier=classifier, settings=_build_settings(**settings))))


def _is_closed(connection) -> bool:
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def _collect_log_records(caplog, level=logging.WARNING):
    return [record for record in caplog.records if record.name == API_LOG and record.levelno == level]


# ---- production wiring: build_default_resources ---------------------------------------------------------------
def test_build_default_resources_opens_its_own_databases_and_runs_the_ehr_in_process_when_mcp_is_off(production, tmp_path):
    resources = production()
    tools = resources.dependencies.tools
    assert resources.owned is True and resources.classifier is None and resources.settings.careflow_api_use_mcp is False
    assert tools.store.path == str(tmp_path / "runtime.db") and (tmp_path / "sessions.db").exists()
    assert isinstance(tools, ResilientTools) and isinstance(tools.inner, CareFlowTools)
    with TestClient(create_app(resources)) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "patients": 400, "pending_reviews": 0, "policy_search": False, "tracing": False,
                    "guard_classifier": "none", "mcp": "off", "breakers": {"ehr": "closed", "policy_index": "closed"}}


def test_build_default_resources_serves_the_ehr_tools_over_mcp_when_asked(served):
    routed = served.dependencies.tools.inner
    assert isinstance(routed, McpRoutedTools) and routed.remote == set(EHR_TOOLS)
    assert routed.client.db_path == served.dependencies.tools.store.path  # the child works on the service's own database
    with TestClient(create_app(served)) as client:
        assert client.get("/health").json()["mcp"] == "running"


def test_a_missing_policy_index_switches_policy_search_off_with_a_warning(production, caplog):
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        resources = production(policy_search=_raise_no_index)
    (record,) = _collect_log_records(caplog)
    assert "policy search is off" in record.getMessage() and record.exc_info[0] is FileNotFoundError
    assert "search_policies" not in resources.dependencies.tools.list_available_tools()


def test_an_empty_policy_index_is_treated_as_no_index_and_a_built_one_is_offered(production, monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        empty = production(policy_search=lambda settings: SimpleNamespace(count=lambda: 0))
        assert "search_policies" not in empty.dependencies.tools.list_available_tools()
        empty.close()
        built = production(policy_search=lambda settings: SimpleNamespace(count=lambda: 212))
    assert _collect_log_records(caplog) == []  # neither is a fault
    with TestClient(create_app(built)) as client:
        assert client.get("/health").json()["policy_search"] is True


def test_an_mcp_server_that_does_not_start_leaves_the_tools_in_process_with_a_warning(production, monkeypatch, caplog):
    def refuse_to_start(local):
        raise McpUnavailableError("MCP server did not start: TimeoutError")

    monkeypatch.setattr("src.mcp.McpRoutedTools", refuse_to_start)
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        resources = production(careflow_api_use_mcp=True)
    assert [record.getMessage() for record in _collect_log_records(caplog) if "MCP" in record.getMessage()] == [
        "the EHR MCP server did not start: its tools run in-process"]
    assert isinstance(resources.dependencies.tools.inner, CareFlowTools)
    with TestClient(create_app(resources)) as client:
        assert client.get("/health").json()["mcp"] == "off"
        assert resources.dependencies.tools.call("check_eligibility", {"patient_ref": "MHP-P-00001"})["ok"] is True


def test_a_local_classifier_is_loaded_at_start_up(production, monkeypatch, tmp_path):
    classifier = LocalClinicalClassifier(_write_weights(tmp_path), _make_encoder({}))
    monkeypatch.setattr(api, "build_classifier", lambda settings: classifier)
    resources = production(careflow_guard_classifier="local")
    assert resources.classifier is classifier and classifier._weights is not None  # warmed before the first message
    with TestClient(create_app(resources)) as client:
        assert client.get("/health").json()["guard_classifier"] == "local:ready"


def test_a_local_classifier_that_cannot_load_is_logged_and_every_message_is_then_refused(production, monkeypatch, tmp_path, caplog):
    classifier = LocalClinicalClassifier(tmp_path / "no-such-weights.npz", _make_encoder({}))
    monkeypatch.setattr(api, "build_classifier", lambda settings: classifier)
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        resources = production(careflow_guard_classifier="local")  # the service still starts
    (record,) = _collect_log_records(caplog, logging.ERROR)
    assert "did not load" in record.getMessage()
    harness = _script(resources)
    with TestClient(create_app(resources)) as client:
        assert client.get("/health").json()["guard_classifier"] == "local:not loaded"
        body = client.post("/v1/chat", json={"message": "Is my plan active?"}).json()
    assert body["route"] == "refuse" and body["guard"]["layer"] == "fail_closed" and harness.offered == []


# ---- production wiring: closing -------------------------------------------------------------------------
def test_close_closes_the_store_and_the_checkpointer(production):
    resources = production()
    store, checkpointer = resources.dependencies.tools.store, resources.checkpointer
    resources.close()
    assert _is_closed(store.connection) and _is_closed(checkpointer.conn)


class _UnstoppableTools:
    """Tools whose backend fails to close (an MCP child that will not stop)."""

    def __init__(self, local):
        self.local, self.close_calls = local, 0

    def __getattr__(self, name):
        return getattr(self.local, name)

    def close(self):
        self.close_calls += 1
        raise McpUnavailableError("the server did not stop")


class _ClosingSink(MemorySink):
    closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize("wrap", [lambda tools: tools, lambda tools: ResilientTools(tools)],
                         ids=["tools with a backend", "the resilience wrapper around them"])
def test_close_still_closes_everything_else_when_a_backend_will_not_stop(tmp_path, caplog, wrap):
    stubborn = _UnstoppableTools(CareFlowTools(store=RuntimeStore(tmp_path / "runtime.db")))
    harness = Harness(tools=wrap(stubborn))
    sink = _ClosingSink()
    harness.dependencies.agent.tracer = Tracer([sink])
    checkpointer = open_session_checkpointer(tmp_path / "sessions.db")
    resources = Resources(harness.dependencies, checkpointer, settings=_build_settings(), owned=True)
    try:
        with caplog.at_level(logging.WARNING, logger=API_LOG):
            resources.close()  # does not raise
    finally:
        checkpointer.conn.close()
        stubborn.local.store.close()
    assert stubborn.close_calls >= 1
    assert {record.getMessage() for record in _collect_log_records(caplog)} == {"a tools backend did not close cleanly"}
    assert sink.closed and _is_closed(checkpointer.conn) and _is_closed(stubborn.local.store.connection)


def test_an_app_built_without_resources_opens_the_production_ones_and_closes_them_at_shutdown(production, monkeypatch):
    resources = production()
    opened = []
    monkeypatch.setattr(api, "build_default_resources", lambda: opened.append(resources) or resources)
    app = create_app()
    assert opened == []  # nothing is opened until the service starts
    with TestClient(app) as client:
        assert opened == [resources] and client.get("/health").json()["status"] == "ok"
        assert not _is_closed(resources.dependencies.tools.store.connection)
    assert _is_closed(resources.dependencies.tools.store.connection) and _is_closed(resources.checkpointer.conn)


# ---- /health ------------------------------------------------------------------------------------------
def _get_health(harness: Harness, classifier=None) -> dict:
    with _open_client(harness, classifier=classifier) as client:
        response = client.get("/health")
    assert response.status_code == 200 and set(response.json()) == HEALTH_KEYS
    return response.json()


def test_health_reports_no_classifier_when_the_gate_is_rules_only():
    assert _get_health(Harness(), classifier=None)["guard_classifier"] == "none"


def test_health_reports_a_hosted_classifier():
    assert _get_health(Harness(), classifier=lambda text: ClinicalRisk(score=0.0))["guard_classifier"] == "hosted"


def test_health_tells_a_loaded_local_classifier_from_one_that_is_not(tmp_path):
    classifier = LocalClinicalClassifier(_write_weights(tmp_path), _make_encoder({}))
    assert _get_health(Harness(), classifier)["guard_classifier"] == "local:not loaded"
    assert classifier._weights is None  # asking for health does not load the model
    classifier.warm()
    assert _get_health(Harness(), classifier)["guard_classifier"] == "local:ready"


@pytest.fixture()
def file_tools(tmp_path):
    tools = CareFlowTools(store=RuntimeStore(tmp_path / "runtime.db"))
    yield tools
    tools.store.close()


def _make_mcp_client(thread):
    return SimpleNamespace(tool_names=set(EHR_TOOLS), _thread=thread, close=lambda: None)


def test_health_reports_mcp_off_when_the_tools_run_in_process():
    body = _get_health(Harness())
    assert body["mcp"] == "off" and body["breakers"] == {}
    assert _get_health(Harness(tools=ResilientTools(CareFlowTools())))["mcp"] == "off"


@pytest.mark.parametrize("wrap", [lambda tools: tools, lambda tools: ResilientTools(tools)],
                         ids=["routed tools", "routed tools behind the resilience wrapper"])
def test_health_reports_whether_the_mcp_client_is_still_running(file_tools, wrap):
    assert _get_health(Harness(tools=wrap(McpRoutedTools(file_tools, client=_make_mcp_client(threading.current_thread())))))["mcp"] == "running"
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()
    assert _get_health(Harness(tools=wrap(McpRoutedTools(file_tools, client=_make_mcp_client(finished)))))["mcp"] == "stopped"


def test_health_lists_each_breaker_with_its_state():
    ehr = CircuitBreaker("ehr", failure_threshold=1, cooldown_s=30.0, now=lambda: 0.0)
    tools = ResilientTools(CareFlowTools(), breakers={"ehr": ehr, "policy_index": CircuitBreaker("policy_index")})
    harness = Harness(tools=tools)
    assert _get_health(harness)["breakers"] == {"ehr": "closed", "policy_index": "closed"}
    ehr.record_failure()
    assert _get_health(harness)["breakers"] == {"ehr": OPEN, "policy_index": "closed"}  # an open breaker is not "not ready"


def test_health_reports_tracing_and_policy_search_when_they_are_on():
    harness = Harness(tools=CareFlowTools(policy_index=SimpleNamespace()))
    harness.dependencies.agent.tracer = Tracer([MemorySink()])
    body = _get_health(harness)
    assert body["tracing"] is True and body["policy_search"] is True


def test_health_counts_the_cases_waiting_for_review():
    harness = Harness(parsed={"urgency": "Urgent"})
    with _open_client(harness) as client:
        assert client.get("/health").json()["pending_reviews"] == 0
        client.post("/v1/triage", json={"raw_text": "I need to be seen soon.", "patient_ref": _find_patient(harness.tools)})
        assert client.get("/health").json()["pending_reviews"] == 1


def test_health_is_503_when_the_checkpointer_cannot_be_read():
    def raise_locked(config):
        raise sqlite3.OperationalError("database is locked")

    harness = Harness()
    harness.checkpointer = SimpleNamespace(get_tuple=raise_locked)
    with _open_client(harness) as client:
        response = client.get("/health")
    assert response.status_code == 503 and response.json() == {"detail": "not ready: OperationalError"}


def test_health_is_503_when_the_runtime_store_cannot_be_read(tmp_path):
    tools = CareFlowTools(store=RuntimeStore(tmp_path / "runtime.db"))
    harness = Harness(tools=tools)
    harness.dependencies.review = lambda message, reply: SafetyVerdict(safe=True)
    app = create_app(Resources(harness.dependencies, harness.checkpointer, settings=_build_settings()))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 200
        tools.store.close()  # the database goes away under the running service
        response = client.get("/health")
    assert response.status_code == 503 and response.json()["detail"].startswith("not ready")


# ---- end to end: HTTP -> agent -> MCP child process -> SQLite ---------------------------------------------
def _find_open_slot(resources: Resources, ref: str, specialty: str) -> str:
    """The first open slot, read through the service's own in-process tools (not through MCP)."""
    local = resources.dependencies.tools.inner.local
    return local.call("find_appointment_slots", {"specialty": specialty, "patient_ref": ref})["slots"][0]["slot_id"]


def _read_slot_row(resources: Resources, slot: str) -> dict:
    """The slot as a brand new connection to the runtime database sees it."""
    other = RuntimeStore(resources.dependencies.tools.store.path)
    try:
        return other.find_one("appointment_slots", "slot_id", slot)
    finally:
        other.close()


def test_a_booking_made_through_chat_is_made_by_the_mcp_server_and_is_in_the_database(served):
    ref = _find_patient(served.dependencies.tools)
    slot = _find_open_slot(served, ref, "Dermatology")
    _script(served, steps=[_make_message(calls=[_make_call("c1", "find_appointment_slots", specialty="Dermatology", patient_ref=ref)]),
                           _make_message(calls=[_make_call("c2", "book_appointment", patient_ref=ref, slot_id=slot)]),
                           _make_message("You are booked.")])
    routed = served.dependencies.tools.inner
    before = len(routed.remote_calls)
    with TestClient(create_app(served)) as client:
        response = client.post("/v1/chat", json={"message": "Please book me a dermatology appointment.", "patient_ref": ref})
        body = response.json()
        assert response.status_code == 200 and set(body) == CHAT_KEYS
        assert body["answer"] == "You are booked." and body["route"] == "auto" and body["patient_ref"] == ref
        assert body["tools_used"] == ["find_appointment_slots", "book_appointment"]
        assert client.get("/health").json()["breakers"]["ehr"] == "closed"
    assert routed.remote_calls[before:] == ["find_appointment_slots", "book_appointment"]  # both went to the child
    assert _read_slot_row(served, slot)["status"] == "booked" and routed.local.booked[slot] == ref


def test_a_booking_made_by_the_triage_workflow_is_made_by_the_mcp_server_and_is_in_the_database(served):
    ref = _find_patient(served.dependencies.tools)
    slot = _find_open_slot(served, ref, "Dermatology")
    _script(served, steps=[_make_message(calls=[_make_call("c1", "book_appointment", patient_ref=ref, slot_id=slot)]),
                           _make_message("You are booked with Dermatology.")])
    routed = served.dependencies.tools.inner
    before = len(routed.remote_calls)
    with TestClient(create_app(served)) as client:
        response = client.post("/v1/triage", json={"raw_text": "I'd like a dermatology appointment.", "patient_ref": ref,
                                                   "channel": "sms", "thread_id": "e2e-triage-1"})
    body = response.json()
    assert response.status_code == 200 and set(body) == TRIAGE_KEYS
    assert (body["status"], body["route"], body["response"]) == ("completed", "auto", "You are booked with Dermatology.")
    assert routed.remote_calls[before:].count("book_appointment") == 1
    assert _read_slot_row(served, slot)["status"] == "booked" and routed.local.booked[slot] == ref


def test_a_case_that_must_wait_for_staff_books_nothing_even_if_the_model_asks_to(served):
    ref = _find_patient(served.dependencies.tools)
    slot = _find_open_slot(served, ref, "Cardiology")  # needs primary-care sign-off, and this patient has no referral
    harness = _script(served, parsed={"specialty": "Cardiology"},
                steps=[_make_message(calls=[_make_call("c1", "book_appointment", patient_ref=ref, slot_id=slot)]), _make_message("You are booked.")])
    routed = served.dependencies.tools.inner
    before = len(routed.remote_calls)
    with TestClient(create_app(served)) as client:
        body = client.post("/v1/triage", json={"raw_text": "I'd like a cardiology appointment.", "patient_ref": ref}).json()
    assert (body["status"], body["route"]) == ("awaiting_human_review", "human_review")
    assert all("book_appointment" not in offered for offered in harness.offered)
    assert "book_appointment" not in routed.remote_calls[before:]  # the call never reached the EHR
    assert _read_slot_row(served, slot)["status"] != "booked" and slot not in routed.local.booked


def test_triage_uses_the_production_intake_parser_and_fails_closed_when_it_cannot_parse(served, monkeypatch):
    ref = _find_patient(served.dependencies.tools)
    harness = _script(served)
    served.dependencies.parse = lambda record: intake.parse_intake(record, served.settings)  # the production parser again
    seen = []

    def return_structured(messages, response_model, settings=None, **kwargs):
        seen.append(messages)
        return PatientRequest(urgency="Urgent", seeks_clinical_advice=False, specialty="Dermatology")

    monkeypatch.setattr(intake, "complete_structured", return_structured)
    with TestClient(create_app(served)) as client:
        parsed = client.post("/v1/triage", json={"raw_text": "My leg is swollen and painful.", "patient_ref": ref}).json()
        assert (parsed["band"], parsed["route"], parsed["status"]) == ("Urgent", "human_review", "awaiting_human_review")
        assert "My leg is swollen and painful." in seen[0][1]["content"] and ref not in str(seen[0])

        def raise_no_model(messages, response_model, settings=None, **kwargs):
            raise LLMClientError("every model failed")

        monkeypatch.setattr(intake, "complete_structured", raise_no_model)
        harness.offered.clear()
        failed = client.post("/v1/triage", json={"raw_text": "Can I get an appointment?", "patient_ref": ref}).json()
    assert (failed["route"], failed["status"], failed["response"]) == ("refuse", "awaiting_human_review", DEFLECTION_SCRIPT)
    assert harness.offered == []  # the assistant never saw a message that could not be read


def test_a_case_paused_for_review_survives_a_restart_and_is_completed_through_the_review_routes(production, monkeypatch):
    first = production(careflow_api_use_mcp=True)
    ref = _find_patient(first.dependencies.tools)
    _script(first, parsed={"urgency": "Urgent"})
    monkeypatch.setattr(api, "build_default_resources", lambda: first)
    with TestClient(create_app()) as client:  # the app owns these resources and closes them at shutdown
        started = client.post("/v1/triage", json={"raw_text": "I need to be seen soon.", "patient_ref": ref,
                                                  "thread_id": "case-9"}).json()
        assert (started["route"], started["status"]) == ("human_review", "awaiting_human_review")
    assert _is_closed(first.dependencies.tools.store.connection) and _is_closed(first.checkpointer.conn)
    assert not first.dependencies.tools.inner.client._thread.is_alive()  # the MCP child was stopped with the service

    second = production(careflow_api_use_mcp=True)  # a new process would do exactly this, on the same files
    assert second.dependencies.tools.store.path == first.dependencies.tools.store.path
    harness = _script(second)
    with TestClient(create_app(second)) as client:
        assert client.get("/health").json()["pending_reviews"] == 1
        queue = client.get("/v1/reviews").json()
        assert queue["count"] == 1 and queue["reviews"][0]["thread_id"] == "case-9"
        payload = client.get("/v1/reviews/case-9").json()
        assert set(payload) == REVIEW_KEYS and payload["review_id"] == started["review_id"]
        assert (payload["band"], payload["patient_ref"], payload["response_sent"]) == ("Urgent", ref, started["response"])

        done = client.post("/v1/reviews/case-9/decision", json={"reviewer_id": "RN-0042", "action": "approve", "note": "Called."})
        body = done.json()
        assert done.status_code == 200 and (body["status"], body["thread_id"]) == ("completed", "case-9")
        assert body["steps"][:len(started["steps"])] == started["steps"] and body["steps"][-1] == "finalize"
        assert body["decision"] == {"reviewer_id": "RN-0042", "action": "approve", "note": "Called.", "urgency": None}
        assert client.get("/v1/reviews").json()["count"] == 0
        # The thread still holds its case after the restart: it cannot be triaged again.
        again = client.post("/v1/triage", json={"raw_text": "Hello again.", "patient_ref": ref, "thread_id": "case-9"})
        assert again.status_code == 409
        (flag,) = [candidate for candidate in second.dependencies.tools.flags if candidate["flag_id"] == started["flag_id"]]
    assert flag["status"] == "reviewed_approve" and harness.offered == []  # the decision needed no model call


def test_a_chat_conversation_is_remembered_across_a_restart(production):
    first = production()
    ref = _find_patient(first.dependencies.tools)
    _script(first, steps=[_make_message("Your plan is active.")])
    with TestClient(create_app(first)) as client:
        thread = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": ref}).json()["thread_id"]
    assert _is_closed(first.checkpointer.conn)  # shut down

    second = production()
    _script(second, steps=[_make_message("Yes, still active.")])
    other = next(patient_ref for patient_ref in second.dependencies.tools.patients if patient_ref != ref)
    with TestClient(create_app(second)) as client:
        again = client.post("/v1/chat", json={"message": "And now?", "thread_id": thread})
        assert again.status_code == 200 and again.json()["patient_ref"] == ref  # still identified, from the checkpoint
        assert client.post("/v1/chat", json={"message": "And me?", "thread_id": thread, "patient_ref": other}).status_code == 409


# ---- contract: the OpenAPI document ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def openapi():
    return create_app(Resources(Harness().dependencies, None, settings=_build_settings())).openapi()


def _list_operations(spec):
    return {(method.upper(), path) for path, item in spec["paths"].items() for method in item}


def test_the_openapi_document_lists_exactly_the_six_routes(openapi):
    assert _list_operations(openapi) == {
        ("GET", "/health"), ("POST", "/v1/triage"), ("GET", "/v1/reviews"), ("GET", "/v1/reviews/{thread_id}"),
        ("POST", "/v1/reviews/{thread_id}/decision"), ("POST", "/v1/chat")}
    assert (openapi["info"]["title"], openapi["info"]["version"]) == ("CareFlow", "1.0")


@pytest.mark.parametrize("path, model, required", [
    ("/v1/triage", "TriageRequest", ["raw_text"]),
    ("/v1/chat", "ChatRequest", ["message"]),
    ("/v1/reviews/{thread_id}/decision", "ReviewDecision", ["reviewer_id", "action"]),
])
def test_each_request_body_names_its_required_fields_and_forbids_others(openapi, path, model, required):
    body = openapi["paths"][path]["post"]["requestBody"]
    assert body["required"] is True
    assert body["content"]["application/json"]["schema"] == {"$ref": f"#/components/schemas/{model}"}
    schema = openapi["components"]["schemas"][model]
    assert schema["required"] == required and schema["additionalProperties"] is False


def test_the_documented_request_limits_are_the_ones_the_service_enforces(openapi):
    schemas = openapi["components"]["schemas"]
    for model, text in (("TriageRequest", "raw_text"), ("ChatRequest", "message")):
        properties = schemas[model]["properties"]
        assert (properties[text]["minLength"], properties[text]["maxLength"]) == (1, api.MAX_MESSAGE_CHARS)
        assert properties["channel"]["enum"] == ["phone_transcript", "web_form", "email", "sms"]
        (thread,) = [option for option in properties["thread_id"]["anyOf"] if option.get("type") == "string"]
        assert (thread["maxLength"], thread["pattern"]) == (api.MAX_THREAD_ID_CHARS, api.THREAD_ID_PATTERN)
    decision = schemas["ReviewDecision"]["properties"]
    assert decision["action"]["enum"] == ["approve", "reject"]


def test_every_v1_route_documents_the_api_key_header_and_health_does_not(openapi):
    for path, item in openapi["paths"].items():
        for operation in item.values():
            headers = [parameter["name"] for parameter in operation.get("parameters", []) if parameter["in"] == "header"]
            assert headers == ([] if path == "/health" else ["x-api-key"]), path


def test_the_published_document_carries_no_internal_notes_on_the_routes(openapi):
    # The handlers are commented, not documented, so nothing internal is published (see create_app).
    for item in openapi["paths"].values():
        for operation in item.values():
            assert "description" not in operation


# ---- contract: requests -----------------------------------------------------------------------------------
@pytest.mark.parametrize("path, body", [
    ("/v1/triage", {"raw_text": "I need an appointment."}),
    ("/v1/chat", {"message": "Is my plan active?"}),
    ("/v1/reviews/case-1/decision", {"reviewer_id": "RN-1", "action": "approve"}),
])
def test_an_unknown_field_is_rejected_and_named_and_nothing_runs(path, body):
    harness = Harness()
    with _open_client(harness) as client:
        response = client.post(path, json={**body, "ground_truth": {"urgency": "Routine"}})
    (error,) = response.json()["detail"]
    assert response.status_code == 422 and error["type"] == "extra_forbidden" and error["loc"] == ["body", "ground_truth"]
    assert harness.offered == [] and harness.tools.flags == [] and harness.tools.store.list_pending_reviews() == []


@pytest.mark.parametrize("path, body, field", [
    ("/v1/triage", {}, "raw_text"),
    ("/v1/chat", {}, "message"),
    ("/v1/chat", {"message": "   "}, "message"),  # whitespace is stripped first, so this is empty
    ("/v1/chat", {"message": "x" * 8001}, "message"),
    ("/v1/chat", {"message": "hi", "channel": "fax"}, "channel"),
    ("/v1/chat", {"message": "hi", "thread_id": "has spaces"}, "thread_id"),
    ("/v1/chat", {"message": "hi", "thread_id": "x" * 65}, "thread_id"),
    ("/v1/chat", {"message": "hi", "patient_ref": "MHP-P-" + "0" * 40}, "patient_ref"),
    ("/v1/triage", {"raw_text": "hi", "received_at": "2026-08-01T09:00:00" + "0" * 30}, "received_at"),
    ("/v1/reviews/case-1/decision", {"action": "approve"}, "reviewer_id"),
    ("/v1/reviews/case-1/decision", {"reviewer_id": "RN-1"}, "action"),
    ("/v1/reviews/case-1/decision", {"reviewer_id": "RN-1", "action": "escalate"}, "action"),
    ("/v1/reviews/case-1/decision", {"reviewer_id": "RN-1", "action": "approve", "urgency": "Critical"}, "urgency"),
])
def test_an_invalid_field_is_a_422_that_points_at_the_field(path, body, field):
    harness = Harness()
    with _open_client(harness) as client:
        response = client.post(path, json=body)
    assert response.status_code == 422
    assert ["body", field] in [error["loc"] for error in response.json()["detail"]]
    assert harness.offered == []


def test_a_message_of_exactly_the_maximum_length_is_accepted():
    with _open_client(Harness()) as client:
        assert client.post("/v1/chat", json={"message": "x" * api.MAX_MESSAGE_CHARS}).status_code == 200
        assert client.post("/v1/chat", json={"message": "hi", "thread_id": "A" * api.MAX_THREAD_ID_CHARS}).status_code == 200


@pytest.mark.parametrize("thread_id", ["chat:abc", "CHAT:abc", "Chat:abc"])
def test_triage_and_review_routes_refuse_a_thread_in_the_chat_namespace(thread_id):
    harness = Harness()
    with _open_client(harness) as client:
        responses = [client.post("/v1/triage", json={"raw_text": "I need an appointment.", "thread_id": thread_id}),
                     client.get(f"/v1/reviews/{thread_id}"),
                     client.post(f"/v1/reviews/{thread_id}/decision", json={"reviewer_id": "RN-1", "action": "approve"})]
    assert [response.status_code for response in responses] == [422, 422, 422]
    assert {response.json()["detail"] for response in responses} == {"thread_id may not start with 'chat:'"}
    assert harness.offered == [] and harness.tools.flags == []


def test_a_chat_thread_and_a_triage_case_with_the_same_id_do_not_collide():
    harness = Harness()
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        case = client.post("/v1/triage", json={"raw_text": "I'd like a dermatology appointment.", "patient_ref": ref,
                                               "thread_id": "shared-1"})
        chat = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": ref, "thread_id": "shared-1"})
        assert (case.status_code, chat.status_code) == (200, 200) and chat.json()["thread_id"] == "shared-1"
        assert client.post("/v1/chat", json={"message": "And now?", "thread_id": "shared-1"}).status_code == 200  # a chat can go on
        assert client.post("/v1/triage", json={"raw_text": "Again.", "thread_id": "shared-1"}).status_code == 409  # a case cannot


# ---- contract: responses ----------------------------------------------------------------------------------
def test_a_triage_response_has_the_documented_keys_whether_completed_or_paused():
    harness = Harness()
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        done = client.post("/v1/triage", json={"raw_text": "I'd like a dermatology appointment.", "patient_ref": ref}).json()
        harness.parsed["urgency"] = "Urgent"
        paused = client.post("/v1/triage", json={"raw_text": "I need to be seen soon.", "patient_ref": ref}).json()
    assert set(done) == TRIAGE_KEYS == set(paused) and set(done["guard"]) == GUARD_KEYS
    assert (done["status"], done["review_id"], done["decision"]) == ("completed", None, None)
    assert paused["status"] == "awaiting_human_review" and paused["review_id"] and paused["flag_id"]
    assert done["thread_id"] and done["thread_id"] != paused["thread_id"]  # the server names a case that has no id
    assert all(isinstance(step, str) for step in done["steps"])  # step names only, never the audit trail


def test_a_chat_response_has_the_documented_keys():
    harness = Harness(steps=[_make_message(calls=[_make_call("c1", "check_eligibility", patient_ref="MHP-P-00001")]), _make_message("Your plan is active.")])
    with _open_client(harness) as client:
        body = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": "MHP-P-00001"}).json()
    assert set(body) == CHAT_KEYS and set(body["guard"]) == GUARD_KEYS
    assert set(body["verdict"]) == {"safe", "category", "reason"} and body["vetoed"] is False


def test_the_review_queue_and_a_review_have_the_documented_keys():
    harness = Harness(parsed={"urgency": "Urgent"})
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        started = client.post("/v1/triage", json={"raw_text": "I need to be seen soon.", "patient_ref": ref,
                                                  "thread_id": "case-7"}).json()
        queue = client.get("/v1/reviews").json()
        review = client.get("/v1/reviews/case-7").json()
        missing = client.get("/v1/reviews/case-8")
    assert set(queue) == {"count", "reviews"} and queue["count"] == len(queue["reviews"]) == 1
    assert set(queue["reviews"][0]) >= {"review_id", "thread_id", "flag_id", "patient_ref", "route", "reasons", "status"}
    assert queue["reviews"][0]["status"] == "pending"
    assert set(review) == REVIEW_KEYS and review["review_id"] == started["review_id"] == queue["reviews"][0]["review_id"]
    assert missing.status_code == 404 and set(missing.json()) == {"detail"}


def test_the_thread_id_chat_returns_leads_back_to_the_conversation():
    harness = Harness()
    ref = _find_patient(harness.tools)
    with _open_client(harness) as client:
        first = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": ref}).json()
        assert first["thread_id"] and not first["thread_id"].startswith(api.CHAT_PREFIX)  # the prefix stays internal
        second = client.post("/v1/chat", json={"message": "And my referrals?", "thread_id": first["thread_id"]}).json()
        other = client.post("/v1/chat", json={"message": "And my referrals?"}).json()
    assert second["thread_id"] == first["thread_id"] and second["patient_ref"] == ref  # no need to identify again
    assert other["thread_id"] != first["thread_id"] and other["patient_ref"] is None


# ---- contract: 401, 409, 503 ------------------------------------------------------------------------------
V1_REQUESTS = [
    ("POST", "/v1/triage", {"raw_text": "I need an appointment."}),
    ("GET", "/v1/reviews", None),
    ("GET", "/v1/reviews/case-1", None),
    ("POST", "/v1/reviews/case-1/decision", {"reviewer_id": "RN-1", "action": "approve"}),
    ("POST", "/v1/chat", {"message": "Is my plan active?"}),
]


@pytest.mark.parametrize("method, path, body", V1_REQUESTS, ids=[f"{method} {path}" for method, path, _ in V1_REQUESTS])
def test_every_v1_route_needs_the_api_key_when_one_is_configured(method, path, body):
    harness = Harness()
    with _open_client(harness, careflow_api_key="s3cret") as client:
        for headers in ({}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}, {"X-API-Key": "s3cre"}, {"X-API-Key": "s3cret "[:-1] + "x"}):
            refused = client.request(method, path, json=body, headers=headers)
            assert refused.status_code == 401 and refused.json() == {"detail": "missing or wrong X-API-Key"}, headers
        assert harness.offered == [] and harness.tools.flags == []  # nothing ran for a caller without the key
        assert client.request(method, path, json=body, headers={"X-API-Key": "s3cret"}).status_code != 401
        assert client.request(method, path, json=body, headers={"x-api-key": "s3cret"}).status_code != 401


def test_health_and_the_openapi_document_stay_open_when_a_key_is_configured():
    with _open_client(Harness(), careflow_api_key="s3cret") as client:
        assert client.get("/health").status_code == 200 and client.get("/openapi.json").status_code == 200


def test_no_key_is_asked_for_when_none_is_configured():
    with _open_client(Harness(), careflow_api_key="") as client:
        assert client.get("/v1/reviews").status_code == 200
        assert client.get("/v1/reviews", headers={"X-API-Key": "anything"}).status_code == 200


def test_a_conversation_verified_for_one_patient_refuses_another_with_409():
    harness = Harness()
    ref, other = [patient_ref for patient_ref, patient in harness.tools.patients.items() if patient["eligibility_status"] == "active"][:2]
    with _open_client(harness) as client:
        assert client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": ref, "thread_id": "t-1"}).status_code == 200
        asked = len(harness.offered)
        refused = client.post("/v1/chat", json={"message": "Is my plan active?", "patient_ref": other, "thread_id": "t-1"})
        assert refused.status_code == 409 and "another patient" in refused.json()["detail"]
        assert other not in refused.text and ref not in refused.text and len(harness.offered) == asked  # no model call
        kept = client.post("/v1/chat", json={"message": "And my referrals?", "patient_ref": ref.lower(), "thread_id": "t-1"})
    assert kept.status_code == 200 and kept.json()["patient_ref"] == ref  # the conversation is still its owner's


def test_a_decision_the_policy_does_not_allow_is_a_422_and_the_case_keeps_waiting():
    harness = Harness(parsed={"urgency": "Urgent"})
    with _open_client(harness) as client:
        client.post("/v1/triage", json={"raw_text": "I need to be seen soon.", "patient_ref": _find_patient(harness.tools), "thread_id": "case-5"})
        lowered = client.post("/v1/reviews/case-5/decision", json={"reviewer_id": "RN-1", "action": "approve", "urgency": "Routine"})
        assert lowered.status_code == 422 and isinstance(lowered.json()["detail"], str)  # the policy's own wording
        assert client.get("/v1/reviews/case-5").status_code == 200


def test_a_model_failure_in_chat_is_a_503_with_the_fixed_wording_and_the_thread_can_be_used_again():
    harness = Harness()
    working = harness.dependencies.agent.model
    harness.dependencies.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
    with _open_client(harness) as client:
        failed = client.post("/v1/chat", json={"message": "Is my plan active?", "thread_id": "t-503"})
        assert failed.status_code == 503 and failed.json() == {"detail": UNAVAILABLE}
        harness.dependencies.agent.model = working
        retried = client.post("/v1/chat", json={"message": "Is my plan active?", "thread_id": "t-503"})
    assert retried.status_code == 200 and retried.json()["answer"] == "Here is what I found."


# ---- logging privacy ----------------------------------------------------------------------------------------
SECRET_TEXT = "My member number is MER64867187 and I was born on 4 March 1961."


def test_a_failed_chat_turn_is_logged_with_its_thread_and_without_the_message_or_the_patient(caplog):
    harness = Harness()
    ref = _find_patient(harness.tools)
    harness.dependencies.agent.model = lambda messages, specs: (_ for _ in ()).throw(RuntimeError("provider down"))
    with caplog.at_level(logging.DEBUG), _open_client(harness) as client:
        response = client.post("/v1/chat", json={"message": SECRET_TEXT, "patient_ref": ref, "thread_id": "t-private"})
    assert response.status_code == 503
    (record,) = [entry for entry in caplog.records if entry.name == API_LOG]
    assert record.levelno == logging.ERROR and record.getMessage() == "chat failed on thread t-private: RuntimeError"
    # Only the error's type is kept: its text and traceback can quote the request (see the next test).
    assert not record.exc_info and "provider down" not in caplog.text
    for private in (SECRET_TEXT, "MER64867187", "1961", ref):
        assert private not in caplog.text, private  # in no record of any logger, tracebacks included
    assert SECRET_TEXT not in response.text and ref not in response.text and "provider down" not in response.text


def test_a_provider_error_that_quotes_the_request_does_not_put_the_message_in_the_log(caplog):
    def echo_request(messages, specs):
        raise RuntimeError(f"400 Bad Request: could not process {messages[-1]['content']!r}")

    harness = Harness()
    ref = _find_patient(harness.tools)
    harness.dependencies.agent.model = echo_request
    with caplog.at_level(logging.DEBUG), _open_client(harness) as client:
        response = client.post("/v1/chat", json={"message": SECRET_TEXT, "patient_ref": ref, "thread_id": "t-echo"})
    assert response.status_code == 503 and [record for record in caplog.records if record.name == API_LOG]
    assert SECRET_TEXT not in caplog.text and ref not in caplog.text


# ---- the intake parser ----------------------------------------------------------------------------------------
RECORD = {"channel": "sms", "received_at": "2026-08-01T09:00:00Z", "raw_text": "Need a skin check, Meridian Silver.",
          "patient_ref": "MHP-P-00001", "intake_id": "INT-0042", "ground_truth": {"urgency": "Routine"}}


def test_only_the_channel_the_time_and_the_message_reach_the_intake_model():
    system, user = intake.build_messages(RECORD)
    assert (system["role"], system["content"]) == ("system", intake.SYSTEM_PROMPT) and user["role"] == "user"
    assert user["content"] == "Channel: sms\nReceived: 2026-08-01T09:00:00Z\n\nMessage:\nNeed a skin check, Meridian Silver."
    for hidden in ("MHP-P-00001", "INT-0042", "ground_truth"):
        assert hidden not in user["content"]


def test_parse_intake_asks_for_a_patient_request_and_takes_the_channel_from_the_record(monkeypatch):
    asked = []
    parsed = PatientRequest(urgency="Routine", seeks_clinical_advice=False, specialty="Dermatology",
                            insurance_plan="MERIDIAN-SILVER", channel="email")  # the model guessed a channel

    def return_structured(messages, response_model, settings=None):
        asked.append((messages, response_model, settings))
        return parsed

    monkeypatch.setattr(intake, "complete_structured", return_structured)
    settings = _build_settings()
    result = intake.parse_intake(RECORD, settings)

    assert asked == [(intake.build_messages(RECORD), PatientRequest, settings)]  # one model call, with the caller's settings
    assert result.channel == "sms" and parsed.channel == "email"  # copied from the record; the parse is not edited
    assert result.model_dump(exclude={"channel"}) == parsed.model_dump(exclude={"channel"})


def test_parse_intake_passes_no_settings_on_when_it_is_given_none(monkeypatch):
    seen = []
    monkeypatch.setattr(intake, "complete_structured", lambda messages, response_model, settings=None: seen.append(settings)
                        or PatientRequest(urgency="Urgent", seeks_clinical_advice=True))
    result = intake.parse_intake({"channel": "phone_transcript", "received_at": "2026-08-01T09:00:00Z", "raw_text": "uh, hello"})
    assert seen == [None] and (result.urgency, result.channel, result.seeks_clinical_advice) == ("Urgent", "phone_transcript", True)


def test_parse_intake_lets_the_model_failure_reach_the_caller(monkeypatch):
    def raise_no_model(messages, response_model, settings=None):
        raise LLMClientError("every model failed")

    monkeypatch.setattr(intake, "complete_structured", raise_no_model)
    with pytest.raises(LLMClientError, match="every model failed"):
        intake.parse_intake(RECORD)


def test_a_record_without_a_message_is_a_programming_error_not_a_model_call(monkeypatch):
    monkeypatch.setattr(intake, "complete_structured", lambda *args, **kwargs: pytest.fail("the model was called"))
    with pytest.raises(KeyError):
        intake.parse_intake({"channel": "sms", "received_at": "2026-08-01T09:00:00Z"})
