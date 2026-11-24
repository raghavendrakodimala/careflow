"""The EHR MCP server and its client, beyond test_mcp.py: the server's handlers in-process, the wire
protocol of `python -m src.mcp` as a real child process on a database in tmp_path, and the client's
failure paths with a scripted stand-in for the MCP SDK's Client (no child process, no waiting)."""

import asyncio
import inspect
import json
import queue
import runpy
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import anyio
import pytest

import src.mcp.client as client_module
import src.mcp.server as server_module
from mcp import Client
from src.mcp import (
    EHR_TOOLS,
    SERVER_NAME,
    McpClient,
    McpRoutedTools,
    McpUnavailableError,
    build_server,
    list_tools,
    run_tool,
)
from src.mcp.client import APP_ROOT, _build_child_environment
from src.reliability import ResilientTools, RetryPolicy
from src.store import RuntimeStore
from src.tool_contracts import TOOLS_BY_NAME
from src.tools import CareFlowTools

KNOWN = "MHP-P-00001"  # a patient in the seed data
WAIT_S = 30  # the longest a test waits for the child process; a healthy one answers in a second or two


def _get_schema(name):
    return TOOLS_BY_NAME[name].build_openai_spec()["function"]["parameters"]


# ---- tool definitions --------------------------------------------------------------------------------
def test_list_tools_is_the_five_ehr_tools_with_the_contract_schemas():
    listed = list_tools()
    assert [tool.name for tool in listed] == ["find_appointment_slots", "book_appointment", "check_eligibility",
                                        "check_preauth", "find_patient"] == list(EHR_TOOLS)
    for tool in listed:
        assert tool.description == TOOLS_BY_NAME[tool.name].description
        assert tool.input_schema == _get_schema(tool.name)
        assert tool.input_schema["type"] == "object"


def test_every_ehr_tool_is_a_tool_the_model_is_also_given():
    assert set(EHR_TOOLS) <= set(TOOLS_BY_NAME)
    # Tools that do not touch EHR data are not served, so they can only ever run in the main process.
    assert not {"flag_for_human", "calculate_copay", "search_policies"} & set(EHR_TOOLS)


# ---- run_tool ----------------------------------------------------------------------------------------
def test_run_tool_returns_what_the_direct_call_returns():
    tools = CareFlowTools()
    got = run_tool(tools, "check_eligibility", {"patient_ref": KNOWN})
    assert got["ok"] is True and got["patient_ref"] == KNOWN
    assert got == tools.call("check_eligibility", {"patient_ref": KNOWN})


@pytest.mark.parametrize("arguments", [None, {}, {"nope": 1}, {"patient_ref": 12345}])
def test_run_tool_reports_bad_arguments_as_a_structured_error(arguments):
    got = run_tool(CareFlowTools(), "check_eligibility", arguments)
    assert got["ok"] is False and isinstance(got["error"], str) and got["error"]


def test_run_tool_refuses_a_tool_outside_the_ehr_set_and_names_the_valid_ones():
    tools = CareFlowTools()
    for name in ("flag_for_human", "calculate_copay", "no_such_tool", ""):
        got = run_tool(tools, name, {"reason": "x"})
        assert got == {"ok": False, "error": f"unknown tool {name!r}", "valid_tools": list(EHR_TOOLS)}
    assert tools.flags == []  # nothing ran


def test_run_tool_turns_a_tool_bug_into_an_error_that_names_only_the_exception_type():
    class BrokenTools:
        def call(self, name, arguments):
            raise KeyError("member MER64867187 not in cache")

    got = run_tool(BrokenTools(), "check_eligibility", {"patient_ref": KNOWN})
    assert got == {"ok": False, "error": "check_eligibility failed: KeyError"}  # the message's data stays out


# ---- the server's handlers, in-process ----------------------------------------------------------------
def _run_in_process(tools, run_session):
    """Run `run_session(client)` against build_server(tools) over the SDK's in-memory transport."""
    async def run():
        async with Client(build_server(tools)) as client:
            return await run_session(client)

    return anyio.run(run)


def _parse_payload(result):
    (item,) = result.content  # one JSON text item per result
    assert item.type == "text"
    return json.loads(item.text)


def test_the_server_lists_the_tool_definitions_unchanged():
    async def run_session(client):
        return (await client.list_tools()).tools

    listed = _run_in_process(CareFlowTools(), run_session)
    assert [(tool.name, tool.description, tool.input_schema) for tool in listed] == [
        (tool.name, tool.description, tool.input_schema) for tool in list_tools()]


def test_a_tool_result_travels_as_one_json_text_item():
    tools = CareFlowTools()

    async def run_session(client):
        return await client.call_tool("check_eligibility", {"patient_ref": KNOWN})

    result = _run_in_process(tools, run_session)
    assert result.is_error is False
    assert _parse_payload(result) == tools.call("check_eligibility", {"patient_ref": KNOWN})


def test_a_failed_call_is_marked_as_an_error_and_the_server_keeps_serving():
    tools = CareFlowTools()

    async def run_session(client):
        return [await client.call_tool("check_eligibility", {"nope": 1}),
                await client.call_tool("flag_for_human", {"reason": "x"}),
                await client.call_tool("check_eligibility", {"patient_ref": "MHP-P-99999"}),
                await client.call_tool("check_eligibility", {"patient_ref": KNOWN})]

    bad, unknown, nobody, good = _run_in_process(tools, run_session)
    assert bad.is_error is True and _parse_payload(bad)["ok"] is False and "bad arguments" in _parse_payload(bad)["error"]
    assert unknown.is_error is True and _parse_payload(unknown)["valid_tools"] == list(EHR_TOOLS) and tools.flags == []
    # "Not found" is an answer, not a failure: is_error follows the tool's own "ok".
    assert nobody.is_error is (not _parse_payload(nobody)["ok"])
    assert good.is_error is False and _parse_payload(good)["ok"] is True


def test_a_booking_made_through_the_server_is_written_to_the_store_it_was_given():
    tools = CareFlowTools()
    ref = next(patient_ref for patient_ref, patient in tools.patients.items() if patient["eligibility_status"] == "active")
    slot = tools.call("find_appointment_slots", {"specialty": "ENT", "patient_ref": ref})["slots"][0]["slot_id"]

    async def run_session(client):
        return [await client.call_tool("book_appointment", {"patient_ref": ref, "slot_id": slot}),
                await client.call_tool("book_appointment", {"patient_ref": ref, "slot_id": slot})]

    first, second = _run_in_process(tools, run_session)
    assert _parse_payload(first)["ok"] is True and tools.booked[slot] == ref
    assert second.is_error is True and _parse_payload(second)["ok"] is False  # a slot is booked once


# ---- serve() and the entry points ---------------------------------------------------------------------
def test_serve_opens_the_configured_database_and_closes_it_when_the_stream_ends(tmp_path, monkeypatch):
    opened = []
    real_store = server_module.RuntimeStore

    def record_store(*args, **kwargs):
        opened.append(real_store(*args, **kwargs))
        return opened[-1]

    @asynccontextmanager
    async def open_closed_stdio():  # a client that has already hung up: the read stream is at its end
        to_server, from_client = anyio.create_memory_object_stream(1)
        to_client, from_server = anyio.create_memory_object_stream(1)
        await to_server.aclose()
        async with from_client, to_client, from_server:
            yield from_client, to_client

    monkeypatch.setattr("src.store.RUNTIME_DB", tmp_path / "runtime.db")
    monkeypatch.setattr(server_module, "RuntimeStore", record_store)
    monkeypatch.setattr(server_module, "stdio_server", open_closed_stdio)

    anyio.run(server_module.serve)

    (store,) = opened
    assert store.path == str(tmp_path / "runtime.db") and (tmp_path / "runtime.db").exists()
    with pytest.raises(Exception, match="closed"):  # sqlite3.ProgrammingError: the connection was closed
        store.connection.execute("SELECT 1")


def test_serve_closes_the_store_when_the_transport_fails(tmp_path, monkeypatch):
    opened = []
    real_store = server_module.RuntimeStore

    @asynccontextmanager
    async def open_broken_stdio():
        raise OSError("stdin is not a pipe")
        yield  # pragma: no cover

    monkeypatch.setattr("src.store.RUNTIME_DB", tmp_path / "runtime.db")
    monkeypatch.setattr(server_module, "RuntimeStore", lambda *args, **kwargs: opened.append(real_store(*args, **kwargs)) or opened[-1])
    monkeypatch.setattr(server_module, "stdio_server", open_broken_stdio)

    with pytest.raises(OSError, match="not a pipe"):
        anyio.run(server_module.serve)
    with pytest.raises(Exception, match="closed"):
        opened[0].connection.execute("SELECT 1")


def test_main_runs_serve_to_completion(monkeypatch):
    ran = []

    async def fake_serve():
        ran.append("served")

    monkeypatch.setattr(server_module, "serve", fake_serve)
    server_module.main()
    assert ran == ["served"]


def test_running_the_package_as_a_module_starts_the_server(monkeypatch):
    started = []
    monkeypatch.setattr(server_module, "main", lambda: started.append("main"))
    runpy.run_module("src.mcp", run_name="__main__")
    assert started == ["main"]


# ---- the wire protocol of the real child process ------------------------------------------------------
class _Stdio:
    """`python -m src.mcp` as a child process, driven with newline-delimited JSON-RPC and no MCP SDK on
    this side. Stdout is read on a thread so that a server that stops answering fails the test, not hangs it."""

    def __init__(self, db_path):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "src.mcp"], cwd=str(APP_ROOT), text=True, encoding="utf-8",
            env={**_build_child_environment(), "CAREFLOW_RUNTIME_DB": str(db_path), "PYTHONPATH": str(APP_ROOT)},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.lines: queue.Queue = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.next_id = 0

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)  # end of stream

    def _send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def notify(self, method):
        self._send({"jsonrpc": "2.0", "method": method})

    def request(self, method, params=None):
        """Send one request and return its response; everything on stdout must be a JSON-RPC message."""
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": self.next_id, "method": method, **({"params": params} if params is not None else {})})
        while True:
            line = self.lines.get(timeout=WAIT_S)
            assert line is not None, "the server closed its output before answering"
            message = json.loads(line)
            assert message["jsonrpc"] == "2.0"
            if message.get("id") == self.next_id:
                return message

    def initialize(self):
        answer = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "careflow-tests", "version": "0"}})
        self.notify("notifications/initialized")
        return answer

    def call(self, name, arguments):
        """tools/call as (is_error, the tool's result dict)."""
        result = self.request("tools/call", {"name": name, "arguments": arguments})["result"]
        (item,) = result["content"]
        return bool(result.get("isError")), json.loads(item["text"])

    def hang_up(self):
        """Close the server's input, as a client that leaves does, and return its exit code."""
        if self.process.poll() is None:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=WAIT_S)
        except subprocess.TimeoutExpired:  # only if the server is broken: do not leave it holding the database
            self.process.kill()
            code = self.process.wait()
        self.reader.join(timeout=WAIT_S)
        self.process.stdout.close()
        return code


@pytest.fixture(scope="module")
def wire(tmp_path_factory):
    """(the child process after the handshake, the initialize response, local tools on the same database)."""
    database_path = tmp_path_factory.mktemp("wire") / "runtime.db"
    local = CareFlowTools(store=RuntimeStore(database_path))
    server = _Stdio(database_path)
    try:
        yield server, server.initialize(), local
    finally:
        server.hang_up()
        local.store.close()


def test_initialize_names_the_server_and_offers_tools_only(wire):
    _, hello, _ = wire
    result = hello["result"]
    assert result["serverInfo"]["name"] == SERVER_NAME == "careflow-ehr"
    assert result["protocolVersion"] == "2025-06-18"
    assert set(result["capabilities"]) == {"tools"}  # no prompts, resources or logging


def test_tools_list_on_the_wire_carries_the_contract_schemas(wire):
    server, _, _ = wire
    listed = server.request("tools/list", {})["result"]["tools"]
    assert [tool["name"] for tool in listed] == list(EHR_TOOLS)
    for tool in listed:
        assert tool["description"] == TOOLS_BY_NAME[tool["name"]].description
        assert tool["inputSchema"] == _get_schema(tool["name"])


def test_tools_call_on_the_wire_returns_the_same_result_as_the_direct_function(wire):
    server, _, local = wire
    for name, arguments in (("check_eligibility", {"patient_ref": KNOWN}),
                            ("find_appointment_slots", {"specialty": "Dermatology", "limit": 2}),
                            ("check_preauth", {"procedure_code": "GAST-COLO", "plan_code": "MERIDIAN-SILVER"})):
        is_error, got = server.call(name, arguments)
        assert is_error is False and got == local.call(name, arguments), name


def test_bad_calls_on_the_wire_are_tool_errors_and_the_process_keeps_answering(wire):
    server, _, local = wire
    is_error, got = server.call("check_eligibility", {"nope": 1})
    assert is_error is True and got == local.call("check_eligibility", {"nope": 1}) and got["ok"] is False
    is_error, got = server.call("flag_for_human", {"reason": "x"})
    assert is_error is True and got["valid_tools"] == list(EHR_TOOLS) and local.flags == []
    assert server.call("check_eligibility", {"patient_ref": KNOWN}) == (False, local.call("check_eligibility", {"patient_ref": KNOWN}))


def test_an_unknown_method_gets_a_json_rpc_error_not_silence(wire):
    server, _, _ = wire
    answer = server.request("careflow/delete_everything", {})
    assert "result" not in answer and isinstance(answer["error"]["code"], int) and answer["error"]["message"]
    assert server.request("tools/list", {})["result"]["tools"]  # still serving


def test_a_booking_on_the_wire_is_seen_by_another_connection_to_the_database(wire):
    server, _, local = wire
    ref = next(patient_ref for patient_ref, patient in local.patients.items() if patient["eligibility_status"] == "active")
    slot = local.call("find_appointment_slots", {"specialty": "ENT", "patient_ref": ref})["slots"][0]["slot_id"]
    is_error, booked = server.call("book_appointment", {"patient_ref": ref, "slot_id": slot})
    assert is_error is False and booked["slot_id"] == slot
    other = RuntimeStore(local.store.path)
    try:
        assert other.find_one("appointment_slots", "slot_id", slot)["status"] == "booked"
    finally:
        other.close()
    assert server.call("book_appointment", {"patient_ref": ref, "slot_id": slot})[0] is True  # never twice


def test_the_server_exits_cleanly_when_the_client_hangs_up_and_leaves_the_database_usable(tmp_path):
    database_path = tmp_path / "runtime.db"
    server = _Stdio(database_path)
    try:
        server.initialize()
        assert server.call("check_eligibility", {"patient_ref": KNOWN})[1]["ok"] is True
    finally:
        code = server.hang_up()
    assert code == 0
    store = RuntimeStore(database_path)  # seeded by the server, and not left locked
    try:
        assert store.find_one("patients", "patient_ref", KNOWN)["patient_ref"] == KNOWN
    finally:
        store.close()


# ---- the client: what the child process is given -------------------------------------------------------
def test_the_child_environment_carries_no_keys(monkeypatch):
    secrets = {"GEMINI_API_KEY": "g", "DATAGEN_GEMINI_API_KEY": "dg", "OPENROUTER_API_KEY": "o",
               "DATAGEN_OPENROUTER_API_KEY": "do", "LANGFUSE_PUBLIC_KEY": "lp", "LANGFUSE_SECRET_KEY": "ls",
               "OPENAI_API_KEY": "oa", "ANTHROPIC_API_KEY": "an", "HF_TOKEN": "hf", "CAREFLOW_API_KEY": "service"}
    for name, value in secrets.items():
        monkeypatch.setenv(name, f"secret-{value}")
    monkeypatch.setenv("CAREFLOW_TOOL_RETRIES", "5")
    monkeypatch.setenv("CAREFLOW_SESSIONS_DB", "elsewhere.db")

    env = _build_child_environment()

    assert not {name.upper() for name in env} & set(secrets)
    assert not [value for value in env.values() if value.startswith("secret-")]
    # CareFlow's own settings and the operating system's basics do travel.
    assert env["CAREFLOW_TOOL_RETRIES"] == "5" and env["CAREFLOW_SESSIONS_DB"] == "elsewhere.db"
    assert {name.upper() for name in env} >= {"PATH"}


def test_the_api_key_is_dropped_whatever_its_case(monkeypatch):
    monkeypatch.delenv("CAREFLOW_API_KEY", raising=False)
    monkeypatch.setitem(client_module.os.environ, "careflow_api_key", "secret-service")  # Windows upper-cases it
    assert "secret-service" not in _build_child_environment().values()


class _ScriptedClient:
    """Stands in for the MCP SDK's Client inside McpClient: no child process. `answer(name, arguments)`
    gives each call's result, or an exception to raise, or an awaitable of either."""

    def __init__(self, answer=None, enter_error=None, listed=EHR_TOOLS):
        self.answer = answer or (lambda name, arguments: _make_text_result(json.dumps({"ok": True, "tool": name})))
        self.enter_error, self.listed = enter_error, listed
        self.params, self.calls, self.left = None, [], False

    def __call__(self, params):  # McpClient does Client(params)
        self.params = params
        return self

    async def __aenter__(self):
        error = self.enter_error
        if inspect.iscoroutinefunction(error):
            error = await error()
        if error:
            raise error
        return self

    async def __aexit__(self, *exc):
        self.left = True
        return False

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name=name, description=f"about {name}", input_schema={"type": "object"})
                                      for name in self.listed])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        out = self.answer(name, arguments)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, Exception):
            raise out
        return out


def _make_text_result(*texts):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text) for text in texts])


@pytest.fixture()
def connect(monkeypatch, tmp_path):
    """connect(scripted, **client_options) -> an McpClient talking to `scripted`; closed at teardown."""
    clients = []

    def _connect(scripted, **options):
        monkeypatch.setattr(client_module, "Client", scripted)
        clients.append(McpClient(tmp_path / "runtime.db", **options))
        return clients[-1]

    yield _connect
    for client in clients:
        client.close()
        assert not client._thread.is_alive()


def test_the_server_is_started_as_this_interpreter_on_the_given_database_without_keys(connect, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-g")
    monkeypatch.setenv("CAREFLOW_API_KEY", "secret-service")
    monkeypatch.setenv("CAREFLOW_RUNTIME_DB", "the-parents-own.db")
    scripted = _ScriptedClient()
    connect(scripted)
    params = scripted.params
    assert (params.command, params.args, params.cwd) == (sys.executable, ["-m", "src.mcp"], str(APP_ROOT))
    assert params.env["CAREFLOW_RUNTIME_DB"] == str(tmp_path / "runtime.db")  # the client's database wins
    assert params.env["PYTHONPATH"] == str(APP_ROOT)
    assert not [value for value in params.env.values() if value.startswith("secret-")]


def test_the_client_keeps_the_listing_the_server_gave(connect):
    client = connect(_ScriptedClient(listed=("check_eligibility", "find_patient")))
    assert client.tool_names == {"check_eligibility", "find_patient"}
    assert client.tools[0] == {"name": "check_eligibility", "description": "about check_eligibility",
                               "input_schema": {"type": "object"}}


# ---- the client: failures ------------------------------------------------------------------------------
def test_a_server_that_fails_to_start_raises_mcp_unavailable_with_the_cause(monkeypatch, tmp_path):
    monkeypatch.setattr(client_module, "Client", _ScriptedClient(enter_error=FileNotFoundError("no interpreter")))
    with pytest.raises(McpUnavailableError, match="did not start: FileNotFoundError: no interpreter") as raised:
        McpClient(tmp_path / "runtime.db")
    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_a_server_that_does_not_come_up_in_time_raises_mcp_unavailable(monkeypatch, tmp_path):
    held = {}

    async def hang_until_released():  # the start hangs until the test lets go; then it fails, which ends the thread
        held["loop"], held["thread"], held["release"] = asyncio.get_running_loop(), threading.current_thread(), asyncio.Event()
        held["started"].set()
        await held["release"].wait()
        return ConnectionError("gave up")

    held["started"] = threading.Event()
    monkeypatch.setattr(client_module, "Client", _ScriptedClient(enter_error=hang_until_released))
    try:
        with pytest.raises(McpUnavailableError, match="did not start: TimeoutError"):
            McpClient(tmp_path / "runtime.db", startup_timeout_s=0.05)
    finally:
        assert held["started"].wait(WAIT_S)
        held["loop"].call_soon_threadsafe(held["release"].set)
        held["thread"].join(WAIT_S)
    assert not held["thread"].is_alive()


def test_a_call_the_server_fails_raises_mcp_unavailable_and_the_next_call_still_works(connect):
    answers = iter([ConnectionError("pipe closed"), _make_text_result('{"ok": true, "eligible": true}')])
    client = connect(_ScriptedClient(lambda name, arguments: next(answers)))
    with pytest.raises(McpUnavailableError, match="check_eligibility: ConnectionError: pipe closed"):
        client.call("check_eligibility", {"patient_ref": KNOWN})
    assert client.call("check_eligibility", {"patient_ref": KNOWN}) == {"ok": True, "eligible": True}


def test_a_call_that_gets_no_answer_in_time_raises_and_its_late_answer_is_not_given_to_the_next_call(connect):
    held = {"started": threading.Event()}

    async def answer_late():
        held["release"] = asyncio.Event()
        held["started"].set()
        await held["release"].wait()
        return _make_text_result('{"ok": true, "answer": "late"}')

    answers = iter([answer_late(), _make_text_result('{"ok": true, "answer": "fresh"}')])
    client = connect(_ScriptedClient(lambda name, arguments: next(answers)), timeout_s=0.05)
    with pytest.raises(McpUnavailableError, match=r"find_patient: no answer in .* the call had been sent"):
        client.call("find_patient", {"member_number": "MER00000000"})
    assert held["started"].wait(WAIT_S)
    client._loop.call_soon_threadsafe(held["release"].set)  # the first call now completes, with nobody waiting
    client.timeout_s = WAIT_S
    assert client.call("find_patient", {"member_number": "MER00000000"})["answer"] == "fresh"


def test_a_call_still_queued_when_it_times_out_is_never_sent(connect):
    held = {"started": threading.Event()}

    async def answer_late():
        held["release"] = asyncio.Event()
        held["started"].set()
        await held["release"].wait()
        return _make_text_result('{"ok": true, "answer": "late"}')

    answers = iter([answer_late(), _make_text_result('{"ok": true, "answer": "fresh"}')])
    scripted = _ScriptedClient(lambda name, arguments: next(answers))
    client = connect(scripted, timeout_s=0.05)
    with pytest.raises(McpUnavailableError, match="the call had been sent"):
        client.call("find_patient", {"member_number": "MER00000000"})
    assert held["started"].wait(WAIT_S)
    # The server is still busy with the first call, so this booking waits in the queue and times out there.
    with pytest.raises(McpUnavailableError, match=r"book_appointment: no answer in .* the call was not sent"):
        client.call("book_appointment", {"patient_ref": KNOWN, "slot_id": "MHP-SL-00001"})
    client._loop.call_soon_threadsafe(held["release"].set)
    client.timeout_s = WAIT_S
    assert client.call("find_patient", {"member_number": "MER00000000"})["answer"] == "fresh"
    assert [name for name, _ in scripted.calls] == ["find_patient", "find_patient"]  # the booking never ran


def test_missing_arguments_are_sent_as_an_empty_object(connect):
    scripted = _ScriptedClient()
    client = connect(scripted)
    client.call("check_eligibility")
    client.call("check_eligibility", None)
    assert scripted.calls == [("check_eligibility", {}), ("check_eligibility", {})]


@pytest.mark.parametrize("result, expected", [
    (_make_text_result("Internal Server Error"), {"ok": False, "error": "Internal Server Error"}),
    (_make_text_result(""), {"ok": False, "error": "empty MCP result"}),
    (_make_text_result(), {"ok": False, "error": "empty MCP result"}),
    (SimpleNamespace(content=[SimpleNamespace(type="image", data="...")]), {"ok": False, "error": "empty MCP result"}),
])
def test_a_result_that_is_not_json_becomes_a_structured_error(connect, result, expected):
    client = connect(_ScriptedClient(lambda name, arguments: result))
    assert client.call("check_eligibility", {"patient_ref": KNOWN}) == expected


def test_close_leaves_the_session_and_may_be_called_twice(monkeypatch, tmp_path):
    scripted = _ScriptedClient()
    monkeypatch.setattr(client_module, "Client", scripted)
    client = McpClient(tmp_path / "runtime.db")
    client.close()
    assert scripted.left is True and not client._thread.is_alive()
    client.close()  # nothing left to stop
    with pytest.raises(McpUnavailableError, match="not running"):
        client.call("check_eligibility", {"patient_ref": KNOWN})
    assert scripted.calls == []


# ---- routing: which calls go to the server -------------------------------------------------------------
@pytest.fixture()
def file_tools(tmp_path):
    tools = CareFlowTools(store=RuntimeStore(tmp_path / "runtime.db"))
    yield tools
    tools.store.close()


class _RemoteClient:
    """A stand-in for McpClient: records calls and answers (or raises) as told."""

    def __init__(self, names=EHR_TOOLS, error=None):
        self.tool_names, self.error, self.calls, self.closed = set(names), error, [], 0

    def call(self, name, arguments=None):
        self.calls.append((name, arguments))
        if self.error:
            raise self.error
        return {"ok": True, "served_by": "mcp"}

    def close(self):
        self.closed += 1


def test_only_ehr_tools_the_server_lists_are_sent_to_it(file_tools):
    # The server offers one EHR tool and one tool that must never leave the main process.
    remote = _RemoteClient(names={"check_eligibility", "flag_for_human"})
    routed = McpRoutedTools(file_tools, client=remote)
    assert routed.remote == {"check_eligibility"}

    assert routed.call("check_eligibility", {"patient_ref": KNOWN}) == {"ok": True, "served_by": "mcp"}
    assert routed.call("find_patient", {"given_name": "No", "family_name": "Body", "member_number": "MER00000000"}) == \
        file_tools.call("find_patient", {"given_name": "No", "family_name": "Body", "member_number": "MER00000000"})
    flag = routed.call("flag_for_human", {"reason": "Needs a callback"}, context={"contact_id": "t1"})
    assert flag["ok"] and file_tools.flags[0]["contact_id"] == "t1"
    assert remote.calls == [("check_eligibility", {"patient_ref": KNOWN})] and routed.remote_calls == ["check_eligibility"]


def test_json_string_arguments_are_parsed_before_they_are_sent(file_tools):
    remote = _RemoteClient()
    routed = McpRoutedTools(file_tools, client=remote)
    routed.call("check_eligibility", json.dumps({"patient_ref": KNOWN}))
    routed.call("check_eligibility", "")
    routed.call("check_eligibility", None)
    assert remote.calls == [("check_eligibility", {"patient_ref": KNOWN}), ("check_eligibility", {}), ("check_eligibility", {})]


def test_closing_the_routed_tools_closes_the_client_but_not_the_store(file_tools):
    remote = _RemoteClient()
    McpRoutedTools(file_tools, client=remote).close()
    assert remote.closed == 1 and file_tools.store.find_one("patients", "patient_ref", KNOWN)


def test_a_dead_server_reaches_the_agent_as_an_unavailable_result_not_an_exception(file_tools):
    remote = _RemoteClient(error=McpUnavailableError("MCP server is not running"))
    tools = ResilientTools(McpRoutedTools(file_tools, client=remote), retry=RetryPolicy(attempts=2, sleep=lambda seconds: None))

    lookup = tools.call("check_eligibility", {"patient_ref": KNOWN})
    assert lookup["ok"] is False and lookup["unavailable"] is True and "McpUnavailableError" in lookup["error"]
    assert len(remote.calls) == 2  # a read is retried

    booking = tools.call("book_appointment", {"patient_ref": KNOWN, "slot_id": "SLOT-000001"})
    assert booking["unavailable"] is True and len(remote.calls) == 3  # a booking is sent once, never retried
    assert file_tools.booked == {}  # and is not quietly made in-process instead
    # Tools that never leave the main process still work.
    assert tools.call("flag_for_human", {"reason": "EHR is down"}, context={"contact_id": "t2"})["ok"] is True
