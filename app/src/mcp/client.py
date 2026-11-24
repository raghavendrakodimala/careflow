"""Use the EHR tools through the MCP server instead of in-process.

McpClient starts `python -m src.mcp` as a child process and talks to it over stdio. The MCP
SDK is async; the agents and the workflow are synchronous, so the client keeps one event loop on a
background thread and exposes plain blocking calls.

McpRoutedTools is what the agents are given: it looks like CareFlowTools, sends the EHR tools to the
server and everything else (policy search, co-pay calculator, staff flags) to the local object. The
graph and the workflow only ever use `dependencies.tools`, so they work with either object.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from ..tools import CareFlowTools
from .server import EHR_TOOLS

APP_ROOT = Path(__file__).resolve().parents[2]  # the folder that holds the `src` package
_STOP = object()  # queued by close() to end the background loop
_JOIN_TIMEOUT_S = 10  # how long close() waits for the background thread (and the child process) to finish
# Added to the error of a call that timed out after it was sent.
IN_FLIGHT_NOTE = "the call had been sent and may still take effect"

# What the EHR server needs from the parent's environment: the operating system's basics and CareFlow's
# own settings. Provider, LangFuse and API keys are not passed on; the server calls no model.
_ENV_KEEP = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR", "HOME",
             "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV",
             "PYTHONIOENCODING", "PYTHONUTF8"}
_ENV_DROP = {"CAREFLOW_API_KEY"}


def _build_child_environment() -> dict[str, str]:
    """The environment the server process is started with (names compared case-insensitively, as on Windows)."""
    return {key: value for key, value in os.environ.items()
            if (key.upper() in _ENV_KEEP or key.upper().startswith("CAREFLOW_")) and key.upper() not in _ENV_DROP}


class McpUnavailableError(RuntimeError):
    """The MCP server could not be started or has stopped answering."""


class McpClient:
    """A blocking MCP client for the EHR server it starts as a child process on `db_path`.

    Calls are handed to the background loop through a queue as (future, name, arguments) and the caller waits
    on the future. Raises McpUnavailableError if the server does not start within `startup_timeout_s`."""

    def __init__(self, db_path: Path | str, timeout_s: float = 10.0, startup_timeout_s: float = 30.0):
        # A call that gets no answer is retried by ResilientTools, so the per-call limit is short; only the
        # cold start of the child process gets the long one.
        self.db_path, self.timeout_s = str(db_path), timeout_s
        self.tools: list[dict] = []  # [{name, description, input_schema}] as the server lists them
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue | None = None  # created on the loop's own thread, in _serve
        self._ready: Future = Future()  # resolved once the server has listed its tools, or failed to start
        self._thread = threading.Thread(target=self._run, name="careflow-mcp", daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=startup_timeout_s)
        except Exception as error:
            raise McpUnavailableError(f"MCP server did not start: {type(error).__name__}: {error}") from error

    # ---- background loop: the client is entered and left in one task, as anyio requires ----------
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        """Start the server, list its tools, then run queued calls one at a time until _STOP."""
        self._queue = asyncio.Queue()
        server_parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "src.mcp"], cwd=str(APP_ROOT),
            env={**_build_child_environment(), "CAREFLOW_RUNTIME_DB": self.db_path, "PYTHONPATH": str(APP_ROOT)},
        )
        try:
            async with Client(server_parameters) as client:
                listed = await client.list_tools()
                self.tools = [{"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
                              for tool in listed.tools]
                self._ready.set_result(True)
                while True:
                    item = await self._queue.get()
                    if item is _STOP:
                        return
                    future, name, arguments = item
                    if not future.set_running_or_notify_cancel():
                        continue  # the caller stopped waiting before this call was sent: do not run it
                    try:
                        future.set_result(await client.call_tool(name, arguments))
                    except Exception as error:
                        future.set_exception(error)
        except Exception as error:
            if not self._ready.done():
                self._ready.set_exception(error)

    # ---- blocking interface ---------------------------------------------------------------------
    @property
    def tool_names(self) -> set[str]:
        return {tool["name"] for tool in self.tools}

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """One tool call as the tool's own result dict. Raises McpUnavailableError if the server is gone."""
        if not self._thread.is_alive() or self._queue is None:
            raise McpUnavailableError("MCP server is not running")
        future: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (future, name, arguments or {}))
        try:
            result = future.result(timeout=self.timeout_s)
        except FutureTimeout as error:
            if future.cancel():  # still queued: it will never be sent
                raise McpUnavailableError(f"{name}: no answer in {self.timeout_s:g}s; the call was not sent") from error
            # Already with the server, which cannot be told to stop: the call may still take effect.
            raise McpUnavailableError(f"{name}: no answer in {self.timeout_s:g}s; {IN_FLIGHT_NOTE}") from error
        except Exception as error:
            raise McpUnavailableError(f"{name}: {type(error).__name__}: {error}") from error
        text = next((item.text for item in result.content if getattr(item, "type", "") == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"ok": False, "error": text or "empty MCP result"}

    def close(self) -> None:
        """Stop the background loop, which ends the server process. Safe to call twice."""
        if self._thread.is_alive() and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _STOP)
            self._thread.join(timeout=_JOIN_TIMEOUT_S)


class McpRoutedTools:
    """CareFlowTools with the EHR tools served over MCP. Everything else is the local object."""

    def __init__(self, local: CareFlowTools, client: McpClient | None = None):
        if local.store.in_memory:
            raise ValueError("MCP needs a file-backed runtime store: the server is a separate process")
        self.local = local
        self.client = client or McpClient(local.store.path)
        self.remote = self.client.tool_names & set(EHR_TOOLS)  # sent to the server; anything else runs locally
        self.remote_calls: list[str] = []  # names sent to the server, in order (tests and the demo read this)

    def __getattr__(self, name: str) -> Any:  # store, clock, patients, list_available_tools, ...
        return getattr(self.local, name)

    def call(self, name: str, arguments: dict | str | None = None, context: dict | None = None) -> dict:
        """CareFlowTools.call, with EHR tools sent to the server. `context` (the contact_id) is only used by
        local tools, so it is not sent."""
        if name not in self.remote:
            return self.local.call(name, arguments, context)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return self.local.call(name, arguments, context)  # the usual "bad arguments" error
        self.remote_calls.append(name)
        return self.client.call(name, arguments if isinstance(arguments, dict) else {})

    def close(self) -> None:
        self.client.close()
