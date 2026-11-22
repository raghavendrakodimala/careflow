"""M6: use the EHR tools through the MCP server instead of in-process.

McpClient starts `python -m careflow.mcp` as a child process and talks to it over stdio. The MCP
SDK is async; the agents and the workflow are synchronous, so the client keeps one event loop on a
background thread and exposes plain blocking calls.

McpRoutedTools is what the agents are given: it looks like CareFlowTools, sends the EHR tools to the
server and everything else (policy search, co-pay calculator, staff flags) to the local object. The
graph and the workflow only ever use `deps.tools`, so they need no change.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from ..tools import CareFlowTools
from .server import EHR_TOOLS

APP_ROOT = Path(__file__).resolve().parents[2]  # the folder that holds the `careflow` package
_STOP = object()


class McpUnavailable(RuntimeError):
    """The MCP server could not be started or has stopped answering."""


class McpClient:
    def __init__(self, db_path: Path | str, timeout_s: float = 30.0):
        self.db_path, self.timeout_s = str(db_path), timeout_s
        self.tools: list[dict] = []  # [{name, description, input_schema}] as the server lists them
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue | None = None
        self._ready: Future = Future()
        self._thread = threading.Thread(target=self._run, name="careflow-mcp", daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=timeout_s)
        except Exception as exc:
            raise McpUnavailable(f"MCP server did not start: {type(exc).__name__}: {exc}") from exc

    # ---- background loop: the client is entered and left in one task, as anyio requires ----------
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        self._queue = asyncio.Queue()
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "careflow.mcp"], cwd=str(APP_ROOT),
            env={**os.environ, "CAREFLOW_RUNTIME_DB": self.db_path, "PYTHONPATH": str(APP_ROOT)},
        )
        try:
            async with Client(params) as client:
                listed = await client.list_tools()
                self.tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                              for t in listed.tools]
                self._ready.set_result(True)
                while True:
                    item = await self._queue.get()
                    if item is _STOP:
                        return
                    future, name, arguments = item
                    try:
                        future.set_result(await client.call_tool(name, arguments))
                    except Exception as exc:
                        future.set_exception(exc)
        except Exception as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)

    # ---- blocking interface ---------------------------------------------------------------------
    @property
    def tool_names(self) -> set[str]:
        return {t["name"] for t in self.tools}

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """One tool call as the tool's own result dict. Raises McpUnavailable if the server is gone."""
        if not self._thread.is_alive() or self._queue is None:
            raise McpUnavailable("MCP server is not running")
        future: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (future, name, arguments or {}))
        try:
            result = future.result(timeout=self.timeout_s)
        except Exception as exc:
            raise McpUnavailable(f"{name}: {type(exc).__name__}: {exc}") from exc
        text = next((c.text for c in result.content if getattr(c, "type", "") == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"ok": False, "error": text or "empty MCP result"}

    def close(self) -> None:
        if self._thread.is_alive() and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _STOP)
            self._thread.join(timeout=10)


class McpRoutedTools:
    """CareFlowTools with the EHR tools served over MCP. Everything else is the local object."""

    def __init__(self, local: CareFlowTools, client: McpClient | None = None):
        if local.store.in_memory:
            raise ValueError("MCP needs a file-backed runtime store: the server is a separate process")
        self.local = local
        self.client = client or McpClient(local.store.path)
        self.remote = self.client.tool_names & set(EHR_TOOLS)
        self.remote_calls: list[str] = []  # names sent to the server, in order (tests and the demo read this)

    def __getattr__(self, name: str) -> Any:  # store, clock, patients, available_tools, ...
        return getattr(self.local, name)

    def call(self, name: str, arguments: dict | str | None = None, context: dict | None = None) -> dict:
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
