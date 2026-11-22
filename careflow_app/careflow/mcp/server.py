"""M6: the mock EHR (scheduling, eligibility, pre-authorisation, patient lookup) as an MCP server over stdio.

    python -m careflow.mcp

No business logic lives here. Every tool is the same CareFlowTools.call() the agents use in-process, and
the name, description and JSON schema of each tool come from the one ToolDefinition list in
tool_contracts.py (ADR-03), so the MCP view and the model's view of a tool cannot drift apart.

The server opens the shared runtime store (out/runtime.db, or CAREFLOW_RUNTIME_DB), which is SQLite in WAL
mode: a booking made through this process is seen at once by the main process, and the other way round.
Stdout carries the protocol, so nothing else may be printed there.
"""

from __future__ import annotations

import json

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from ..store import RuntimeStore
from ..tool_contracts import TOOLS_BY_NAME
from ..tools import CareFlowTools

SERVER_NAME = "careflow-ehr"
# The toolkit's "Scheduling / EHR MCP server (M6)": appointment_slots, appointments, providers, patients.
EHR_TOOLS = ("find_appointment_slots", "book_appointment", "check_eligibility", "check_preauth", "find_patient")


def tool_list() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=TOOLS_BY_NAME[name].description,
            input_schema=TOOLS_BY_NAME[name].openai_spec()["function"]["parameters"],
        )
        for name in EHR_TOOLS
    ]


def run_tool(tools: CareFlowTools, name: str, arguments: dict | None) -> dict:
    """One tool call as the plain result dict. Never raises: an unknown tool, bad arguments or a failure
    inside the tool all come back as {"ok": False, "error": ...}, the same shape the agents already handle."""
    if name not in EHR_TOOLS:
        return {"ok": False, "error": f"unknown tool {name!r}", "valid_tools": list(EHR_TOOLS)}
    try:
        return tools.call(name, arguments or {})
    except Exception as exc:  # a tool bug must not take the server down
        return {"ok": False, "error": f"{name} failed: {type(exc).__name__}"}


def build_server(tools: CareFlowTools) -> Server:
    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tool_list())

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        result = await anyio.to_thread.run_sync(run_tool, tools, params.name, params.arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result))],
            is_error=not result.get("ok", False),
        )

    return Server(SERVER_NAME, version="1.0", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve() -> None:
    tools = CareFlowTools(store=RuntimeStore())
    server = build_server(tools)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        tools.store.close()


def main() -> None:
    anyio.run(serve)

