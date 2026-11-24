"""The mock EHR (scheduling, eligibility, pre-authorisation, patient lookup) as an MCP server over stdio.

    python -m src.mcp

No business logic lives here. Every tool is the same CareFlowTools.call() the agents use in-process, and
the name, description and JSON schema of each tool come from the one ToolDefinition list in
tool_contracts.py, so the MCP view and the model's view of a tool cannot drift apart.

The server opens the shared runtime store (out/runtime.db, or CAREFLOW_RUNTIME_DB), which is SQLite in WAL
mode: a booking made through this process is seen at once by the main process, and the other way round.
Stdout carries the protocol, so nothing else may be printed there.
"""

from __future__ import annotations

import json

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from mcp import types

from ..store import RuntimeStore
from ..tool_contracts import TOOLS_BY_NAME, ToolName
from ..tools import CareFlowTools

SERVER_NAME = "careflow-ehr"
# The tools that read or write EHR data (appointment slots, appointments, providers, patients): the only ones
# this server offers. Policy search, the co-pay calculator and staff flags stay in the main process.
EHR_TOOLS = tuple(tool.value for tool in (
    ToolName.FIND_APPOINTMENT_SLOTS, ToolName.BOOK_APPOINTMENT, ToolName.CHECK_ELIGIBILITY, ToolName.CHECK_PREAUTH,
    ToolName.FIND_PATIENT))


def list_tools() -> list[types.Tool]:
    """The EHR tools as MCP tool definitions, with the same description and schema the model is given."""
    return [
        types.Tool(
            name=name,
            description=TOOLS_BY_NAME[name].description,
            input_schema=TOOLS_BY_NAME[name].build_openai_spec()["function"]["parameters"],
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
    except Exception as error:  # a tool bug must not take the server down
        return {"ok": False, "error": f"{name} failed: {type(error).__name__}"}


def build_server(tools: CareFlowTools) -> Server:
    """The MCP server over `tools`. A tool's result dict travels as one JSON text item, which the client
    (client.McpClient.call) parses back."""

    async def handle_list_tools(context, parameters) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list_tools())

    async def handle_call_tool(context, parameters: types.CallToolRequestParams) -> types.CallToolResult:
        # The tools are synchronous (SQLite): run them on a worker thread so the event loop keeps serving.
        result = await anyio.to_thread.run_sync(run_tool, tools, parameters.name, parameters.arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result))],
            is_error=not result.get("ok", False),
        )

    return Server(SERVER_NAME, version="1.0", on_list_tools=handle_list_tools, on_call_tool=handle_call_tool)


async def serve() -> None:
    """Serve on stdio until the client closes the stream, then close the store."""
    tools = CareFlowTools(store=RuntimeStore())
    server = build_server(tools)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        tools.store.close()


def main() -> None:
    anyio.run(serve)

