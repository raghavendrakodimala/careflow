"""The mock EHR as an MCP server (server.py) and the client the agents use to reach it (client.py)."""

from .client import McpClient, McpRoutedTools, McpUnavailableError
from .server import EHR_TOOLS, SERVER_NAME, build_server, list_tools, run_tool

__all__ = ["EHR_TOOLS", "SERVER_NAME", "McpClient", "McpRoutedTools", "McpUnavailableError", "build_server", "list_tools",
           "run_tool"]
