"""M6: the mock EHR as an MCP server (server.py) and the client the agents use to reach it (client.py)."""

from .client import McpClient, McpRoutedTools, McpUnavailable
from .server import EHR_TOOLS, SERVER_NAME, build_server, run_tool, tool_list

__all__ = ["EHR_TOOLS", "SERVER_NAME", "McpClient", "McpRoutedTools", "McpUnavailable", "build_server", "run_tool",
           "tool_list"]
