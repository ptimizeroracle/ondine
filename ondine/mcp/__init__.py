"""Ondine MCP server (L5 front door).

Exposes four pipeline operations as MCP tools (ondine_estimate, ondine_run,
ondine_status, ondine_collect). Requires the ``ondine[mcp]`` extra (FastMCP).
See :mod:`ondine.mcp.server` for the implementation.
"""

from ondine.mcp.progress import RegistryProgressObserver


def __getattr__(name: str):
    # Lazy: importing the package does not require fastmcp; only building the
    # server does. This keeps `import ondine.mcp` cheap and dep-free.
    if name == "MCPService":
        from ondine.mcp.server import MCPService

        return MCPService
    if name == "create_server":
        from ondine.mcp.server import create_server

        return create_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MCPService", "create_server", "RegistryProgressObserver"]
