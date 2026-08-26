"""Built-in MCP management tools.

These let the coding agent manage MCP servers at runtime through a standardized
interface instead of the user hand-editing ``.mcp.json`` and restarting:

- ``get_mcp``    — list configured servers and their live status / tools.
- ``add_mcp``    — register + hot-start a stdio MCP server.
- ``delete_mcp`` — remove + hot-stop a server.

They're static (built-in) tools, always bound, and survive rebinds. They reach
the running ``McpManager`` via the ``plugins.mcp.manager`` singleton, which
``main.py`` sets at startup.
"""

from __future__ import annotations

from langchain_core.tools import tool


def _manager():
    import clyde.plugins.mcp as mcpmod

    m = mcpmod.manager
    if m is None:
        raise RuntimeError("MCP manager is not initialized.")
    return m


@tool
def get_mcp() -> str:
    """List configured MCP servers with connection status and exposed tools.

    Returns one line per server: name, command, env var names (values redacted),
    connected/tool-count/tool-names, or the connect error if it failed.
    """
    servers = _manager().list_servers()
    if not servers:
        return "No MCP servers configured. Use add_mcp to add one."
    lines = []
    for s in servers:
        env = ", ".join(s.get("env_keys", [])) or "(none)"
        if s.get("connected"):
            tools = s.get("tools", [])
            lines.append(
                f"- {s['name']}: {s['command']} {s.get('args', [])} | env: {env} | "
                f"connected, {s.get('tool_count', 0)} tool(s): {', '.join(tools)}"
            )
        else:
            err = s.get("error", "not connected")
            lines.append(
                f"- {s['name']}: {s['command']} {s.get('args', [])} | env: {env} | {err}"
            )
    return "\n".join(lines)


@tool
def add_mcp(
    name: str,
    command: str,
    args: list[str],
    env: dict = {},
    timeout: int = 200,
) -> str:
    """Add and hot-start a stdio MCP server.

    Args:
        name: A unique server name (kebab-case, e.g. 'linkedin').
        command: Executable to launch the server (e.g. 'uvx', 'npx', 'python').
        args: Arguments passed to the command (e.g. ['mcp-server-linkedin@latest']).
        env: Environment variables for the server process (e.g. {'UV_HTTP_TIMEOUT':'300'}).
        timeout: Per-tool-call wait in seconds (raise for slow/browser servers; default 200).

    Writes the server to .mcp.json, starts it, and binds its tools into the agent
    immediately. Returns the tool list or the connect error.
    """
    return _manager().add_server(name, command, args, env, timeout)


@tool
def delete_mcp(name: str) -> str:
    """Remove and hot-stop an MCP server by name.

    Closes its session, drops its tools from the agent, and removes it from .mcp.json.
    """
    return _manager().delete_server(name)
