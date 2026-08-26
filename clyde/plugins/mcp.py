"""MCP server manager for Clyde.

Owns one ``MultiServerMCPClient`` (constructed, NOT as a context manager —
that was removed in langchain-mcp-adapters 0.3.2) plus a persistent background
asyncio loop. Each configured server gets a **long-lived session** via
``client.session(name)`` + ``load_mcp_tools(session)`` held open on the
background loop; MCP tools are async, so sync calls from the graph dispatch to
that loop via ``run_coroutine_threadsafe``.

Servers can be added/removed at runtime through the management tools
(``tools/mcp_manage.py``): ``add_server`` writes ``.mcp.json``, opens a session,
loads tools, and rebinds the executor + tool registry; ``delete_server`` closes
the session and rebinds. Rebind lands between LLM invokes within a turn, so new
tools are usable the same turn.

A missing/empty ``.mcp.json`` yields a manager with no servers (and ``add_server``
can still hot-add later).
"""

from __future__ import annotations

import asyncio
import json
import queue as _queue
import threading
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Per-tool-call wait for an MCP tool to return. Browser-backed servers (LinkedIn)
# need a lot more than the old 60s default.
DEFAULT_TIMEOUT = 200
# How long add_server blocks for the initial connect + tool listing.
_CONNECT_TIMEOUT = 60


class McpManager:
    """Owns MCP server sessions for the lifetime of the app."""

    def __init__(self, connections: dict, timeouts: dict, path: str):
        # name -> {command, args, env, transport} (no timeout; that's our wrapper's)
        self._client = MultiServerMCPClient(dict(connections))
        self._timeouts: dict[str, int] = dict(timeouts)  # name -> per-tool timeout
        self._path = path

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # name -> threading.Event (signals _hold to close the session)
        self._stops: dict[str, threading.Event] = {}
        # name -> threading.Event (signals add_server that connect finished)
        self._ready: dict[str, threading.Event] = {}
        # name -> threading.Event (signals delete_server that _hold finished)
        self._done: dict[str, threading.Event] = {}

        self._servers: dict[str, dict] = {}  # name -> {"tools": [StructuredTool]}
        self._errors: dict[str, str] = {}    # name -> error string

        self._rebind_lock = threading.Lock()

    @classmethod
    def from_config(cls, path: str = ".mcp.json") -> "McpManager":
        """Build a manager from a ``.mcp.json`` file; empty if missing/unreadable."""
        p = Path(path)
        if not p.is_file():
            return cls({}, {}, path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls({}, {}, path)
        servers = data.get("mcpServers") or {}

        connections: dict[str, dict] = {}
        timeouts: dict[str, int] = {}
        for name, spec in servers.items():
            spec = dict(spec)
            timeout = int(spec.pop("timeout", DEFAULT_TIMEOUT))
            spec.setdefault("transport", "stdio")
            connections[name] = spec
            timeouts[name] = timeout
        return cls(connections, timeouts, path)

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Start the background loop + open a session per configured server."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        for name in list(self._client.connections):
            self._schedule_hold(name)

    def _schedule_hold(self, name: str) -> None:
        self._stops[name] = threading.Event()
        self._ready[name] = threading.Event()
        self._done[name] = threading.Event()
        asyncio.run_coroutine_threadsafe(self._hold(name), self._loop)

    async def _hold(self, name: str) -> None:
        """Open a long-lived session, load + wrap tools, hold until told to close."""
        stop = self._stops[name]
        loop = asyncio.get_event_loop()
        try:
            async with self._client.session(name) as session:
                tools = await load_mcp_tools(
                    session, server_name=name, tool_name_prefix=True
                )
                timeout = self._timeouts.get(name, DEFAULT_TIMEOUT)
                wrapped = [self._wrap(t, timeout) for t in tools]
                self._servers[name] = {"tools": wrapped}
                self._rebind()
                self._ready[name].set()
                # Hold the session open until delete/shutdown sets the stop event.
                await loop.run_in_executor(None, stop.wait)
        except Exception as e:
            self._errors[name] = f"{type(e).__name__}: {e}"
            self._ready[name].set()
        finally:
            self._servers.pop(name, None)
            self._stops.pop(name, None)
            self._errors.pop(name, None)
            self._rebind()
            done = self._done.pop(name, None)
            if done is not None:
                done.set()

    def shutdown(self) -> None:
        """Close all sessions and stop the background loop."""
        if self._loop is None:
            return
        for stop in list(self._stops.values()):
            stop.set()
        # Let _hold coroutines unwind (close subprocesses).
        for done in list(self._done.values()):
            done.wait(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    # --- runtime management (called by the management tools) -------------

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        if self._loop is None:
            return "MCP manager not started."
        if not name or not command:
            return "add_mcp: 'name' and 'command' are required."
        if name in self._client.connections:
            return f"add_mcp: a server named '{name}' already exists. Use delete_mcp first."

        env = env or {}
        conn = {"command": command, "args": list(args), "env": dict(env), "transport": "stdio"}
        self._client.connections[name] = conn
        self._timeouts[name] = timeout
        self._write_config()
        self._schedule_hold(name)

        ready = self._ready[name]
        ready.wait(timeout=_CONNECT_TIMEOUT)

        if name in self._servers:
            tool_names = [t.name for t in self._servers[name]["tools"]]
            return f"added '{name}' — {len(tool_names)} tool(s): {', '.join(tool_names)}"
        if name in self._errors:
            return (
                f"added '{name}' to .mcp.json, but connect failed: {self._errors[name]}. "
                "Config saved; fix the server and restart, or delete_mcp to remove."
            )
        return (
            f"added '{name}' to .mcp.json; still connecting after {_CONNECT_TIMEOUT}s. "
            "It will bind when ready."
        )

    def delete_server(self, name: str) -> str:
        if name not in self._client.connections:
            return f"delete_mcp: no server named '{name}'."
        stop = self._stops.get(name)
        done = self._done.get(name)
        if stop is not None:
            stop.set()
        self._client.connections.pop(name, None)
        self._timeouts.pop(name, None)
        self._write_config()
        if done is not None:
            done.wait(timeout=10)
        return f"removed '{name}'."

    def list_servers(self) -> list[dict]:
        """Configured servers with live status."""
        out = []
        for name, conn in self._client.connections.items():
            entry = {
                "name": name,
                "command": conn.get("command"),
                "args": conn.get("args", []),
                "env_keys": list((conn.get("env") or {}).keys()),  # values redacted
                "connected": name in self._servers,
            }
            if name in self._servers:
                tools = self._servers[name]["tools"]
                entry["tool_count"] = len(tools)
                entry["tools"] = [t.name for t in tools]
            elif name in self._errors:
                entry["error"] = self._errors[name]
            out.append(entry)
        return out

    def all_tools(self) -> list:
        """Flat list of every wrapped MCP tool across all live servers."""
        out = []
        for srv in self._servers.values():
            out.extend(srv["tools"])
        return out

    # --- tool bridging ---------------------------------------------------

    def _wrap(self, mcp_tool, timeout: int) -> StructuredTool:
        """Wrap an async MCP tool in a sync StructuredTool dispatching to our loop."""
        loop = self._loop

        def _run(**kwargs):
            fut = asyncio.run_coroutine_threadsafe(mcp_tool.ainvoke(kwargs), loop)
            return fut.result(timeout=timeout)

        return StructuredTool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            args_schema=getattr(mcp_tool, "args_schema", None),
            func=_run,
        )

    # --- config persistence ---------------------------------------------

    def _write_config(self) -> None:
        """Persist current connections back to .mcp.json."""
        servers = {}
        for name, conn in self._client.connections.items():
            spec = dict(conn)
            if name in self._timeouts:
                spec["timeout"] = self._timeouts[name]
            servers[name] = spec
        try:
            Path(self._path).write_text(
                json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass  # best-effort; don't crash a turn over a config write

    # --- rebind (static + MCP tools into the executor + registry) --------

    def _rebind(self) -> None:
        """Rebind the executor LLM + tool registry with static + current MCP tools."""
        with self._rebind_lock:
            try:
                from tools import tools as static_tools
                from agents.coding_agent.model import configure_executor
                from agents.coding_agent.tools import configure_tools

                all_tools = list(static_tools) + self.all_tools()
                configure_executor(all_tools)
                configure_tools(all_tools)
            except Exception:
                # Rebind must never crash a turn; trace and move on.
                import traceback

                traceback.print_exc()


# Module-level singleton, set by main.py. The management tools reach it via
# `import plugins.mcp as mcpmod; mcpmod.manager`.
manager: McpManager | None = None


def rebind() -> None:
    """Shim so other modules can trigger a rebind without holding the manager ref."""
    if manager is not None:
        manager._rebind()
