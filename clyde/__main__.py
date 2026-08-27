"""Clyde entry point: `clyde` console script (and `python -m clyde`).

Subcommands:
    clyde                   launch the TUI (the default; starts a fresh session)
    clyde last session      resume the most recent session in this directory
    clyde all sessions      list every saved session
    clyde help              show all commands
    clyde login [prov...]   set API keys in the OS keychain (interactive)
    clyde logout [prov...]  remove keys (no args / --all => all)
    clyde auth              show configured providers + the active one

Sessions persist under ~/.clyde/sessions/<sid>/ (see clyde.sessions).
Credentials live in the OS keychain (see clyde.auth) — no .env, no env fallback.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


_HELP_CARD = """\
Clyde — a terminal coding agent.

Usage:
  clyde                       launch the TUI (fresh session)
  clyde last session          resume the most recent session in this directory
  clyde all sessions          list every saved session
  clyde help                  show this help
  clyde login [provider ...]  set API keys in the OS keychain (interactive)
  clyde logout [provider ...] remove keys (clyde logout --all clears every key)
  clyde auth                  show configured providers + the active one

Run `clyde` with no args to start coding. Quit with ctrl-c; the resume hint is
printed on exit. Sessions are stored locally at ~/.clyde/sessions/.
"""


def _print_help() -> None:
    print(_HELP_CARD)


def _list_sessions() -> None:
    """Print every saved session as a table (newest first)."""
    from rich.console import Console
    from rich.table import Table

    from clyde.sessions import list_sessions
    from clyde.sessions.meta import format_last_active

    metas = list_sessions()
    if not metas:
        print("No saved sessions yet. Start one with: clyde")
        return

    table = Table(title="Clyde sessions", show_header=True, header_style="bold")
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("cwd")
    table.add_column("msgs", justify="right")
    table.add_column("last active")
    table.add_column("model / title")
    for m in metas:
        cwd = m.get("cwd", "")
        cwd_disp = Path(cwd).name if cwd else ""
        title = m.get("title", "")
        model = (m.get("model_id") or "").split("/")[-1]
        label = f"{model} · {title}" if title else model
        table.add_row(m.get("id", ""), cwd_disp, str(m.get("message_count", 0)),
                      format_last_active(m), label)
    Console().print(table)


def _run_tui(resume_sid: str | None = None) -> None:
    # Light imports only — defer the agent/LLM graph until we know a key exists,
    # so `clyde` with no key prints a clean login hint instead of crashing on the
    # import-time LLM build.
    from clyde.auth import configured_providers, has_key
    from clyde.config import load_config, save_config
    from clyde.llm.registry import PROVIDERS

    cfg = load_config()
    provider = cfg.get("provider") or "fireworks"

    if not has_key(provider):
        have = configured_providers()
        if not have:
            print(f"No API key for '{provider}'. Run: clyde login {provider}")
            return
        # Auto-switch to a configured provider + its default model.
        provider = have[0]
        model_id = PROVIDERS[provider].default_model
        save_config({"provider": provider, "model_id": model_id})
        print(f"Active provider '{cfg.get('provider')}' has no key — switched to '{provider}'.")

    import clyde.trace as _trace
    from clyde.ui import ClydeApp

    # Resume: load the saved transcript into the app so it replays on launch.
    resume_messages = None
    if resume_sid is not None:
        from clyde.sessions import load_messages

        resume_messages = load_messages(resume_sid)
        if not resume_messages:
            print(f"Session {resume_sid} is empty or unreadable. Starting fresh.")
            resume_sid = None

    def _build():
        """Heavy work, run in a background worker after the TUI paints so the
        app appears instantly. Builds the LLM, compiles the graph, starts MCP,
        and loads skills. Returns (graph, history, skills, manager)."""
        from langchain_core.messages import SystemMessage

        from clyde.agents import default_graph as graph, default_system_prompt as system_prompt

        import clyde.plugins.mcp as mcpmod
        from clyde.plugins.mcp import McpManager
        from clyde.plugins.skills import builtin_skills_dir, load_skills

        # .mcp.json is per-project (CWD); a missing file just means no servers yet.
        manager = McpManager.from_config(".mcp.json")
        mcpmod.manager = manager  # management tools reach it via this singleton
        manager.start()
        manager._rebind()  # bind static + any pre-configured MCP tools

        # Skills: bundled (ship with the package) + user (~/.clyde/skills).
        skills = load_skills([builtin_skills_dir(), Path.home() / ".clyde" / "skills"])

        history = [SystemMessage(content=system_prompt)]
        return graph, history, skills, manager

    app = ClydeApp(builder=_build, session_id=resume_sid, resume_messages=resume_messages)

    # Route trace lines into the TUI transcript (dimmed) instead of stdout.
    _trace.compact_trace.set_sink(app.post_trace)

    try:
        app.run()
    finally:
        app.shutdown_manager()
        # Safety-net flush in case the last turn didn't settle, then show the
        # resume hint so the user knows how to pick the conversation back up.
        app.persist()
        if app.session_id is not None:
            from rich.console import Console
            from rich.text import Text

            Console().print(Text.assemble(
                "resume with ",
                Text("clyde last session", style="#888888"),
            ))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clyde",
        description="Clyde — a terminal coding agent. Run with no args to launch the TUI.",
    )
    sub = parser.add_subparsers(dest="command")

    p_login = sub.add_parser("login", help="set API keys in the OS keychain")
    p_login.add_argument("providers", nargs="*", help="providers to configure (default: prompt)")

    p_logout = sub.add_parser("logout", help="remove API keys from the keychain")
    p_logout.add_argument("providers", nargs="*", help="providers to remove (default: all)")
    p_logout.add_argument("--all", action="store_true", help="remove every key")

    sub.add_parser("auth", help="show configured providers + the active one")
    sub.add_parser("help", help="show all commands")

    # `clyde last session` / `clyde all sessions` — the trailing word is a
    # literal filler token consumed by argparse (kept for the natural phrasing).
    p_last = sub.add_parser("last", help="resume the most recent session in this directory")
    p_last.add_argument("session", help="literal 'session'")

    p_all = sub.add_parser("all", help="list every saved session")
    p_all.add_argument("sessions", help="literal 'sessions'")

    args = parser.parse_args()

    if args.command == "login":
        from clyde.auth import login
        login(args.providers or None)
    elif args.command == "logout":
        from clyde.auth import logout
        logout(None if (args.all or not args.providers) else args.providers)
    elif args.command == "auth":
        from clyde.auth import auth_status
        auth_status()
    elif args.command == "help":
        _print_help()
    elif args.command == "all":
        _list_sessions()
    elif args.command == "last":
        from clyde.sessions import latest_session
        sid = latest_session(os.getcwd())
        if sid is None:
            print(f"No sessions found in {os.getcwd()}.")
            return
        _run_tui(resume_sid=sid)
    else:
        _run_tui()


if __name__ == "__main__":
    main()
