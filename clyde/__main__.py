"""Clyde entry point: `clyde` console script (and `python -m clyde`).

Subcommands:
    clyde                   launch the TUI (the default)
    clyde login [prov...]   set API keys in the OS keychain (interactive)
    clyde logout [prov...]  remove keys (no args / --all => all)
    clyde auth              show configured providers + the active one

Credentials live in the OS keychain (see clyde.auth) — no .env, no env fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _run_tui() -> None:
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

    app = ClydeApp(builder=_build)

    # Route trace lines into the TUI transcript (dimmed) instead of stdout.
    _trace.compact_trace.set_sink(app.post_trace)

    try:
        app.run()
    finally:
        app.shutdown_manager()


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
    else:
        _run_tui()


if __name__ == "__main__":
    main()
