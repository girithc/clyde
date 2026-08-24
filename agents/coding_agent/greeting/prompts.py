"""Prompt strings for the proactive session greeting."""

from __future__ import annotations

from agents.coding_agent.greeting.context import session_context

GREET_PROMPT = (
    "You are Clyde, a coding assistant. Greet the user briefly and warmly to start "
    "the session. Reference the context (the time, their name, the project, "
    "what they were doing last) naturally, as if you noticed it. Keep it to one or "
    "two short sentences. No markdown headers, no preamble, no list of what you can do."
)


def build_greet_prompt() -> str:
    """Build the greeting prompt from live session context."""
    c = session_context()
    parts = [f"Time: {c['time']}"]
    if c.get("user"):
        parts.append(f"User: {c['user']}")
    if c.get("repo"):
        parts.append(f"Project: {c['repo']}")
    if c.get("last_commit"):
        parts.append(f"Last commit: {c['last_commit']}")
    return "\n".join(parts)
