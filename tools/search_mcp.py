"""Search-the-internet MCP discovery tool.

``search_mcp(query)`` finds MCP servers matching a use case (e.g. "notion",
"postgres", "github") by fetching the curated ``awesome-mcp-servers`` list and
filtering entries. Returns name, repo URL, and description per match. The agent
then calls ``add_mcp`` to register the chosen one.

This is the discovery half of the manage flow: ``search_mcp`` (find on the
internet) -> ``add_mcp`` (register + hot-start) -> ``get_mcp`` (list local).
"""

from __future__ import annotations

import re
import threading
import urllib.request

from langchain_core.tools import tool

_AWESOME_URL = (
    "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md"
)

# `- [Name](url) ...rest`
_ENTRY_RE = re.compile(r"^-\s+\[([^\]]+)\]\(([^)]+)\)\s*(.*)$")
# Glama badge: `[![alt](badge.svg)](glama-url)`
_BADGE_RE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")

_CACHE: str | None = None
_CACHE_LOCK = threading.Lock()


def _fetch_list() -> str:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:
            return _CACHE
        req = urllib.request.Request(_AWESOME_URL, headers={"User-Agent": "clyde/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            _CACHE = resp.read().decode("utf-8", "replace")
        return _CACHE


def _parse(text: str) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        m = _ENTRY_RE.match(line)
        if not m:
            continue
        name, url, rest = m.group(1), m.group(2), m.group(3)
        rest = _BADGE_RE.sub("", rest)
        # Description is everything after the first " - " separator.
        desc = rest.split(" - ", 1)[1].strip() if " - " in rest else rest.strip()
        entries.append((name, url, desc))
    return entries


@tool
def search_mcp(query: str) -> str:
    """Search the internet for MCP servers matching a use case.

    Pass a use case or keyword, e.g. "notion", "postgres database", "github",
    "slack". Returns up to 15 matches with name, repo URL, and description.
    Use ``add_mcp`` to register one you pick.
    """
    try:
        entries = _parse(_fetch_list())
    except Exception as e:
        return f"search_mcp: failed to fetch the MCP directory: {type(e).__name__}: {e}"

    terms = [t for t in query.lower().split() if t]
    if not terms:
        return "search_mcp: provide a use-case query, e.g. 'notion' or 'postgres database'."

    def matches(pred) -> list[tuple[str, str, str]]:
        out = []
        for name, url, desc in entries:
            blob = (name + " " + desc).lower()
            if pred(blob):
                out.append((name, url, desc))
        return out

    result = matches(lambda b: all(t in b for t in terms))
    if not result:  # fall back to ANY term
        result = matches(lambda b: any(t in b for t in terms))
    if not result:
        return f"No MCP servers found for '{query}'."

    result = result[:15]
    lines = [f"Found {len(result)} MCP server(s) for '{query}':"]
    for name, url, desc in result:
        d = desc if len(desc) <= 160 else desc[:160] + "…"
        lines.append(f"- {name} — {d}\n  {url}")
    return "\n".join(lines)
