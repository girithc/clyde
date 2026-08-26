"""Gather session context for the proactive greeting.

The greeting is context-aware: it knows the time of day, the user's name, the project, and what they were doing last (the most recent commit). Everything is best-effort — if git isn't available or the dir isn't a repo, the missing pieces fall back to harmless defaults so the greet never fails.
"""

from __future__ import annotations

import datetime
import os
import subprocess


def _run(args: list[str]) -> str:
    """Run a git command, returning stdout stripped; '' on any failure."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=3
        ).stdout.strip()
    except Exception:
        return ""


def session_context() -> dict:
    """Best-effort session context for the greeting prompt."""
    ctx: dict[str, str] = {}

    now = datetime.datetime.now()
    ctx["time"] = now.strftime("%H:%M on %A")

    name = _run(["git", "config", "user.name"]) or "friend"
    ctx["user"] = name

    origin = _run(["git", "remote", "get-url", "origin"])
    if origin:
        # https://github.com/owner/repo.git -> repo
        tail = origin.rstrip("/").split("/")[-1]
        ctx["repo"] = tail[:-4] if tail.endswith(".git") else tail
    else:
        ctx["repo"] = os.path.basename(os.getcwd())

    ctx["last_commit"] = _run(["git", "log", "-1", "--pretty=%s"])

    return ctx
