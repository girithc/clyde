"""On-disk layout + per-session directory I/O for Clyde sessions.

Sessions live under ``~/.clyde/sessions/<sid>/`` — one directory per session
holding ``meta.json`` (metadata) and ``messages.jsonl`` (the transcript). All
access is best-effort: a missing/corrupt session is treated as absent and a
read-only home dir never crashes the app. Mirrors ``clyde.config``'s style.
"""

from __future__ import annotations

import json
from pathlib import Path

SESSIONS_DIR = Path.home() / ".clyde" / "sessions"


def _session_dir(sid: str) -> Path:
    return SESSIONS_DIR / sid


def _ensure_root() -> None:
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # persistence is a convenience, never fatal


def _read_meta(sid: str) -> dict:
    try:
        return json.loads((_session_dir(sid) / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def _write_meta(sid: str, meta: dict) -> None:
    try:
        d = _session_dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    except OSError:
        pass


def _read_lines(sid: str) -> list[str]:
    try:
        return (_session_dir(sid) / "messages.jsonl").read_text().splitlines()
    except OSError:
        return []


def _write_lines(sid: str, lines: list[str]) -> None:
    try:
        d = _session_dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "messages.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    except OSError:
        pass


def _all_session_ids() -> list[str]:
    try:
        return [p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()]
    except OSError:
        return []
