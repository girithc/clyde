"""Clyde session persistence — public API.

Sessions are stored on disk under ``~/.clyde/sessions/<sid>/`` (see
``store.py``) as ``meta.json`` + ``messages.jsonl``. This package re-exports a
small surface so callers never touch the internal modules:

    from clyde.sessions import (
        create_session, save_messages, load_messages, load_meta,
        touch, set_title, list_sessions, latest_session,
    )

All operations are best-effort: a missing/corrupt session is treated as absent
and never raises.
"""

from __future__ import annotations

import uuid
from typing import Iterable

from langchain_core.messages import BaseMessage

from clyde.sessions import meta as _meta
from clyde.sessions import serialize as _ser
from clyde.sessions import store as _store

__all__ = [
    "create_session",
    "save_messages",
    "load_messages",
    "load_meta",
    "touch",
    "set_title",
    "list_sessions",
    "latest_session",
]


def create_session(cwd: str, provider: str, model_id: str) -> str:
    """Create a new session on disk and return its id."""
    _store._ensure_root()
    sid = uuid.uuid4().hex[:12]
    _store._write_meta(sid, _meta.new_meta(cwd, provider, model_id, sid))
    return sid


def save_messages(sid: str, messages: Iterable[BaseMessage], system_prompt: str = "") -> None:
    """Persist the full message history for ``sid``.

    The base agent system prompt (matched by content) is filtered out so it
    isn't duplicated on reload; the caller re-prepends a fresh one. Also bumps
    ``last_active`` and derives the title from the first HumanMessage.
    """
    msgs = list(messages)
    lines = _ser.dump_messages(msgs, system_prompt)
    _store._write_lines(sid, lines)
    m = _store._read_meta(sid)
    if not m:
        return
    # Count the persisted (base-prompt-filtered) messages, not the raw history,
    # so the number matches what's actually in messages.jsonl.
    m["message_count"] = len(lines)
    _meta.touch(m)
    _meta.set_title(m, _ser.first_human_text(msgs))
    _store._write_meta(sid, m)


def load_messages(sid: str) -> list[BaseMessage]:
    """Read a session's message history (empty list if missing/corrupt)."""
    return _ser.load_messages(_store._read_lines(sid))


def load_meta(sid: str) -> dict:
    """Read a session's metadata (empty dict if missing/corrupt)."""
    return _store._read_meta(sid)


def touch(sid: str) -> None:
    """Bump a session's ``last_active`` to now."""
    m = _store._read_meta(sid)
    if m:
        _meta.touch(m)
        _store._write_meta(sid, m)


def set_title(sid: str, text: str) -> None:
    """Set a session's title from ``text`` (first user message)."""
    m = _store._read_meta(sid)
    if m:
        _meta.set_title(m, text)
        _store._write_meta(sid, m)


def list_sessions(cwd: str | None = None) -> list[dict]:
    """All sessions, newest-first. Optionally filtered by ``cwd``.

    Zero-message sessions (an immediate quit before any turn) are hidden so
    they don't clutter ``clyde all sessions``.
    """
    metas = []
    for sid in _store._all_session_ids():
        m = _store._read_meta(sid)
        if not m:
            continue
        if m.get("message_count", 0) == 0:
            continue
        if cwd is not None and m.get("cwd") != cwd:
            continue
        metas.append(m)
    metas.sort(key=lambda m: m.get("last_active", ""), reverse=True)
    return metas


def latest_session(cwd: str) -> str | None:
    """The most recent session id in ``cwd``, or None if there is none."""
    metas = list_sessions(cwd=cwd)
    return metas[0]["id"] if metas else None
