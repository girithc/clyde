"""Session metadata model + relative-time formatting.

A session's ``meta.json`` is a small dict (mirroring ``clyde.config``'s dict
style — no dataclass, so a corrupt/old file degrades to ``{}`` cleanly). All
time fields are ISO-8601 strings in the local timezone.
"""

from __future__ import annotations

import datetime
from typing import Any


# Fields every meta.json carries. Kept explicit for readability; not enforced.
META_KEYS = (
    "id",
    "cwd",
    "provider",
    "model_id",
    "created_at",
    "last_active",
    "message_count",
    "title",
)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_meta(cwd: str, provider: str, model_id: str, sid: str) -> dict[str, Any]:
    """Build a fresh metadata dict for a just-created session."""
    now = _now_iso()
    return {
        "id": sid,
        "cwd": cwd,
        "provider": provider,
        "model_id": model_id,
        "created_at": now,
        "last_active": now,
        "message_count": 0,
        "title": "",
    }


def touch(meta: dict) -> dict:
    """Bump ``last_active`` to now. Returns the same dict (mutated in place)."""
    if meta:
        meta["last_active"] = _now_iso()
    return meta


def set_title(meta: dict, text: str) -> dict:
    """Set the session title from the first user message text (truncated)."""
    if not meta:
        return meta
    title = (text or "").strip().replace("\n", " ")
    if len(title) > 60:
        title = title[:57].rstrip() + "…"
    # Don't overwrite a title already set with a non-empty one.
    if title and not meta.get("title"):
        meta["title"] = title
    return meta


def format_last_active(meta: dict) -> str:
    """Short relative time: 'just now', '5m ago', '3h ago', '2d ago', or a date."""
    raw = (meta or {}).get("last_active")
    if not raw:
        return "—"
    try:
        ts = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return "—"
    delta = datetime.datetime.now() - ts
    secs = int(delta.total_seconds())
    if secs < 0:
        return "just now"
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    if secs < 86400 * 7:
        return f"{secs // 86400}d ago"
    return ts.strftime("%Y-%m-%d")
