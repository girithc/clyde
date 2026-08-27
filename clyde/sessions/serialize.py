"""BaseMessage <-> JSONL serialization for session storage.

Each line of ``messages.jsonl`` is one ``BaseMessage`` serialized via
langchain_core's ``messages_to_dict`` / ``messages_from_dict``. The base agent
system prompt is filtered out on dump (it's re-added fresh on load) so a prompt
change between sessions doesn't leave a stale duplicate in history.
"""

from __future__ import annotations

import json
from typing import Iterable

from langchain_core.messages import BaseMessage, SystemMessage, messages_from_dict, messages_to_dict


def dump_messages(messages: Iterable[BaseMessage], system_prompt: str) -> list[str]:
    """Serialize messages to a list of JSON strings, dropping the base prompt.

    The base system prompt is identified by content equality (not position):
    ``_stream`` reassigns ``self.history`` from the graph's final values, so
    index 0 is not guaranteed to be the prompt.
    """
    prompt = (system_prompt or "").strip()
    kept = []
    for m in messages:
        if isinstance(m, SystemMessage) and prompt and (m.content or "").strip() == prompt:
            continue
        kept.append(m)
    return [json.dumps(d) for d in messages_to_dict(kept)]


def load_messages(lines: Iterable[str]) -> list[BaseMessage]:
    """Parse JSONL lines back into ``BaseMessage`` objects.

    Blank/corrupt lines are skipped so a partially-written file (e.g. a crash
    mid-flush) degrades to the last complete message instead of raising.
    """
    dicts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            dicts.append(json.loads(line))
        except ValueError:
            continue
    if not dicts:
        return []
    return messages_from_dict(dicts)


def first_human_text(messages: Iterable[BaseMessage]) -> str:
    """The text of the first HumanMessage, for the session title.

    Multimodal content (a list of parts) is flattened to its text parts; if
    there are none, returns "".
    """
    from langchain_core.messages import HumanMessage

    for m in messages:
        if not isinstance(m, HumanMessage):
            continue
        c = m.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(t for t in texts if t)
        return ""
    return ""
