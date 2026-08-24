"""Skill loader and trigger matcher.

A skill is a markdown file (``SKILL.md`` or ``*.skill``) with YAML frontmatter:

    ---
    name: caveman
    description: >
      Ultra-compressed mode. Use when user says "caveman mode", "be brief",
      or invokes /caveman.
    ---
    Respond terse like smart caveman...

Skills are pure prompt text — no runtime. We scan a local ``./plugins`` tree,
parse the frontmatter, and on each user turn inject the body of any skill whose
triggers match the user's input as a ``SystemMessage``. The text then persists
in the transcript for subsequent turns (persistence wording in the skill body
itself, e.g. "ACTIVE EVERY RESPONSE", is honored by the model).

Trigger phrases are extracted heuristically from the ``description``: any quoted
string plus the skill ``name``. This is fuzzy by design — plugin authors phrase
triggers as prose ("Use when user says 'X'"), not as a structured list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    """One loaded skill."""

    name: str
    description: str
    body: str
    triggers: list[str]

    def matches(self, user_input: str) -> bool:
        text = user_input.lower()
        return any(t in text for t in self.triggers)


# Files that define a skill, anywhere under the scan root.
_SKILL_GLOBS = ("**/SKILL.md", "**/*.skill")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n?(.*)\Z", re.DOTALL)
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,})['\"]")


def _extract_triggers(name: str, description: str) -> list[str]:
    """Lowercased trigger phrases: quoted strings in the description + the name."""
    triggers = {m.group(1).lower() for m in _QUOTED_RE.finditer(description or "")}
    triggers.add(name.lower())
    return sorted(t for t in triggers if t)


def _parse_skill(path: Path) -> Skill | None:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        # No frontmatter — not a skill we can drive; skip.
        return None
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None

    name = str(meta.get("name") or path.stem).strip()
    if not name:
        return None
    description = str(meta.get("description") or "")
    body = match.group(2).strip()
    return Skill(
        name=name,
        description=description,
        body=body,
        triggers=_extract_triggers(name, description),
    )


def load_skills(root: str = "./plugins") -> list[Skill]:
    """Load every skill under ``root``. Returns [] if the dir is missing."""
    base = Path(root)
    if not base.is_dir():
        return []

    seen: set[str] = set()
    skills: list[Skill] = []
    for pattern in _SKILL_GLOBS:
        for path in base.glob(pattern):
            skill = _parse_skill(path)
            if skill is None or skill.name in seen:
                continue
            seen.add(skill.name)
            skills.append(skill)
    return skills


def match_skills(user_input: str, skills: list[Skill]) -> list[Skill]:
    """Return every skill whose triggers match the user's input."""
    if not user_input.strip():
        return []
    return [s for s in skills if s.matches(user_input)]
