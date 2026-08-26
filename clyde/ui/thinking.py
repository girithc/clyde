"""Single-line animated 'thinking' indicator (Claude Code style).

Shown inline in the transcript while a turn runs. The line looks like:

    ✻ Nesting… (33s · ↓ 2.0k tokens)

- ``✻`` a spinner glyph that rotates every tick (~0.12s).
- the first word rotates through a fixed set of 50 gerunds (``Nesting``,
  ``Pondering``, ``Musing``…) every ~2.5s.
- ``33s`` elapsed wall-time since the turn started.
- ``↓ 2.0k tokens`` output tokens accumulated from the LLM trace this turn.

On the right of the row, the trace controls are shown while active
(``ctrl+t expand · ctrl+r mode``). When the turn finishes the line stops
animating and settles in place to ``Thought for 33s · ↓ 2.0k tokens`` and
stays until the next turn.
"""

from __future__ import annotations

import time

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

_SPINNER = ["✻", "✦", "✸", "✺", "✹", "✷", "✶", "✵", "✴", "✳"]

# 50 rotating first-words (Claude Code style gerunds).
_WORDS = [
    "Nesting", "Pondering", "Musing", "Thinking", "Working",
    "Processing", "Reasoning", "Deliberating", "Ruminating", "Reflecting",
    "Contemplating", "Considering", "Mulling", "Meditating", "Brainstorming",
    "Analyzing", "Synthesizing", "Formulating", "Composing", "Crafting",
    "Drafting", "Designing", "Structuring", "Organizing", "Planning",
    "Scheming", "Plotting", "Mapping", "Exploring", "Investigating",
    "Searching", "Probing", "Digging", "Uncovering", "Discovering",
    "Revising", "Refining", "Polishing", "Honing", "Tweaking",
    "Adjusting", "Assembling", "Building", "Constructing", "Developing",
    "Coding", "Programming", "Implementing", "Debugging", "Verifying",
]

_HINT = "ctrl+t expand"
_TICK_S = 0.12
_WORD_TICKS = round(2.5 / _TICK_S)  # advance the word every ~2.5s


def status_from_line(line: str) -> str:
    """Map a raw trace line to a short indicator status (unused for display now,
    kept for compatibility)."""
    if line.startswith("[LLM] start"):
        return "Thinking"
    if line.startswith("[Tool] start"):
        try:
            return "Calling " + line.split("—", 1)[1].split("(", 1)[0].strip()
        except Exception:
            return "Calling tool"
    return "Thinking"


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


class ThinkingIndicator(Static):
    """A one-line animated status indicator; settles to 'Thought for Xs'."""

    def __init__(self) -> None:
        super().__init__("")
        self._frame = 0
        self._word_idx = 0
        self._word_counter = 0
        self._tokens = 0
        self._started_at = 0.0

    def on_mount(self) -> None:
        self.set_interval(_TICK_S, self._tick)

    def _row(self, left: Text, hint: bool) -> Table:
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_row(left, Text(_HINT, style="dim") if hint else Text(""))
        return grid

    def _active_text(self) -> Text:
        spinner = _SPINNER[self._frame % len(_SPINNER)]
        word = _WORDS[self._word_idx % len(_WORDS)]
        elapsed = int(time.monotonic() - self._started_at) if self._started_at else 0
        tokens = f" · ↓ {_fmt_tokens(self._tokens)} tokens" if self._tokens else ""
        return Text(
            f"{spinner} {word}… ({elapsed}s{tokens})",
            style="dim",
        )

    def _tick(self) -> None:
        if self._started_at == 0.0:
            return
        self._frame = (self._frame + 1) % len(_SPINNER)
        self._word_counter += 1
        if self._word_counter >= _WORD_TICKS:
            self._word_counter = 0
            self._word_idx = (self._word_idx + 1) % len(_WORDS)
        self.update(self._active_text())

    def start(self) -> None:
        self._frame = 0
        self._word_idx = 0
        self._word_counter = 0
        self._tokens = 0
        self._started_at = time.monotonic()
        self.visible = True
        self.update(self._active_text())

    def add_tokens(self, n: int) -> None:
        """Accumulate output tokens reported by an [LLM] end trace line."""
        self._tokens += int(n or 0)
        if self._started_at:
            self.update(self._active_text())

    def done(self) -> None:
        """Stop animating; settle to 'Thought for Xs · ↓ Nk tokens' and stay."""
        if self._started_at == 0.0:
            return
        dur = time.monotonic() - self._started_at
        self._started_at = 0.0
        self.visible = True
        tokens = f" · ↓ {_fmt_tokens(self._tokens)} tokens" if self._tokens else ""
        self.update(
            Text(
                f"Thought for {dur:.1f}s{tokens}",
                style="dim",
            )
        )

    def hide(self) -> None:
        """Fully hide (used on shutdown)."""
        self._started_at = 0.0
        self.update("")
        self.visible = False
