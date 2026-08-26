"""Scrolling conversation transcript.

A ``VerticalScroll`` that appends each piece of the conversation (user
messages, the live thinking indicator, trace lines, the streaming answer, the
final answer, "Thought for Xs") as child widgets, top-down — newest at the
bottom, auto-scrolled. Nothing is pinned except the input bar; the thinking
line lives inline like any other message, exactly like Claude Code.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class Transcript(VerticalScroll):
    """A vertically scrolling list of conversation line widgets."""

    async def append(self, renderable) -> Static:
        """Append a frozen line (a Static showing `renderable`)."""
        widget = Static(renderable, classes="tline")
        await self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)
        return widget

    async def append_live(self, widget) -> "VerticalScroll":
        """Append a live widget (indicator / streaming) that updates in place."""
        await self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)
        return widget
