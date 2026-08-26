"""Full trace block widget.

A height-capped, internally scrollable ``TraceBlock`` mounted in the transcript
when ``trace_mode = "full"``. The header is the one-line event summary; the body
is the full content (LLM input messages, full response text, or untruncated tool
output). Capping ``max-height`` keeps a huge payload from eating the transcript
— the block scrolls internally instead.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static
from rich.text import Text


class TraceBlock(VerticalScroll):
    """A scrollable, height-capped trace event block (header + optional body)."""

    DEFAULT_CSS = """
    TraceBlock {
        height: auto;
        max-height: 16;
        border: round #555555;
        padding: 0 1;
        margin: 0 0 0 2;
        scrollbar-size: 0 0;
    }
    TraceBlock Static {
        height: auto;
    }
    """

    def __init__(self, header: str, body: str | None = None) -> None:
        super().__init__()
        self._header = header
        self._body = body

    def compose(self):
        yield Static(Text(self._header, style="dim bold"))
        if self._body:
            yield Static(Text(self._body, style="dim"))
