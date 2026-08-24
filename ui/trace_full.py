"""Full trace line producer.

Returns a dim ``Text`` for a raw trace line, to be appended inline in the
transcript when the thinking view is expanded with ``trace_mode = "full"``.
"""

from __future__ import annotations

from rich.text import Text


def full_trace_renderable(line: str) -> Text:
    """The full event line, dimmed."""
    return Text(line, style="dim")
