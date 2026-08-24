"""Minimalist trace line producer.

Returns a dim ``Text`` summary of a raw trace line (e.g. "thinking",
"calling read_file", "done 3.12s"), to be appended inline in the transcript
when the thinking view is expanded with ``trace_mode = "minimal"``.
"""

from __future__ import annotations

from rich.text import Text


def _summarize(line: str) -> str:
    """Compress a raw trace line into a short label."""
    if line.startswith("[LLM] start"):
        return "thinking"
    if line.startswith("[LLM] end"):
        # "[LLM] end — 3.12s, 294 out tokens, finish=stop"
        try:
            right = line.split("—", 1)[1]
            secs = right.split(",", 1)[0].strip()
            return f"  done {secs}"
        except Exception:
            return "  done"
    if line.startswith("[Tool] start"):
        # "[Tool] start — read_file({'file_path': ...})"
        try:
            rest = line.split("—", 1)[1]
            name = rest.split("(", 1)[0].strip()
            return f"calling {name}"
        except Exception:
            return "calling tool"
    if line.startswith("[Tool] end"):
        return "  tool done"
    return line


def minimal_trace_renderable(line: str) -> Text:
    """A compact one-line summary of the event, dimmed."""
    return Text(_summarize(line), style="dim")
