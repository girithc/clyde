"""Full trace block + per-turn panel widgets.

``TracePanel`` is the one container per user turn in ``trace_mode = "full"``:
the whole trace (every event) plus the agent's final answer live inside it,
height-capped so a large payload scrolls internally instead of eating the
transcript. Each event with a body is a click-to-collapse ``Collapsible``
(prompts, tool I/O, intermediate LLM responses), collapsed by default; the
final answer renders expanded at the bottom.

``TraceBlock`` remains as the fallback for trace events that fire with no
active panel (e.g. the greeting).
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static
from rich.text import Text

from clyde.trace import TraceEvent


class TraceBlock(VerticalScroll):
    """A scrollable, height-capped trace event block (header + optional body).

    Fallback used when no per-turn ``TracePanel`` is active.
    """

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


class TracePanel(VerticalScroll):
    """One scrollable, height-capped panel per turn holding the full trace + answer.

    Events are added via ``add_event``; the streaming/answer widget is added via
    ``append``. Both mount children here and scroll to the bottom so the panel
    follows the live stream.
    """

    DEFAULT_CSS = """
    TracePanel {
        height: auto;
        max-height: 12;
        border: round #555555;
        padding: 0 1;
        margin: 0 0 0 2;
        scrollbar-size: 0 0;
    }
    TracePanel Static {
        height: auto;
    }
    TracePanel Collapsible {
        height: auto;
        padding: 0;
        margin: 0;
    }
    """

    async def append(self, widget) -> None:
        """Mount a child widget (e.g. the streaming answer) and follow it."""
        await self.mount(widget)
        self._follow()

    async def add_event(self, event: TraceEvent) -> None:
        """Render one trace event: an expanded Collapsible if it has a body,
        otherwise a plain dim header line."""
        if event.body:
            await self.mount(
                Collapsible(
                    Static(Text(event.body, style="dim")),
                    title=event.header,
                    collapsed=False,
                )
            )
        else:
            await self.mount(Static(Text(event.header, style="dim")))
        self._follow()

    def _follow(self) -> None:
        """Pin the panel to the bottom so the latest event stays in view.

        Expanded Collapsibles lay out after mount, so we scroll on the next
        refresh and again a tick later once their body height is known.
        """
        self.call_after_refresh(self.scroll_end, animate=False)
        self.set_timer(0.05, lambda: self.scroll_end(animate=False))
