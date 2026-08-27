"""Full trace block + per-turn tabbed panel.

``TracePanel`` is the one container per user turn in ``trace_mode = "full"``. It
holds a row of tabs along the top — a ``main`` tab for the supervisor / scout /
synthesize trace, plus one tab per worker sub-agent (labeled by the task the LLM
emitted) — so a fan-out no longer overflows one panel. Clicking a tab switches
the panel to that sub-agent's trace. Each tab's lane is a scrollable list of
events; the panel is height-capped so it scrolls internally instead of eating the
transcript.

Events arrive carrying a graph namespace (``TraceEvent.ns``); the panel routes
each to its lane: non-worker namespaces collapse to ``main``, ``worker:<uuid>``
gets its own tab. Event bodies are expanded-by-default ``Collapsible`` rows.

``TraceBlock`` remains as the fallback for trace events that fire with no active
panel (e.g. the greeting).
"""

from __future__ import annotations

from textual.containers import Container, VerticalScroll
from textual.widgets import Collapsible, Static, Tab, Tabs
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


class TracePanel(Container):
    """One tabbed panel per turn: main + a tab per worker sub-agent.

    A ``Tabs`` bar across the top switches which lane's scroll is visible below
    it. Events are routed to a lane by ``TraceEvent.ns``; a worker lane + tab is
    created on first sighting of a worker namespace. Height-capped; each lane
    scrolls internally.
    """

    DEFAULT_CSS = """
    TracePanel {
        height: auto;
        max-height: 12;
        border: round #555555;
        padding: 0;
        margin: 0 0 0 2;
        scrollbar-size: 0 0;
    }
    #trace-tabs {
        height: 1;
        dock: top;
        padding: 0;
    }
    #trace-content {
        height: 1fr;
        padding: 0;
    }
    TracePanel VerticalScroll {
        scrollbar-size: 0 0;
    }
    TracePanel Collapsible {
        height: auto;
        padding: 0;
        margin: 0;
    }
    TracePanel Static {
        height: auto;
    }
    """

    MAIN = "main"

    def __init__(self) -> None:
        super().__init__()
        self._lane_scrolls: dict[str, VerticalScroll] = {}
        # ns <-> valid Tab id (ns contains ':', which ids can't).
        self._ns_to_id: dict[str, str] = {}
        self._id_to_ns: dict[str, str] = {}
        self._worker_count = 0
        self._active_ns = self.MAIN

    def compose(self):
        self._lane_scrolls[self.MAIN] = VerticalScroll(id="main-scroll")
        self._ns_to_id[self.MAIN] = self.MAIN
        self._id_to_ns[self.MAIN] = self.MAIN
        yield Tabs(Tab(self.MAIN, id=self.MAIN), active=self.MAIN, id="trace-tabs")
        with Container(id="trace-content"):
            yield self._lane_scrolls[self.MAIN]

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Switch the visible lane to the one the user clicked."""
        ns = self._id_to_ns.get(event.tab.id, self.MAIN)
        await self._show_lane(ns)

    async def _show_lane(self, ns: str) -> None:
        self._active_ns = ns
        content = self.query_one("#trace-content")
        for lane_ns, scroll in self._lane_scrolls.items():
            scroll.styles.display = "block" if lane_ns == ns else "none"
        # Scroll the now-visible lane to its end.
        scroll = self._lane_scrolls.get(ns)
        if scroll is not None:
            scroll.call_after_refresh(scroll.scroll_end, animate=False)

    async def add_event(self, event: TraceEvent) -> None:
        """Route one trace event to its lane, creating a worker tab on first sight."""
        ns = event.ns or ""
        # Non-worker namespaces (planner, scout, synthesize, direct_answer, "")
        # all collapse to the main tab.
        lane_key = ns if ns.startswith("worker:") else self.MAIN

        scroll = self._lane_scrolls.get(lane_key)
        if scroll is None:
            # First event for this worker: create its lane + tab.
            label = event.label or f"worker {self._worker_count + 1}"
            self._worker_count += 1
            lane_id = f"lane{self._worker_count}"
            self._ns_to_id[lane_key] = lane_id
            self._id_to_ns[lane_id] = lane_key
            scroll = VerticalScroll(id=f"{lane_id}-scroll")
            self._lane_scrolls[lane_key] = scroll
            tabs = self.query_one("#trace-tabs")
            tabs.add_tab(Tab(label, id=lane_id))
            # Mount hidden; only the active lane is shown.
            scroll.styles.display = "none"
            await self.query_one("#trace-content").mount(scroll)

        # Render the event into the lane's scroll.
        if event.body:
            await scroll.mount(
                Collapsible(
                    Static(Text(event.body, style="dim")),
                    title=event.header,
                    collapsed=False,
                )
            )
        else:
            await scroll.mount(Static(Text(event.header, style="dim")))

        # Follow the stream in the lane the user is currently viewing.
        if lane_key == self._active_ns:
            scroll.call_after_refresh(scroll.scroll_end, animate=False)
