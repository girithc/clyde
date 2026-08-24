"""Clyde textual TUI: a scrolling transcript + fixed input bar at the bottom.

Everything in the conversation flows top-down in one scrolling transcript —
user messages, the live "thinking" indicator, trace lines (when expanded),
the streaming answer, and the final "Thought for Xs" line. Nothing is pinned
except the input bar; the thinking line lives inline like any other message,
exactly like Claude Code.

The LangGraph `graph.stream` is sync and blocking; textual is async. Each
turn runs in a worker thread (`run_worker(..., thread=True)`) and posts
**non-blocking** messages back to the UI thread, which awaits mounts into the
transcript.

A message submitted while a turn is running is surfaced **inline immediately**
(within the current turn's flow) and processed as a seamless continuation
right after the current turn — no separate "Planning" block / new-turn framing.
"""

from __future__ import annotations

import time
from collections import deque

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rich.markdown import Markdown
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Input, Select, Static

import llm
from llm.registry import PROVIDERS
from ui.renderer import (
    agent_renderable,
    spacer_renderable,
    user_renderable,
    welcome_renderable,
)
from ui.thinking import ThinkingIndicator
from ui.trace_full import full_trace_renderable
from ui.trace_minimal import minimal_trace_renderable
from ui.transcript import Transcript


# --- worker -> UI messages (posted from the turn thread, handled on the UI thread) ---

class _TraceLine(Message):
    def __init__(self, line: str) -> None:
        self.line = line
        super().__init__()


class _StreamChunk(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class _AgentContent(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class _TurnDone(Message):
    pass


class TraceStatus(Static):
    """Clickable trace-mode status (none / minimal / full)."""

    def on_click(self, event) -> None:
        self.app.cycle_trace_mode()


class StatusSep(Static):
    """A muted '>>' separator between status groups (disabled-UI color)."""

    def __init__(self) -> None:
        super().__init__(">>", classes="sep")


class ModeStatus(Static):
    """Clickable agent-mode status (auto / plan / ask)."""

    def on_click(self, event) -> None:
        self.app.cycle_agent_mode()


class ModelStatus(Static):
    """Status showing the current model id; click to edit inline."""

    def on_click(self, event) -> None:
        self.app.enter_model_edit()


_CSS = """
Screen {
    background: black;
}
#transcript {
    height: 1fr;
    padding: 0 1;
    scrollbar-size: 0 0;
}
.tline {
    padding: 0;
    height: auto;
}
#input-bar {
    height: auto;
    border: round white;
    padding: 0;
}
#input-bar Input {
    border: none;
    padding: 0 2;
    background: transparent;
    width: 1fr;
}
.model-edit {
    display: none;
}
#provider {
    border: none;
    padding: 0 2;
    background: transparent;
    width: 18;
    min-width: 0;
}
#model-id {
    border: none;
    padding: 0 2;
    background: transparent;
    width: 1fr;
}
#confirm {
    border: none;
    margin: 0 1;
    min-width: 0;
    height: 1;
    padding: 0 2;
    align: center middle;
}
#close {
    border: none;
    margin: 0 1;
    min-width: 0;
    height: 1;
    padding: 0 2;
    align: right top;
}
#status-bar {
    height: 1;
    padding: 0;
}
#trace-status {
    color: $accent;
    width: auto;
    min-width: 0;
    margin: 0 1 0 0;
}
.sep {
    color: $text-muted;
    width: auto;
    min-width: 0;
    margin: 0 1;
}
#mode-status {
    color: $accent;
    width: auto;
    min-width: 0;
    margin: 0 1;
}
#model-status {
    color: $accent;
    width: auto;
    min-width: 0;
    margin: 0 0 0 1;
}
#exit-hint {
    color: $text-muted;
    width: 1fr;
    min-width: 0;
    text-align: right;
}
"""

# Throttle live stream updates so we don't post a message per token.
_STREAM_THROTTLE_S = 0.03


class ClydeApp(App):
    """Clyde TUI: scrolling transcript + fixed input bar, inline thinking."""

    CSS = _CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+t", "toggle_trace", "Toggle trace", priority=True),
        Binding("escape", "cancel_model_edit", "Cancel", priority=True),
    ]

    def __init__(self, graph, history, skills):
        super().__init__()
        self.graph = graph
        self.history = history
        self.skills = skills
        self._busy = False
        self._pending: list = []  # mid-turn messages, processed as a continuation
        self._pending_model: tuple | None = None  # (provider, model_id) applied after a turn
        self.trace_mode = "none"  # "none" | "minimal" | "full"
        self.agent_mode = "auto"  # "auto" | "plan" | "ask" (UI state only)
        self._edit_model = False  # inline model-edit mode in the input bar
        self.indicator: ThinkingIndicator | None = None
        self.streaming: Static | None = None

    # --- layout ---

    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")
        with Horizontal(id="input-bar"):
            yield Input(id="input", placeholder="Message…")
            yield Select(
                [(p, p) for p in PROVIDERS],
                id="provider",
                classes="model-edit",
            )
            yield Input(id="model-id", classes="model-edit")
            yield Button("Confirm", id="confirm", classes="model-edit")
            yield Button("✕", id="close", classes="model-edit")
        with Horizontal(id="status-bar"):
            yield TraceStatus(id="trace-status")
            yield StatusSep()
            yield ModeStatus(id="mode-status")
            yield StatusSep()
            yield ModelStatus(id="model-status")
            yield Static("ctrl-c to exit", id="exit-hint")

    async def on_mount(self) -> None:
        self.transcript = self.query_one("#transcript", Transcript)
        self.input = self.query_one("#input", Input)
        self.provider_select = self.query_one("#provider", Select)
        self.model_id_input = self.query_one("#model-id", Input)
        self.trace_status = self.query_one("#trace-status", TraceStatus)
        self.mode_status = self.query_one("#mode-status", ModeStatus)
        self.model_status = self.query_one("#model-status", ModelStatus)
        self.update_trace_status()
        self.update_mode_status()
        self.update_model_status()
        self._apply_edit_visibility()
        await self.transcript.append(welcome_renderable())
        self.input.focus()

    # --- trace mode (none / minimal / full) ---

    _TRACE_MODES = ("none", "minimal", "full")

    def cycle_trace_mode(self) -> None:
        i = self._TRACE_MODES.index(self.trace_mode) if self.trace_mode in self._TRACE_MODES else 0
        self.trace_mode = self._TRACE_MODES[(i + 1) % len(self._TRACE_MODES)]
        self.update_trace_status()
        self.notify(f"trace: {self.trace_mode}", timeout=2)

    def update_trace_status(self) -> None:
        self.trace_status.update(f"trace: {self.trace_mode}")

    def action_toggle_trace(self) -> None:
        self.cycle_trace_mode()

    # --- agent mode (auto / plan / ask) — UI state only, not wired to the planner ---

    _AGENT_MODES = ("auto", "plan", "ask")

    def cycle_agent_mode(self) -> None:
        i = self._AGENT_MODES.index(self.agent_mode) if self.agent_mode in self._AGENT_MODES else 0
        self.agent_mode = self._AGENT_MODES[(i + 1) % len(self._AGENT_MODES)]
        self.update_mode_status()
        self.notify(f"mode: {self.agent_mode}", timeout=2)

    def update_mode_status(self) -> None:
        self.mode_status.update(f"mode: {self.agent_mode}")

    # --- model selection (provider + model id) — inline in the input bar ---

    def enter_model_edit(self) -> None:
        """Swap the input bar into an inline provider/model edit row."""
        self._edit_model = True
        self.provider_select.value = llm.CURRENT_PROVIDER
        self.model_id_input.value = llm.CURRENT_MODEL_ID
        self._apply_edit_visibility()
        self.model_id_input.focus()

    def exit_model_edit(self) -> None:
        """Revert the input bar to the normal message input."""
        self._edit_model = False
        self._apply_edit_visibility()
        self.input.focus()

    def _apply_edit_visibility(self) -> None:
        """Show the message input OR the inline model-edit row."""
        if self._edit_model:
            self.input.styles.display = "none"
            for w in self.query(".model-edit"):
                w.styles.display = "block"
        else:
            self.input.styles.display = "block"
            for w in self.query(".model-edit"):
                w.styles.display = "none"

    def action_cancel_model_edit(self) -> None:
        if self._edit_model:
            self.exit_model_edit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            provider = self.provider_select.value
            model_id = self.model_id_input.value.strip()
            if not model_id:
                self.notify("Enter a model id.", timeout=3)
                return
            self.exit_model_edit()
            self._apply_model(provider, model_id)
        elif event.button.id == "close":
            self.exit_model_edit()

    def update_model_status(self) -> None:
        self.model_status.update(f"model: {llm.CURRENT_MODEL_ID.split('/')[-1]}")

    def _apply_model(self, provider: str, model_id: str) -> None:
        """Rebuild every LLM with the new provider + model.

        Deferred to between turns if a turn is running, so the running worker keeps
        the old LLM until the turn ends; the next turn uses the new model.
        """
        if self._busy:
            self._pending_model = (provider, model_id)
            self.notify(f"model: {model_id} (queued — applies after current turn)", timeout=3)
            return
        try:
            from llm import set_model
            from tools import tools as _static_tools
            import plugins.mcp as _mcpmod
            import agents.coding_agent.model as _model
            import agents.coding_agent.supervisor.nodes as _sup
            import agents.coding_agent.worker.nodes as _worker
            import tools.summarize as _sum

            set_model(provider, model_id)
            mcp_tools = _mcpmod.manager.all_tools() if _mcpmod.manager is not None else []
            all_tools = list(_static_tools) + mcp_tools
            _model.configure_model(provider, model_id, all_tools)
            _sup.configure_model(provider, model_id)
            _worker.configure_model(provider, model_id)
            # tools.summarize imports as the StructuredTool (name collision), not the
            # module — reach the module via sys.modules to call its configure_model.
            import sys

            _sum = sys.modules.get("tools.summarize")
            if _sum is not None and hasattr(_sum, "configure_model"):
                _sum.configure_model(provider, model_id)
            self.update_model_status()
            self.notify(f"model: {model_id}", timeout=3)
        except Exception as e:
            self.notify(f"model switch failed: {e}", timeout=5)

    # --- message handlers (UI thread; async so we can mount into the transcript) ---

    async def on__trace_line(self, message: _TraceLine) -> None:
        line = message.line
        if self.indicator is not None and line.startswith("[LLM] end"):
            # Accumulate output tokens: "[LLM] end — 3.12s · ↑ 1234 in · ↓ 294 out · …"
            import re

            m = re.search(r"↓\s*(\d+)\s*out", line)
            if m:
                self.indicator.add_tokens(int(m.group(1)))
        if self.trace_mode == "full":
            await self.transcript.append(full_trace_renderable(line))
        elif self.trace_mode == "minimal":
            await self.transcript.append(minimal_trace_renderable(line))
        # "none" -> no inline trace line

    async def on__stream_chunk(self, message: _StreamChunk) -> None:
        if self.streaming is None:
            self.streaming = Static("")
            await self.transcript.append_live(self.streaming)
        self.streaming.update(message.text)

    async def on__agent_content(self, message: _AgentContent) -> None:
        # Freeze the streaming widget into the rendered Markdown answer.
        if self.streaming is None:
            await self.transcript.append(agent_renderable(message.text))
        else:
            self.streaming.update(agent_renderable(message.text))
            self.streaming = None

    async def on__turn_done(self, message: _TurnDone) -> None:
        # Settle the indicator inline to "Thought for Xs" and leave it there.
        if self.indicator is not None:
            self.indicator.done()
            self.indicator = None
        # Apply a deferred model switch now that the turn is done.
        if self._pending_model is not None:
            provider, model_id = self._pending_model
            self._pending_model = None
            self._apply_model(provider, model_id)
            return
        # Continue with any mid-turn messages (seamless, no new-turn framing).
        if self._pending:
            self.history.extend(self._pending)
            self._pending = []
            await self._start_continue()
        else:
            self._busy = False

    # --- trace sink target (called from main.py on the worker/MCP threads) ---

    def post_trace(self, line: str) -> None:
        self.post_message(_TraceLine(line))

    # --- input handling ---

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if text.lower() in ("exit", "quit"):
            self.exit()
            return
        self.input.value = ""
        await self.transcript.append(user_renderable(text))
        await self.transcript.append(spacer_renderable())
        if self._busy:
            # Mid-turn: surface inline now, process as a continuation after.
            from plugins.skills import match_skills  # local import avoids a cycle

            pending = []
            for skill in match_skills(text, self.skills):
                pending.append(
                    SystemMessage(content=f"[Skill: {skill.name}]\n{skill.body}")
                )
            pending.append(HumanMessage(content=text))
            self._pending.extend(pending)
        else:
            await self._start_turn(text)

    async def _start_turn(self, text: str) -> None:
        self._busy = True
        self.streaming = None
        self.indicator = ThinkingIndicator()
        await self.transcript.append_live(self.indicator)
        self.indicator.start()
        self.run_worker(lambda: self._turn(text), thread=True)

    async def _start_continue(self) -> None:
        # Same as _start_turn but the pending HumanMessage is already in history.
        self._busy = True
        self.streaming = None
        self.indicator = ThinkingIndicator()
        await self.transcript.append_live(self.indicator)
        self.indicator.start()
        self.run_worker(self._continue_run, thread=True)

    # --- one turn, runs in a worker thread (sync) ---

    def _stream(self) -> None:
        seen = len(self.history)
        last_values = None
        buffer = ""
        stream_ns = None
        last_post = 0.0

        for mode, data in self.graph.stream(
            {"messages": self.history},
            stream_mode=["messages", "values"],
            config={"recursion_limit": 100},
        ):
            if mode == "messages":
                chunk, cmeta = data
                content = getattr(chunk, "content", "")
                if not content:
                    continue
                ns = cmeta.get("langgraph_checkpoint_ns") if isinstance(cmeta, dict) else None
                if ns != stream_ns:
                    stream_ns = ns
                    buffer = ""
                buffer += content
                now = time.monotonic()
                if now - last_post >= _STREAM_THROTTLE_S:
                    last_post = now
                    self.post_message(_StreamChunk(buffer))
            else:  # values
                last_values = data
                for msg in data["messages"][seen:]:
                    if isinstance(msg, AIMessage) and msg.content:
                        self.post_message(_AgentContent(msg.content))
                seen = len(data["messages"])

        if last_values is not None:
            self.history = last_values["messages"]

    def _turn(self, text: str) -> None:
        from plugins.skills import match_skills  # local import avoids a cycle

        try:
            for skill in match_skills(text, self.skills):
                self.history.append(
                    SystemMessage(content=f"[Skill: {skill.name}]\n{skill.body}")
                )
            self.history.append(HumanMessage(content=text))
            self._stream()
        finally:
            self.post_message(_TurnDone())

    def _continue_run(self) -> None:
        # Pending messages are already in self.history; just run the graph.
        try:
            self._stream()
        finally:
            self.post_message(_TurnDone())
