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

import base64
import mimetypes
import time
import uuid
from collections import deque
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Input, Select, Static

import clyde.llm as llm
from clyde.llm.registry import PROVIDERS, supports_vision
from clyde.ui.filepicker import pick_image_native
from clyde.ui.renderer import (
    agent_renderable,
    spacer_renderable,
    user_renderable,
)
from clyde.ui.thinking import ThinkingIndicator
from clyde.ui.trace_full import TraceBlock, TracePanel
from clyde.ui.trace_minimal import minimal_trace_renderable
from clyde.ui.transcript import Transcript


# --- worker -> UI messages (posted from the turn thread, handled on the UI thread) ---

class _TraceLine(Message):
    def __init__(self, event) -> None:
        self.event = event
        super().__init__()


class _Ready(Message):
    """Posted once the graph + LLM + MCP stack has finished building."""


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


class _VisionProbed(Message):
    """Posted from a probe worker with the live vision-capability result."""

    def __init__(self, result: bool, key: tuple[str, str]) -> None:
        self.result = result
        self.key = key
        super().__init__()


class _FilePicked(Message):
    """Posted from the native-picker worker with the chosen path (or None)."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        super().__init__()


class _AskUser(Message):
    """Posted from the turn thread when the planner asks for clarification.

    Carries the clarify payload: a list of questions, each with its own options
    ({label, description, recommended}). The UI renders one option group per
    question and collects one pick per question.
    """

    def __init__(self, questions: list[dict]) -> None:
        self.questions = questions
        super().__init__()


class OptionButton(Static):
    """A clickable clarification option: label + dim description subtitle.

    `variant` is "rec" | "opt" | "chat" | "continue" | "confirm". `q_index` is
    the question this option belongs to (-1 for the global actions). `choice`
    is the string resumed into the graph when picked (the option label; empty
    for chat/continue/confirm).
    """

    def __init__(
        self, label: str, description: str, variant: str, choice: str, q_index: int = -1
    ) -> None:
        super().__init__("")
        self.label = label
        self.description = description
        self.variant = variant
        self.choice = choice
        self.q_index = q_index
        self.selected = False
        self._render_content()

    def _render_content(self) -> None:
        if self.variant in ("chat", "continue", "confirm"):
            head = Text(self.label)
        else:
            marker = "★ " if self.variant == "rec" else "  "
            sel = "✓ " if self.selected else ""
            head = Text(
                f"{sel}{marker}{self.label}",
                style="bold" if (self.variant == "rec" or self.selected) else "",
            )
            if self.description:
                head = Text.assemble(
                    head, "\n", Text(f"    {self.description}", style="dim")
                )
        self.update(head)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._render_content()
        if selected:
            self.add_class("sel")
        else:
            self.remove_class("sel")

    async def on_click(self, event) -> None:
        await self.app._on_option_clicked(self)


class TraceStatus(Static):
    """Clickable trace-mode status (none / minimal / full)."""

    def on_click(self, event) -> None:
        self.app.cycle_trace_mode()


class StatusSep(Static):
    """A muted '▸' separator between status groups (disabled-UI color)."""

    def __init__(self) -> None:
        super().__init__("▸", classes="sep")


class ModeStatus(Static):
    """Clickable agent-mode status (auto / plan / ask)."""

    def on_click(self, event) -> None:
        self.app.cycle_agent_mode()


class ModelStatus(Static):
    """Status showing the current model id; click to edit inline."""

    def on_click(self, event) -> None:
        self.app.enter_model_edit()


class AttachChip(Static):
    """Shows pending image attachments; click to clear them all."""

    def on_click(self, event) -> None:
        self.app.clear_attachments()


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
    align-horizontal: left;
}
#input-row {
    height: auto;
    padding: 0;
}
#attach {
    border: none;
    min-width: 3;
    width: 3;
    height: 1;
    padding: 0;
    margin: 0 1 0 0;
    color: white;
    background: transparent;
}
#attach:hover {
    text-style: bold;
}
#attach:disabled {
    color: $text-muted;
}
#attach-chip {
    height: 1;
    width: auto;
    padding: 0 1;
    margin: 0 0 0 1;
    color: black;
    background: white;
    border: none;
    text-style: none;
    display: none;
}
#input-row Input {
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
    width: 22;
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
OptionButton {
    height: auto;
    padding: 0 1;
    margin: 0 0 0 0;
    color: $text;
}
OptionButton:hover {
    background: $boost;
    text-style: bold;
}
OptionButton.rec {
    color: $accent;
    text-style: bold;
}
OptionButton.rec:hover {
    background: $boost;
}
OptionButton.chat {
    color: $text-muted;
}
OptionButton.continue {
    color: $accent;
    text-style: bold;
}
OptionButton.continue:disabled {
    color: $text-disabled;
    text-style: none;
}
OptionButton.confirm {
    color: $accent;
    text-style: bold;
}
OptionButton.sel {
    background: $boost;
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
        Binding("shift+tab", "cycle_mode", "Cycle mode", priority=True),
        Binding("escape", "cancel_model_edit", "Cancel", priority=True),
    ]

    def __init__(self, builder, session_id=None, resume_messages=None):
        super().__init__()
        self.builder = builder
        # Session persistence. On a fresh launch session_id is None and is
        # created once the build finishes (on__ready); on `clyde last session`
        # session_id + resume_messages are passed in and the transcript is
        # replayed instead of greeting.
        self.session_id = session_id
        self._resume_msgs = list(resume_messages) if resume_messages else None
        # Heavy state — built in a background worker after the TUI paints so the
        # app appears instantly instead of blocking on the langchain/langgraph
        # import + graph compile (~1.2s). None until _Ready.
        self.graph = None
        self.history = None
        self.skills = None
        self.manager = None
        self._built = False
        self._warmup: Static | None = None
        self._queued: dict | None = None  # message submitted before ready
        self._trace_panel: TracePanel | None = None  # active per-turn panel (full mode)
        self._busy = False
        self._pending: list = []  # mid-turn messages, processed as a continuation
        self._pending_model: tuple | None = None  # (provider, model_id) applied after a turn
        self._turn_config: dict | None = None  # {"configurable": {"thread_id": ...}, ...}
        self._awaiting_choice = False  # True while a clarify prompt is on screen
        self._ask_groups: list[list[OptionButton]] = []  # option buttons per question
        self._ask_selections: list[OptionButton | None] = []  # chosen button per question
        self._ask_questions: list[dict] = []  # the clarify questions being answered
        self._continue_btn: OptionButton | None = None  # confirm button
        self.trace_mode = "none"  # "none" | "minimal" | "full"
        self.agent_mode = "auto"  # "auto" | "plan" | "ask" (UI state only)
        self._edit_model = False  # inline model-edit mode in the input bar
        self.indicator: ThinkingIndicator | None = None
        self.streaming: Static | None = None
        # Image attachment state — only meaningful when the current model is
        # vision-capable (see self._vision).
        self._image_parts: list[dict] = []   # pending image_url content parts
        self._image_names: list[str] = []    # filenames for display only
        self._vision: bool = False           # recomputed on mount + model switch
        self._vision_probing: set[tuple[str, str]] = set()  # in-flight probe keys

    # --- layout ---

    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")
        with Container(id="input-bar"):
            with Horizontal(id="input-row"):
                yield Input(id="input", placeholder="Message…")
                yield Select(
                    [(p, p) for p in PROVIDERS],
                    id="provider",
                    classes="model-edit",
                )
                yield Input(id="model-id", classes="model-edit")
                yield Button("Confirm", id="confirm", classes="model-edit")
                yield Button("✕", id="close", classes="model-edit")
            yield AttachChip("", id="attach-chip")
        with Horizontal(id="status-bar"):
            yield Button("+", id="attach")
            yield TraceStatus(id="trace-status")
            yield StatusSep()
            yield ModeStatus(id="mode-status")
            yield StatusSep()
            yield ModelStatus(id="model-status")
            yield Static("ctrl-c to exit", id="exit-hint")  # rotates: see _rotate_hint

    async def on_mount(self) -> None:
        self.transcript = self.query_one("#transcript", Transcript)
        self.input = self.query_one("#input", Input)
        self.provider_select = self.query_one("#provider", Select)
        self.model_id_input = self.query_one("#model-id", Input)
        self.trace_status = self.query_one("#trace-status", TraceStatus)
        self.mode_status = self.query_one("#mode-status", ModeStatus)
        self.model_status = self.query_one("#model-status", ModelStatus)
        self.exit_hint = self.query_one("#exit-hint", Static)
        self.update_trace_status()
        self.update_mode_status()
        # Model label + vision gating need llm.CURRENT_*, which pulls in the
        # langchain stack — defer to on__ready so the TUI paints first.
        self.model_status.update("…")
        self._apply_edit_visibility()
        self._refresh_attach()
        # "Warming up" line shown until the background build finishes.
        self._warmup = Static(Text("Clyde warming up…", style="dim"), classes="tline")
        await self.transcript.append_live(self._warmup)
        # Build the graph + LLM + MCP manager off the UI thread, then post _Ready.
        self.run_worker(self._build_run, thread=True)
        self.input.focus()
        # Rotate the bottom-right hint between keybindings so each gets a turn.
        self._hint_index = 0
        self._hint_timer = self.set_interval(8, self._rotate_hint)

    def _build_run(self) -> None:
        """Background worker: run the heavy builder, then signal readiness."""
        try:
            graph, history, skills, manager = self.builder()
        except Exception:
            # If the build blows up, surface it and leave the app usable for exit.
            self.post_message(_AgentContent("Clyde failed to start. Try `clyde login`."))
            self.post_message(_TurnDone())
            return
        self.graph = graph
        self.history = history
        self.skills = skills
        self.manager = manager
        self._built = True
        self.post_message(_Ready())

    async def on__ready(self, message: _Ready) -> None:
        # Drop the warming-up line.
        if self._warmup is not None:
            self._warmup.remove()
            self._warmup = None
        # Now that langchain/langgraph are loaded, set the model label + vision.
        self._vision = supports_vision(llm.CURRENT_PROVIDER, llm.CURRENT_MODEL_ID)
        self._refresh_attach()
        self.maybe_probe_vision(llm.CURRENT_PROVIDER, llm.CURRENT_MODEL_ID)
        self.update_model_status()
        # If a message was typed during warmup, fold it into the pending queue
        # (built into a HumanMessage now that langchain is loaded). It's sent
        # after the greeting on a fresh launch, or right after replay on resume.
        if self._queued is not None:
            from langchain_core.messages import HumanMessage, SystemMessage

            from clyde.plugins.skills import match_skills  # local import avoids a cycle

            item = self._queued
            self._queued = None
            parts = item.pop("parts")
            if parts:
                content: list[dict] = []
                if item["text"]:
                    content.append({"type": "text", "text": item["text"]})
                content.extend(parts)
                item["msg"] = HumanMessage(content=content)
            else:
                item["msg"] = HumanMessage(content=item["text"])
            item["skills"] = [
                SystemMessage(content=f"[Skill: {s.name}]\n{s.body}")
                for s in match_skills(item["text"], self.skills)
            ]
            self._pending.append(item)
        # Resume replays the saved transcript and skips the greeting; a fresh
        # launch creates a session on disk and greets.
        if self._resume_msgs is not None:
            await self._replay_session()
            await self._drain_pending()
        else:
            if self.session_id is None:
                import os

                from clyde.sessions import create_session

                self.session_id = create_session(
                    os.getcwd(), llm.CURRENT_PROVIDER, llm.CURRENT_MODEL_ID
                )
            await self._start_greet()

    def shutdown_manager(self) -> None:
        """Stop the MCP manager if it was started (called from __main__ finally)."""
        if self.manager is not None:
            self.manager.shutdown()

    _HINTS = ("ctrl-c to exit", "shift-tab to cycle mode")

    def _rotate_hint(self) -> None:
        self._hint_index = (self._hint_index + 1) % len(self._HINTS)
        self.exit_hint.update(self._HINTS[self._hint_index])

    # --- trace mode (none / minimal / full) ---

    _TRACE_MODES = ("none", "minimal", "full")
    _TRACE_LABELS = {"none": "no trace", "minimal": "minimal trace", "full": "full trace"}

    def cycle_trace_mode(self) -> None:
        i = self._TRACE_MODES.index(self.trace_mode) if self.trace_mode in self._TRACE_MODES else 0
        self.trace_mode = self._TRACE_MODES[(i + 1) % len(self._TRACE_MODES)]
        self.update_trace_status()
        self.notify(self._TRACE_LABELS[self.trace_mode], timeout=2)

    def update_trace_status(self) -> None:
        self.trace_status.update(self._TRACE_LABELS[self.trace_mode])

    def action_toggle_trace(self) -> None:
        self.cycle_trace_mode()

    def action_cycle_mode(self) -> None:
        self.cycle_agent_mode()

    # --- agent mode (auto / plan / ask) — UI state only, not wired to the planner ---

    _AGENT_MODES = ("auto", "plan", "ask")

    def cycle_agent_mode(self) -> None:
        i = self._AGENT_MODES.index(self.agent_mode) if self.agent_mode in self._AGENT_MODES else 0
        self.agent_mode = self._AGENT_MODES[(i + 1) % len(self._AGENT_MODES)]
        self.update_mode_status()
        self.notify(f"{self.agent_mode} mode", timeout=2)

    def update_mode_status(self) -> None:
        self.mode_status.update(f"{self.agent_mode} mode")

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
        self._refresh_attach()

    def action_cancel_model_edit(self) -> None:
        if self._edit_model:
            self.exit_model_edit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "attach":
            # Open the native macOS file picker; only when idle and vision-capable.
            # Runs off the UI thread so the Textual loop doesn't block on the dialog.
            if self._busy or not self._vision:
                return
            self.run_worker(self._pick_image_worker, thread=True)
            return
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
        self.model_status.update(llm.CURRENT_MODEL_ID.split('/')[-1])

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
            from clyde.llm import set_model
            from clyde.tools import tools as _static_tools
            import clyde.plugins.mcp as _mcpmod
            import clyde.agents.coding_agent.model as _model
            import clyde.agents.coding_agent.supervisor.nodes as _sup
            import clyde.agents.coding_agent.worker.nodes as _worker
            import clyde.tools.summarize as _sum

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
            # Recompute vision capability for the new model and gate the + button.
            self._vision = supports_vision(provider, model_id)
            if not self._vision:
                self.clear_attachments()
            self._refresh_attach()
            # Unknown to the pattern list? Probe in the background.
            self.maybe_probe_vision(provider, model_id)
            self.update_model_status()
            self.notify(f"model: {model_id}", timeout=3)
        except Exception as e:
            self.notify(f"model switch failed: {e}", timeout=5)

    # --- message handlers (UI thread; async so we can mount into the transcript) ---

    async def on__trace_line(self, message: _TraceLine) -> None:
        event = message.event
        header = event.header
        if self.indicator is not None and header.startswith("[LLM] end"):
            # Accumulate output tokens: "[LLM] end — 3.12s · ↑ 1234 in · ↓ 294 out · …"
            import re

            m = re.search(r"↓\s*(\d+)\s*out", header)
            if m:
                self.indicator.add_tokens(int(m.group(1)))
        if self.trace_mode == "full":
            if self._trace_panel is not None:
                await self._trace_panel.add_event(event)
            else:
                # No active panel (e.g. greeting): fall back to a standalone block.
                await self.transcript.append_live(TraceBlock(header, event.body))
        elif self.trace_mode == "minimal":
            await self.transcript.append(minimal_trace_renderable(header))
        # "none" -> no inline trace line

    async def on__stream_chunk(self, message: _StreamChunk) -> None:
        # The answer always lives in the transcript (after the trace panel in full
        # mode), not inside the panel.
        if self.streaming is None:
            self.streaming = Static("")
            await self.transcript.append_live(self.streaming)
        self.streaming.update(message.text)

    async def on__agent_content(self, message: _AgentContent) -> None:
        # Freeze the streaming widget into the rendered Markdown answer. When a
        # trace panel is active the trailing gap is added in on__turn_done (after
        # the panel + answer); otherwise it's added here.
        if self.streaming is None:
            await self.transcript.append(agent_renderable(message.text))
        else:
            self.streaming.update(agent_renderable(message.text))
            self.streaming = None
        if self._trace_panel is None:
            await self.transcript.append(spacer_renderable())

    async def on__turn_done(self, message: _TurnDone) -> None:
        # Settle the indicator inline to "Thought for Xs" and leave it there.
        if self.indicator is not None:
            self.indicator.done()
            self.indicator = None
        # Close the per-turn trace panel (full mode): separate it from the next
        # turn with a trailing gap, then drop the active-panel reference so later
        # events (e.g. greeting) don't route into a completed turn's panel.
        if self._trace_panel is not None:
            await self.transcript.append(spacer_renderable())
            self._trace_panel = None
        # No spacer here otherwise: each conversation/live message (user,
        # indicator, answer) owns its own trailing gap at render time.
        # Apply a deferred model switch now that the turn is done.
        if self._pending_model is not None:
            provider, model_id = self._pending_model
            self._pending_model = None
            self._apply_model(provider, model_id)
            return
        # Continue with any messages queued while Clyde was busy (renders them
        # in order after the prior answer, then runs the graph again).
        if await self._drain_pending():
            return
        self._busy = False
        self._refresh_attach()
        self.persist()

    async def _drain_pending(self) -> bool:
        """Render + enqueue any messages queued while Clyde was busy.

        Renders each queued user line in order (with a gap) so they land after
        the prior answer instead of interleaving with its stream, appends them
        to history, then starts a continuation turn. Returns True if a turn was
        started, False if the queue was empty (caller should then go idle).
        """
        if not self._pending:
            return False
        for item in self._pending:
            for s in item["skills"]:
                self.history.append(s)
            self.history.append(item["msg"])
            await self.transcript.append(
                user_renderable(item["text"], item["images"])
            )
            await self.transcript.append(spacer_renderable())
        self._pending = []
        await self._start_continue()
        return True

    async def _replay_session(self) -> None:
        """Replay a resumed session's messages into the transcript (no greeting).

        Rebuilds history as a fresh system prompt + the saved messages, then
        renders each user/assistant turn in order. System/tool messages are
        skipped (they're context, not things the user saw on screen).
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from clyde.agents import default_system_prompt

        msgs = self._resume_msgs or []
        self.history = [SystemMessage(content=default_system_prompt)] + list(msgs)
        n = len(msgs)
        await self.transcript.append(
            Static(Text(f"Resumed session · {n} messages", style="dim"), classes="tline")
        )
        await self.transcript.append(spacer_renderable())
        for m in msgs:
            if isinstance(m, HumanMessage):
                text, images = self._human_display(m)
                await self.transcript.append(user_renderable(text, images))
                await self.transcript.append(spacer_renderable())
            elif isinstance(m, AIMessage):
                content = m.content
                text = content if isinstance(content, str) else self._text_from_parts(content)
                await self.transcript.append(agent_renderable(text))
                await self.transcript.append(spacer_renderable())
            # SystemMessage / ToolMessage: context only — not replayed on screen.
        self._resume_msgs = None
        self.input.focus()

    @staticmethod
    def _human_display(m) -> tuple[str, list[str]]:
        """Extract (text, image-names) from a HumanMessage for display.

        Names aren't persisted, so attached images are shown as generic
        `🖼 imageN` markers rather than their original filenames.
        """
        c = m.content
        if isinstance(c, str):
            return c, []
        if isinstance(c, list):
            texts: list[str] = []
            imgs = 0
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    texts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    imgs += 1
            return " ".join(t for t in texts if t), [f"image{i + 1}" for i in range(imgs)]
        return str(c), []

    @staticmethod
    def _text_from_parts(content) -> str:
        """Flatten a multimodal content list to its text parts."""
        if not isinstance(content, list):
            return str(content)
        return " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )

    def persist(self) -> None:
        """Flush the current history to the session file (best-effort).

        Called after each settled turn and once on exit. The base system prompt
        is filtered out by the sessions package so it isn't duplicated on reload.
        """
        if self.session_id is None or self.history is None:
            return
        try:
            from clyde.agents import default_system_prompt
            from clyde.sessions import save_messages

            save_messages(self.session_id, self.history, default_system_prompt)
        except Exception:
            pass  # persistence is a convenience, never fatal

    # --- clarify / ask-the-user (interrupt-based) ---

    async def on__ask_user(self, message: _AskUser) -> None:
        """Render each clarifying question + its options, then a Continue button.

        The turn is paused at the graph's interrupt(); self._busy stays True so
        any input typed while waiting queues (processed after the resumed turn).
        """
        # Stop the thinking spinner while we wait for picks.
        if self.indicator is not None:
            self.indicator.hide()
            self.indicator = None

        questions = list(message.questions or [])
        self._ask_questions = questions
        self._ask_groups = []
        self._ask_selections = [None] * len(questions)

        for qi, q in enumerate(questions):
            await self.transcript.append(
                Static(Text(f"❓ {q.get('question', '')}", style="bold yellow"))
            )
            await self.transcript.append(spacer_renderable())
            opts = list(q.get("options") or [])
            opts.sort(key=lambda o: 0 if o.get("recommended") else 1)
            group: list[OptionButton] = []
            for o in opts:
                variant = "rec" if o.get("recommended") else "opt"
                btn = OptionButton(
                    o.get("label", ""),
                    o.get("description", ""),
                    variant,
                    o.get("label", ""),
                    q_index=qi,
                )
                btn.add_class(variant)
                await self.transcript.append_live(btn)
                group.append(btn)
            await self.transcript.append(spacer_renderable())
            self._ask_groups.append(group)

        # Continue: confirm all picks and resume. Disabled until every question
        # has a selection.
        cont = OptionButton("Continue", "", "continue", "")
        cont.add_class("continue")
        cont.disabled = True
        await self.transcript.append_live(cont)
        self._continue_btn = cont
        # Chat about this: abandon, return to free input.
        chat = OptionButton("Chat about this", "", "chat", "")
        chat.add_class("chat")
        await self.transcript.append_live(chat)
        await self.transcript.append(spacer_renderable())
        self._awaiting_choice = True

    def _refresh_continue(self) -> None:
        """Enable Continue only when every question has a selection."""
        if self._continue_btn is None:
            return
        self._continue_btn.disabled = not all(
            sel is not None for sel in self._ask_selections
        )

    async def _show_review(self) -> None:
        """Render a review screen: each question (white) + chosen answer (green),
        then a Confirm button. Confirm resumes the paused graph with all picks.
        """
        for qi, sel in enumerate(self._ask_selections):
            q = self._ask_questions[qi].get("question", "") if qi < len(self._ask_questions) else ""
            answer = sel.choice if sel is not None else "(none)"
            await self.transcript.append(
                Static(Text(q, style="bold white"))
            )
            await self.transcript.append(Static(Text(answer, style="green")))
            await self.transcript.append(spacer_renderable())
        confirm = OptionButton("Confirm", "", "confirm", "")
        confirm.add_class("confirm")
        await self.transcript.append_live(confirm)
        await self.transcript.append(spacer_renderable())

    async def _on_option_clicked(self, button: OptionButton) -> None:
        """Called from OptionButton.on_click — route the pick."""
        if not self._awaiting_choice:
            return
        if button.variant == "chat":
            self._awaiting_choice = False
            self._abandon_ask(notify="type your question")
            return
        if button.variant == "continue":
            if button.disabled:
                return
            await self._show_review()
            return
        if button.variant == "confirm":
            self._awaiting_choice = False
            choices = [
                sel.choice if sel is not None else "" for sel in self._ask_selections
            ]
            await self._resume_turn(choices)
            return
        # An option within a question group: record the selection.
        qi = button.q_index
        prev = self._ask_selections[qi]
        if prev is not None:
            prev.set_selected(False)
        button.set_selected(True)
        self._ask_selections[qi] = button
        self._refresh_continue()

    def _abandon_ask(self, notify: str) -> None:
        """Drop the paused turn (no resume); orphaned checkpoint is harmless."""
        self._turn_config = None
        self._busy = False
        self._awaiting_choice = False
        self._ask_groups = []
        self._ask_selections = []
        self._ask_questions = []
        self._continue_btn = None
        # Drop input typed while the ask was on screen — it was meant for a turn
        # we're now abandoning, and leaving it would leak into a later turn.
        self._pending = []
        self._refresh_attach()
        self.input.focus()
        self.notify(notify, timeout=3)

    async def _resume_turn(self, choices: list[str]) -> None:
        """Continue the paused graph with the user's picks (one per question)."""
        self._busy = True
        self._refresh_attach()
        self.streaming = None
        self._ask_groups = []
        self._ask_selections = []
        self._ask_questions = []
        self._continue_btn = None
        # Reuse the same thread_id config so the graph resumes from checkpoint.
        self.indicator = ThinkingIndicator()
        await self.transcript.append_live(self.indicator)
        self.indicator.start()
        await self.transcript.append(spacer_renderable())
        self.run_worker(lambda: self._resume_run(choices), thread=True)

    def _resume_run(self, choices: list[str]) -> None:
        interrupted = False
        try:
            interrupted = self._stream(resume=choices)
        finally:
            if not interrupted:
                self.post_message(_TurnDone())
            self._refresh_attach()

    # --- trace sink target (called from main.py on the worker/MCP threads) ---

    def post_trace(self, event) -> None:
        self.post_message(_TraceLine(event))

    # --- image attachment handling ---

    def _refresh_attach(self) -> None:
        """Show/hide the + button: only when vision-capable and not editing."""
        try:
            btn = self.query_one("#attach", Button)
        except Exception:
            return
        show = self._vision and not self._edit_model
        btn.styles.display = "block" if show else "none"
        btn.disabled = self._busy or not self._vision

    def _render_chip(self) -> None:
        """Render the pending-attachments chip (hidden when empty)."""
        chip = self.query_one("#attach-chip", AttachChip)
        if self._image_names:
            chip.update("  " + "  ".join(f"🖼 {n} ✕" for n in self._image_names))
            chip.styles.display = "block"
        else:
            chip.update("")
            chip.styles.display = "none"

    def clear_attachments(self) -> None:
        """Drop all pending image attachments (chip click)."""
        self._image_parts = []
        self._image_names = []
        self._render_chip()

    def maybe_probe_vision(self, provider: str, model_id: str) -> None:
        """If vision capability is unknown, ask the LLM in a worker thread.

        Pattern match + cache are instant; this only fires for models neither
        resolves (e.g. a freshly-typed id like muse-glimmer-30b). The probe runs
        once per model — an in-flight guard prevents duplicate probes during
        rapid model switching.
        """
        from clyde.llm.registry import needs_vision_probe

        key = ((provider or "").lower(), (model_id or "").lower())
        if not needs_vision_probe(provider, model_id) or key in self._vision_probing:
            return
        self._vision_probing.add(key)
        self.run_worker(lambda: self._probe_vision_run(provider, model_id, key), thread=True)

    def _probe_vision_run(self, provider: str, model_id: str, key: tuple[str, str]) -> None:
        from clyde.llm.registry import probe_vision

        try:
            result = probe_vision(provider, model_id)
        except Exception:
            result = False
        self.post_message(_VisionProbed(result, key))

    async def on__vision_probed(self, message: _VisionProbed) -> None:
        self._vision_probing.discard(message.key)
        # Only adopt the result if it's still for the current model.
        if ((llm.CURRENT_PROVIDER or "").lower(), (llm.CURRENT_MODEL_ID or "").lower()) != message.key:
            return
        self._vision = message.result
        if not message.result:
            self.clear_attachments()
        self._refresh_attach()
        if message.result:
            self.notify("model supports images — + enabled", timeout=3)

    def _pick_image_worker(self) -> None:
        """Worker thread: open the native macOS picker and post the path back."""
        path = pick_image_native()
        self.post_message(_FilePicked(path))

    async def on__file_picked(self, message: _FilePicked) -> None:
        """Encode the picked image and attach it to the pending message."""
        path = message.path
        if not path:
            return  # cancelled
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as e:
            self.notify(f"couldn't read {p.name}: {e}", timeout=4)
            return
        mime, _ = mimetypes.guess_type(str(p))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        self._image_parts.append({"type": "image_url", "image_url": {"url": url}})
        self._image_names.append(p.name)
        self._render_chip()
        self.input.focus()

    def _build_human_message(self, text: str):
        """Build the user message, attaching any pending images as multimodal parts.

        Images are consumed here — the chip clears once they're folded into the
        message that enters the graph.
        """
        from langchain_core.messages import HumanMessage

        if not self._image_parts:
            return HumanMessage(content=text)
        parts: list[dict] = []
        if text:
            parts.append({"type": "text", "text": text})
        parts.extend(self._image_parts)
        self._image_parts = []
        self._image_names = []
        self._render_chip()
        return HumanMessage(content=parts)

    # --- input handling ---

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text and not self._image_parts:
            return
        if text.lower() in ("exit", "quit"):
            self.exit()
            return
        # If the model lost vision between attach and submit, drop the images.
        if self._image_parts and not self._vision:
            self.notify("current model can't see images — attach cleared", timeout=4)
            self.clear_attachments()
        # Still warming up (graph + LLM building in the background): stash the
        # raw text + image parts and send once _Ready fires. Don't build the
        # HumanMessage here — that would pull langchain_core into the UI thread.
        if not self._built:
            self.input.value = ""
            images = list(self._image_names)
            parts = list(self._image_parts)
            self._image_parts = []
            self._image_names = []
            self._render_chip()
            self._queued = {"text": text, "images": images, "parts": parts}
            self.notify("warming up — your message will send when ready", timeout=3)
            return
        self.input.value = ""
        images = list(self._image_names)
        msg = self._build_human_message(text)
        if self._busy:
            # Clyde is working (incl. the greeting): queue silently, don't render
            # yet. Rendered + processed only after the current action finishes, so
            # the transcript order stays clean (no interleaving with the stream).
            from langchain_core.messages import SystemMessage

            from clyde.plugins.skills import match_skills  # local import avoids a cycle

            skills_msgs = [
                SystemMessage(content=f"[Skill: {s.name}]\n{s.body}")
                for s in match_skills(text, self.skills)
            ]
            self._pending.append(
                {"text": text, "images": images, "msg": msg, "skills": skills_msgs}
            )
        else:
            await self.transcript.append(user_renderable(text, images))
            await self.transcript.append(spacer_renderable())
            await self._start_turn(text, msg)

    def _new_turn_config(self) -> dict:
        """Fresh thread_id per turn so each request starts a clean checkpoint.

        A resume after a clarify reuses this same config (same thread_id) so the
        graph continues from the paused checkpoint instead of restarting.
        """
        self._turn_config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 100,
        }
        return self._turn_config

    async def _start_turn(self, text: str, msg: HumanMessage) -> None:
        self._busy = True
        self._refresh_attach()
        self.streaming = None
        self._new_turn_config()
        self.indicator = ThinkingIndicator()
        await self.transcript.append_live(self.indicator)
        self.indicator.start()
        # One-line gap after the live thinking indicator.
        await self.transcript.append(spacer_renderable())
        # Full mode: one bordered panel collects the whole trace + the answer
        # for this turn (height-capped, scrolls internally).
        if self.trace_mode == "full":
            self._trace_panel = TracePanel()
            await self.transcript.append_live(self._trace_panel)
        self.run_worker(lambda: self._turn(text, msg), thread=True)

    async def _start_continue(self) -> None:
        # Same as _start_turn but the pending HumanMessage is already in history.
        self._busy = True
        self._refresh_attach()
        self.streaming = None
        self._new_turn_config()
        self.indicator = ThinkingIndicator()
        await self.transcript.append_live(self.indicator)
        self.indicator.start()
        # One-line gap after the live thinking indicator.
        await self.transcript.append(spacer_renderable())
        # Full mode: one bordered panel collects the whole trace + the answer
        # for this turn (height-capped, scrolls internally).
        if self.trace_mode == "full":
            self._trace_panel = TracePanel()
            await self.transcript.append_live(self._trace_panel)
        self.run_worker(self._continue_run, thread=True)

    async def _start_greet(self) -> None:
        """Proactive greeting at session start — streamed, context-aware, non-blocking."""
        self._busy = True
        self._refresh_attach()
        self.streaming = None
        self.indicator = ThinkingIndicator()
        await self.transcript.append_live(self.indicator)
        self.indicator.start()
        # One-line gap after the live thinking indicator.
        await self.transcript.append(spacer_renderable())
        self.run_worker(self._greet_run, thread=True)

    def _greet_run(self) -> None:
        """Run the greeting graph and stream it into the transcript.

        Not part of conversation_history (display only). Falls back to a
        static greet if the LLM call fails so a session never hangs on greet.
        """
        from clyde.agents.coding_agent.greeting import build_greet_graph
        from langchain_core.messages import AIMessage

        try:
            greet_graph = build_greet_graph()
            seen = 0
            last_values = None
            buffer = ""
            stream_ns = None
            last_post = 0.0
            for mode, data in greet_graph.stream(
                {"messages": []},
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
        except Exception:
            # Fallback: a static greet so the session still starts cleanly.
            self.post_message(
                _AgentContent("Clyde ready. What are we working on?")
            )
        finally:
            self.post_message(_TurnDone())

    # --- one turn, runs in a worker thread (sync) ---

    def _stream(self, resume: str | None = None) -> bool:
        """Stream one graph execution. Returns True if it paused on a clarify interrupt.

        `resume` is the user's chosen option label when continuing from a paused
        checkpoint; omit it for a fresh turn. On interrupt, posts `_AskUser` and
        returns True without posting `_TurnDone` (the turn is paused, not done).
        """
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        inputs = Command(resume=resume) if resume is not None else {"messages": self.history}
        seen = len(self.history)
        last_values = None
        buffer = ""
        stream_ns = None
        last_post = 0.0

        for mode, data in self.graph.stream(
            inputs,
            stream_mode=["messages", "values", "updates"],
            config=self._turn_config,
        ):
            if mode == "updates":
                # Interrupts surface here as {"__interrupt__": [Interrupt(...)]}.
                if isinstance(data, dict) and "__interrupt__" in data:
                    intr = data["__interrupt__"][0]
                    val = getattr(intr, "value", None) or []
                    self.post_message(_AskUser(val))
                    if last_values is not None:
                        self.history = last_values["messages"]
                    return True
                continue
            elif mode == "messages":
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
        return False

    def _turn(self, text: str, msg: HumanMessage) -> None:
        from langchain_core.messages import SystemMessage

        from clyde.plugins.skills import match_skills  # local import avoids a cycle

        interrupted = False
        try:
            for skill in match_skills(text, self.skills):
                self.history.append(
                    SystemMessage(content=f"[Skill: {skill.name}]\n{skill.body}")
                )
            self.history.append(msg)
            interrupted = self._stream()
        finally:
            if not interrupted:
                self.post_message(_TurnDone())

    def _continue_run(self) -> None:
        # Pending messages are already in self.history; just run the graph.
        interrupted = False
        try:
            interrupted = self._stream()
        finally:
            if not interrupted:
                self.post_message(_TurnDone())
