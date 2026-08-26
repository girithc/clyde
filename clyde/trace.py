"""Compact structured tracing.

Replaces LangChain's default verbose/debug JSON dumps with one clean line per
event. Covers LLM calls and tool runs; the graph node flow is surfaced by the
UI.

Each event is emitted as a ``TraceEvent`` carrying a one-line ``header`` (the
summary minimal mode shows) and an optional ``body`` (full content for full
mode: the LLM's input messages, the full response text, or the untruncated tool
output). Output goes through a configurable ``sink`` (default: ``print``). The
TUI sets the sink to route events into the scrolling transcript — as a compact
label in minimal mode, or as a height-capped scrollable block in full mode —
instead of `print` corrupting the full-screen app.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class TraceEvent:
    """A single trace event: a one-line header + optional full-content body.

    ``header`` is what minimal mode renders (and what the thinking indicator
    parses for output tokens). ``body`` is the full detail full mode renders
    inside a scrollable block (None or empty → no body, header only).
    """

    header: str
    body: str | None = None


class CompactTraceHandler(BaseCallbackHandler):
    """One-line-per-event trace printer with full-content payloads."""

    def __init__(self):
        self._llm_starts: dict[str, float] = {}
        self._tool_starts: dict[str, float] = {}
        self._sink: Callable[[TraceEvent], Any] = lambda e: print(e.header)

    def set_sink(self, sink: Callable[[TraceEvent], Any]) -> None:
        """Redirect trace events (default ``print``; the TUI sets its own)."""
        self._sink = sink

    def _emit(self, header: str, body: str | None = None) -> None:
        self._sink(TraceEvent(header, body))

    # --- LLM ---

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_starts[str(run_id)] = time.time()
        n = len(prompts[0].splitlines()) if prompts else 0
        body = "\n\n".join(p for p in (prompts or []) if p) or None
        self._emit(f"[LLM] start — {n} message-lines in context", body)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        # Chat models fire this instead of on_llm_start; implementing it captures
        # the real BaseMessage objects so full mode can show every input message.
        self._llm_starts[str(run_id)] = time.time()
        convo = messages[0] if messages else []
        n = len(convo)
        lines = [f"{type(m).__name__}: {m.content}" for m in convo]
        body = "\n".join(lines) or None
        self._emit(f"[LLM] start — {n} message-lines in context", body)

    def on_llm_end(self, response, *, run_id, **kwargs):
        started = self._llm_starts.pop(str(run_id), None)
        secs = f"{time.time() - started:.2f}s" if started else "?s"
        usage = (response.llm_output or {}).get("token_usage", {}) or {}
        in_toks = usage.get("prompt_tokens") or usage.get("input_tokens")
        out_toks = usage.get("completion_tokens") or usage.get("output_tokens")
        # Streaming responses usually leave llm_output.token_usage empty; the
        # counts arrive on the final AIMessage instead (usage_metadata for
        # langchain-core >=0.2, response_metadata.token_usage otherwise).
        if in_toks is None or out_toks is None:
            try:
                msg = response.generations[0][0].message
            except (IndexError, AttributeError):
                msg = None
            if msg is not None:
                um = getattr(msg, "usage_metadata", None) or {}
                if in_toks is None:
                    in_toks = um.get("input_tokens")
                if out_toks is None:
                    out_toks = um.get("output_tokens")
                if in_toks is None or out_toks is None:
                    rm = getattr(msg, "response_metadata", None) or {}
                    tu = rm.get("token_usage") or rm.get("usage") or {}
                    if in_toks is None:
                        in_toks = tu.get("prompt_tokens") or tu.get("input_tokens")
                    if out_toks is None:
                        out_toks = tu.get("completion_tokens") or tu.get("output_tokens")
        in_toks = "?" if in_toks is None else in_toks
        out_toks = "?" if out_toks is None else out_toks
        finish = None
        try:
            gen = response.generations[0][0]
            finish = gen.generation_info.get("finish_reason") if gen else None
        except (IndexError, AttributeError):
            pass
        body = None
        try:
            body = response.generations[0][0].text or None
        except (IndexError, AttributeError):
            pass
        self._emit(
            f"[LLM] end — {secs} · ↑ {in_toks} in · ↓ {out_toks} out · finish={finish}",
            body,
        )

    # --- Tools ---

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._tool_starts[str(run_id)] = time.time()
        name = (serialized or {}).get("name", "tool")
        self._emit(f"[Tool] start — {name}", input_str or None)

    def on_tool_end(self, output, *, run_id, **kwargs):
        started = self._tool_starts.pop(str(run_id), None)
        secs = f"{(time.time() - started) * 1000:.0f}ms" if started else "?ms"
        body = str(output).strip() or None
        self._emit(f"[Tool] end — {secs}", body)


# Shared singleton — attach on the LLM in llm/factory.py via callbacks=[...].
compact_trace = CompactTraceHandler()
