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

import re
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

    ``ns`` is the LangGraph checkpoint namespace the event came from (e.g.
    ``"planner:<uuid>"``, ``"worker:<uuid>"``); the UI groups events into tabs
    by this. Non-worker namespaces collapse to the ``"main"`` tab. ``label`` is a
    human-friendly tab title for worker namespaces (the task description), if
    known when the event is emitted.
    """

    header: str
    body: str | None = None
    ns: str = ""
    label: str | None = None


class CompactTraceHandler(BaseCallbackHandler):
    """One-line-per-event trace printer with full-content payloads."""

    def __init__(self):
        self._llm_starts: dict[str, float] = {}
        self._tool_starts: dict[str, float] = {}
        # run_id -> checkpoint namespace, learned from start events (on_llm_end /
        # on_tool_end metadata lacks the ns, so they look it up here).
        self._run_ns: dict[str, str] = {}
        # worker namespace -> tab label, parsed once from the first task_planner call
        self._ns_label: dict[str, str] = {}
        self._sink: Callable[[TraceEvent], Any] = lambda e: print(e.header)

    def set_sink(self, sink: Callable[[TraceEvent], Any]) -> None:
        """Redirect trace events (default ``print``; the TUI sets its own)."""
        self._sink = sink

    def _ns(self, run_id, kwargs) -> str:
        """Resolve the checkpoint namespace for an event.

        Start events carry it in their metadata; end events don't, so they reuse
        the ns recorded at their run's start.
        """
        metadata = (kwargs.get("metadata") or {}) if kwargs else {}
        ns = metadata.get("langgraph_checkpoint_ns")
        if ns:
            self._run_ns[str(run_id)] = ns
            return ns
        return self._run_ns.get(str(run_id), "")

    def _label_for(self, ns: str, messages) -> str | None:
        """First time we see a worker namespace, derive a tab label from the
        task_planner prompt (``Task: {description}``). Subsequent calls return
        the cached label."""
        if not ns.startswith("worker:") or ns in self._ns_label:
            return self._ns_label.get(ns)
        label = None
        try:
            content = getattr(messages[0][0], "content", "") if messages else ""
            m = re.search(r"Task:\s*(.+)", str(content))
            if m:
                label = m.group(1).strip().splitlines()[0][:24]
        except Exception:
            pass
        self._ns_label[ns] = label or ns
        return self._ns_label[ns]

    def _emit(self, header, body, ns="", label=None) -> None:
        self._sink(TraceEvent(header, body, ns, label))

    # --- LLM ---

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_starts[str(run_id)] = time.time()
        ns = self._ns(run_id, kwargs)
        n = len(prompts[0].splitlines()) if prompts else 0
        body = "\n\n".join(p for p in (prompts or []) if p) or None
        self._emit(f"[LLM] start — {n} message-lines in context", body, ns)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        # Chat models fire this instead of on_llm_start; implementing it captures
        # the real BaseMessage objects so full mode can show every input message.
        self._llm_starts[str(run_id)] = time.time()
        ns = self._ns(run_id, kwargs)
        label = self._label_for(ns, messages)
        convo = messages[0] if messages else []
        n = len(convo)
        lines = [f"{type(m).__name__}: {m.content}" for m in convo]
        body = "\n".join(lines) or None
        self._emit(f"[LLM] start — {n} message-lines in context", body, ns, label)

    def on_llm_end(self, response, *, run_id, **kwargs):
        started = self._llm_starts.pop(str(run_id), None)
        ns = self._ns(run_id, kwargs)
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
            ns,
        )

    # --- Tools ---

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._tool_starts[str(run_id)] = time.time()
        ns = self._ns(run_id, kwargs)
        name = (serialized or {}).get("name", "tool")
        self._emit(f"[Tool] start — {name}", input_str or None, ns)

    def on_tool_end(self, output, *, run_id, **kwargs):
        started = self._tool_starts.pop(str(run_id), None)
        ns = self._ns(run_id, kwargs)
        secs = f"{(time.time() - started) * 1000:.0f}ms" if started else "?ms"
        body = str(output).strip() or None
        self._emit(f"[Tool] end — {secs}", body, ns)


# Shared singleton — attach on the LLM in llm/factory.py via callbacks=[...].
compact_trace = CompactTraceHandler()
