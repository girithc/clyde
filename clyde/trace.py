"""Compact structured tracing.

Replaces LangChain's default verbose/debug JSON dumps with one clean line per
event. Covers LLM calls and tool runs; the graph node flow is surfaced by the
UI.

Output goes through a configurable ``sink`` (default: ``print``). The TUI sets
the sink to route trace lines into the scrolling transcript as dimmed text,
instead of `print` corrupting the full-screen app.
"""

import time
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler


class CompactTraceHandler(BaseCallbackHandler):
    """One-line-per-event trace printer."""

    def __init__(self):
        self._llm_starts: dict[str, float] = {}
        self._tool_starts: dict[str, float] = {}
        self._sink: Callable[[str], Any] = print

    def set_sink(self, sink: Callable[[str], Any]) -> None:
        """Redirect trace lines (default ``print``; the TUI sets its own)."""
        self._sink = sink

    def _emit(self, line: str) -> None:
        self._sink(line)

    # --- LLM ---

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_starts[str(run_id)] = time.time()
        n = len(prompts[0].splitlines()) if prompts else 0
        self._emit(f"[LLM] start — {n} message-lines in context")

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
        self._emit(
            f"[LLM] end — {secs} · ↑ {in_toks} in · ↓ {out_toks} out · finish={finish}"
        )

    # --- Tools ---

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._tool_starts[str(run_id)] = time.time()
        name = (serialized or {}).get("name", "tool")
        self._emit(f"[Tool] start — {name}({input_str})")

    def on_tool_end(self, output, *, run_id, **kwargs):
        started = self._tool_starts.pop(str(run_id), None)
        secs = f"{(time.time() - started) * 1000:.0f}ms" if started else "?ms"
        preview = str(output).strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        self._emit(f"[Tool] end — {secs}: {preview}")


# Shared singleton — attach on the LLM in llm.py via callbacks=[...].
compact_trace = CompactTraceHandler()
